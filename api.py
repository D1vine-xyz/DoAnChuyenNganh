"""
Backend RESTful API Server bằng FastAPI (Ngày 14)
Đồ án Chuyên ngành: Phát hiện văn bản do LLM sinh sử dụng Mô hình kết hợp (Ensemble PLMs)

Các tính năng chính:
- Nạp Singleton mô hình Ensemble (`AITextDetector`) tại sự kiện khởi động server (lifespan).
- Cấu hình CORS Middleware mở để Chrome Extension và Frontend web gọi dễ dàng.
- Cung cấp các endpoints chuẩn RESTful:
  * GET  /health               : Kiểm tra sức khỏe hệ thống và tài nguyên phần cứng.
  * POST /api/v1/detect        : Dự đoán văn bản đơn lẻ (Xác suất AI, nhãn, độ tin cậy, phân rã mô hình).
  * POST /api/v1/detect/sentences : Phân tích cấp câu (Highlight mã màu Xanh/Vàng/Đỏ phục vụ Web Extension).
  * POST /api/v1/detect/batch  : Phân tích danh sách hàng loạt văn bản.
- Tự động sinh tài liệu Swagger UI chuẩn tại `/docs` và ReDoc tại `/redoc`.
"""

import os
import sys
import time
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from inference import AITextDetector

# Biến toàn cục lưu trữ Singleton detector
detector_instance: Optional[AITextDetector] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Quản lý vòng đời ứng dụng FastAPI:
    Nạp sẵn các mô hình ngôn ngữ lớn vào RAM/VRAM khi khởi động server,
    tránh nạp lại cho mỗi request gây chậm trễ.
    """
    global detector_instance
    print("\n" + "=" * 70)
    print("  KHỞI ĐỘNG FASTAPI BACKEND SERVER - AI TEXT DETECTOR")
    print("=" * 70)

    models_dir = os.environ.get("MODELS_DIR", "output")
    try:
        t0 = time.time()
        detector_instance = AITextDetector(models_dir=models_dir)
        load_time = time.time() - t0
        print(f"[v] Đã nạp thành công bộ suy luận Ensemble trong {load_time:.2f}s.")
    except Exception as e:
        print(f"[!] CẢNH BÁO: Lỗi khởi tạo mô hình: {e}")
        detector_instance = None

    yield

    print("\n[*] Đang tắt FastAPI Backend Server. Giải phóng tài nguyên...")
    detector_instance = None


app = FastAPI(
    title="AI-Generated Text Detection API (Ensemble PLMs)",
    description=(
        "API phát hiện văn bản do các Mô hình Ngôn ngữ Lớn (ChatGPT, Gemini, LLMs) sinh ra, "
        "kết hợp sức mạnh của 3 mô hình tiền huấn luyện: BERT, RoBERTa và Longformer (1024 tokens) "
        "với chiến lược Weighted Soft Voting tối ưu."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Cấu hình CORS để tiện ích Chrome Extension và Website có thể gọi API mà không bị chặn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Hỗ trợ chrome-extension://, localhost, v.v.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# ĐỊNH NGHĨA SCHEMAS (PYDANTIC MODELS)
# ==============================================================================

class DetectRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        description="Đoạn văn bản cần kiểm tra (tiếng Anh).",
        examples=["Artificial intelligence has transformed modern software development and academic research."]
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Ngưỡng phân loại nhị phân (mặc định 0.5).",
    )
    detailed: bool = Field(
        default=True,
        description="Có trả về phân rã điểm số chi tiết của từng mô hình thành phần hay không.",
    )


class SentenceAnalysisRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=5,
        description="Đoạn văn bản cần phân tích và bóc tách từng câu để tô màu.",
        examples=[
            "Yesterday evening, my family gathered for dinner and we watched a local football game. "
            "However, artificial intelligence algorithms continue to synthesize text with high coherence. "
            "We enjoyed our meal together before heading to sleep."
        ]
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Ngưỡng quyết định nhị phân.",
    )


class BatchDetectRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Danh sách các đoạn văn bản cần phân tích (tối đa 50 mẫu / request).",
    )
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


# ==============================================================================
# CÁC API ENDPOINTS
# ==============================================================================

@app.get("/", tags=["General"])
async def root():
    """Endpoint gốc chào mừng và dẫn tới tài liệu API."""
    return {
        "project": "AI-Generated Text Detection API (Ensemble PLMs)",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_check": "/health",
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Kiểm tra tình trạng hoạt động của Server và các mô hình đã nạp."""
    is_ready = detector_instance is not None
    device_info = "cpu"
    if is_ready:
        device_info = str(detector_instance.device)
        if detector_instance.device.type == "cuda":
            device_info += f" ({torch.cuda.get_device_name(0)})"

    return {
        "status": "healthy" if is_ready else "initializing_or_error",
        "ready": is_ready,
        "device": device_info,
        "models_loaded": {
            "bert": is_ready and detector_instance.model_bert is not None,
            "roberta": is_ready and detector_instance.model_roberta is not None,
            "longformer": is_ready and detector_instance.model_longformer is not None,
            "stacking_meta_learner": is_ready and detector_instance.meta_classifier is not None,
        },
        "ensemble_weights": {
            "longformer": round(float(detector_instance.weights[2]), 4) if is_ready else None,
            "bert": round(float(detector_instance.weights[0]), 4) if is_ready else None,
            "roberta": round(float(detector_instance.weights[1]), 4) if is_ready else None,
        } if is_ready else None,
    }


@app.post("/api/v1/detect", tags=["Detection"])
async def detect_text(req: DetectRequest):
    """
    Phân tích một đoạn văn bản thô:
    Trả về xác suất văn bản do AI sinh (%), nhãn nhị phân, mức độ tin cậy
    và điểm số đóng góp của từng mô hình (BERT, RoBERTa, Longformer).
    """
    if detector_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Mô hình AI chưa sẵn sàng hoặc đang khởi tạo. Vui lòng thử lại sau.",
        )

    clean_text = req.text.strip()
    if len(clean_text) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Văn bản quá ngắn để có thể nhận diện tin cậy (yêu cầu ít nhất 3 ký tự).",
        )

    try:
        result = detector_instance.predict(
            text=clean_text,
            threshold=req.threshold,
            detailed=req.detailed,
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi trong quá trình suy luận mô hình: {str(e)}",
        )


@app.post("/api/v1/detect/sentences", tags=["Detection"])
async def detect_sentences(req: SentenceAnalysisRequest):
    """
    Phân tích diễn giải cấp câu (AI Hotspots) phục vụ Chrome Extension:
    Bóc tách đoạn văn bản thành từng câu độc lập, tính toán xác suất AI
    và gán nhãn màu cảnh báo (🟢 Xanh: Người viết, 🟡 Vàng: Nghi vấn, 🔴 Đỏ: Chắc chắn AI).
    """
    if detector_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Mô hình AI chưa sẵn sàng.",
        )

    clean_text = req.text.strip()
    if len(clean_text) < 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Văn bản quá ngắn để bóc tách câu.",
        )

    try:
        t0 = time.time()
        # 1. Dự đoán tổng quan toàn văn bản
        overall_pred = detector_instance.predict(clean_text, threshold=req.threshold, detailed=False)
        # 2. Phân tích chi tiết từng câu
        sentences = detector_instance.analyze_sentences(clean_text, threshold=req.threshold)
        total_time_ms = round((time.time() - t0) * 1000, 2)

        # Thống kê nhanh số lượng câu theo từng mức độ
        red_count = sum(1 for s in sentences if s["highlight_color"] == "red")
        yellow_count = sum(1 for s in sentences if s["highlight_color"] == "yellow")
        green_count = sum(1 for s in sentences if s["highlight_color"] == "green")

        return {
            "overall": overall_pred,
            "sentence_summary": {
                "total_sentences": len(sentences),
                "ai_hotspots_red": red_count,
                "ai_suspicious_yellow": yellow_count,
                "human_green": green_count,
            },
            "sentences": sentences,
            "total_processing_time_ms": total_time_ms,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi phân tích cấp câu: {str(e)}",
        )


@app.post("/api/v1/detect/batch", tags=["Detection"])
async def detect_batch(req: BatchDetectRequest):
    """
    Phân tích danh sách hàng loạt văn bản (Batch Processing):
    Hỗ trợ xử lý tới 50 văn bản trong cùng một request.
    """
    if detector_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Mô hình AI chưa sẵn sàng.",
        )

    if not req.texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Danh sách văn bản không được rỗng.",
        )

    try:
        t0 = time.time()
        results = []
        for idx, text in enumerate(req.texts):
            clean = text.strip()
            if clean:
                pred = detector_instance.predict(clean, threshold=req.threshold, detailed=False)
                results.append({"index": idx, "success": True, "result": pred})
            else:
                results.append({"index": idx, "success": False, "error": "Văn bản rỗng."})

        total_time_ms = round((time.time() - t0) * 1000, 2)
        return {
            "total_items": len(req.texts),
            "processed_items": len(results),
            "total_time_ms": total_time_ms,
            "results": results,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi xử lý hàng loạt: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"\n[*] Đang khởi chạy Uvicorn Server tại http://{host}:{port} ...")
    uvicorn.run("api:app", host=host, port=port, reload=False)

