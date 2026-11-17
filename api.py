"""
Backend RESTful API Server bằng FastAPI & Web Application Dashboard
Đồ án Chuyên ngành Kỹ thuật Phần mềm (Chủ đề 6)
Đề tài: Phát hiện văn bản do LLM sinh sử dụng Mô hình kết hợp (Ensemble PLMs)

Các tính năng hệ thống:
1. Nạp Singleton mô hình Ensemble (AITextDetector) trong vòng đời Lifespan.
2. Cung cấp Web Dashboard SPA hiện đại phục vụ người dùng tại đường dẫn gốc `/`.
3. Tiếp nhận tải tệp tài liệu đa định dạng: .PDF, .DOCX, .TXT.
4. Cơ sở dữ liệu SQLite lưu trữ nhật ký thẩm định và thống kê KPI.
5. Xuất báo cáo chứng thực PDF chuẩn khoa học (ReportLab Engine).
6. Tương thích 100% với Chrome Extension Manifest v3 qua các endpoints API mở CORS.
"""

import os
import sys
import time
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from inference import AITextDetector
import database as db
from file_parser import extract_text_from_file
from pdf_report import generate_pdf_report

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
    print("  KHỞI ĐỘNG FASTAPI BACKEND SERVER - VERITEXT AI")
    print("=" * 70)

    # Đảm bảo CSDL SQLite đã sẵn sàng
    db.init_db()

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
        "Hệ thống Phát hiện Văn bản do Mô hình Ngôn ngữ Lớn (LLMs) sinh, "
        "kết hợp sức mạnh của 3 mô hình tiền huấn luyện: BERT, RoBERTa và Longformer (1024 tokens) "
        "với thuật toán Weighted Soft Voting tối ưu."
    ),
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Cấu hình CORS mở để Chrome Extension và Frontend web gọi API dễ dàng
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Phục vụ thư mục tĩnh cho Web Dashboard
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
if os.path.exists(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


# ==============================================================================
# SCHEMAS (PYDANTIC MODELS)
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


class WebDetectRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Văn bản gửi từ Web Dashboard")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    title: Optional[str] = Field(default=None, description="Tiêu đề hoặc tên tài liệu")


class SentenceAnalysisRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=5,
        description="Đoạn văn bản cần phân tích và bóc tách từng câu để tô màu.",
    )
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class BatchDetectRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Danh sách các đoạn văn bản cần phân tích (tối đa 50 mẫu).",
    )
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


# ==============================================================================
# 1. WEB DASHBOARD & HEALTH ENDPOINTS
# ==============================================================================

@app.get("/", tags=["Web Dashboard"])
async def serve_dashboard():
    """Phục vụ giao diện người dùng Web Dashboard chính."""
    index_file = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "project": "AI-Generated Text Detection API (Ensemble PLMs)",
        "status": "online",
        "docs_url": "/docs",
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


# ==============================================================================
# 2. CORE DETECTION ENDPOINTS
# ==============================================================================

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
            detail="Văn bản quá ngắn để có thể nhận diện tin cậy.",
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
    Phân tích diễn giải cấp câu (AI Hotspots) phục vụ Chrome Extension & Web Dashboard:
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
        overall_pred = detector_instance.predict(clean_text, threshold=req.threshold, detailed=True)
        sentences = detector_instance.analyze_sentences(clean_text, threshold=req.threshold)
        total_time_ms = round((time.time() - t0) * 1000, 2)

        red_count = sum(1 for s in sentences if s["highlight_color"] == "red")
        yellow_count = sum(1 for s in sentences if s["highlight_color"] == "yellow")
        green_count = sum(1 for s in sentences if s["highlight_color"] == "green")

        # Lưu tự động vào CSDL SQLite
        scan_id = db.save_scan(
            title=clean_text[:40] + "...",
            full_text=clean_text,
            word_count=len(clean_text.split()),
            char_count=len(clean_text),
            ai_probability=overall_pred["ai_probability"],
            verdict=overall_pred["label"],
            confidence=overall_pred["confidence"],
            inference_time_ms=total_time_ms,
            score_bert=overall_pred["model_breakdown"]["individual_models"]["bert"]["probability"],
            score_roberta=overall_pred["model_breakdown"]["individual_models"]["roberta"]["probability"],
            score_longformer=overall_pred["model_breakdown"]["individual_models"]["longformer"]["probability"],
            details_json={
                "sentences": sentences,
                "model_breakdown": overall_pred.get("model_breakdown"),
            },
        )

        return {
            "scan_id": scan_id,
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


@app.post("/api/v1/detect/web", tags=["Web Dashboard"])
async def detect_web(req: WebDetectRequest):
    """
    Endpoint tối ưu cho Web Dashboard: Phân tích toàn diện cả đoạn văn và từng câu,
    tự động lưu vào CSDL và trả về ID để hỗ trợ xuất báo cáo PDF.
    """
    return await detect_sentences(SentenceAnalysisRequest(text=req.text, threshold=req.threshold))


@app.post("/api/v1/detect/upload", tags=["Web Dashboard"])
async def detect_uploaded_file(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
):
    """
    Tiếp nhận tệp tin tải lên (.PDF, .DOCX, .TXT):
    Tự động trích xuất văn bản sạch, tiến hành phân tích Ensemble PLMs
    và lưu trữ lịch sử thẩm định kèm tên tệp gốc.
    """
    if detector_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Mô hình AI chưa sẵn sàng.",
        )

    content_bytes = await file.read()
    if not content_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tệp tin tải lên rỗng (Empty file).",
        )

    try:
        extracted_text, ext = extract_text_from_file(file.filename, content_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Lỗi đọc nội dung tệp tin: {str(e)}",
        )

    if len(extracted_text.split()) < 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nội dung văn bản trong tệp quá ngắn (yêu cầu ít nhất 5 từ).",
        )

    t0 = time.time()
    overall_pred = detector_instance.predict(extracted_text, threshold=threshold, detailed=True)
    sentences = detector_instance.analyze_sentences(extracted_text, threshold=threshold)
    total_time_ms = round((time.time() - t0) * 1000, 2)

    red_count = sum(1 for s in sentences if s["highlight_color"] == "red")
    yellow_count = sum(1 for s in sentences if s["highlight_color"] == "yellow")
    green_count = sum(1 for s in sentences if s["highlight_color"] == "green")

    # Lưu vào SQLite
    scan_id = db.save_scan(
        title=f"Tài liệu: {file.filename}",
        filename=file.filename,
        full_text=extracted_text,
        word_count=len(extracted_text.split()),
        char_count=len(extracted_text),
        ai_probability=overall_pred["ai_probability"],
        verdict=overall_pred["label"],
        confidence=overall_pred["confidence"],
        inference_time_ms=total_time_ms,
        score_bert=overall_pred["model_breakdown"]["individual_models"]["bert"]["probability"],
        score_roberta=overall_pred["model_breakdown"]["individual_models"]["roberta"]["probability"],
        score_longformer=overall_pred["model_breakdown"]["individual_models"]["longformer"]["probability"],
        details_json={
            "sentences": sentences,
            "model_breakdown": overall_pred.get("model_breakdown"),
        },
    )

    return {
        "scan_id": scan_id,
        "filename": file.filename,
        "extracted_text": extracted_text,
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


@app.post("/api/v1/detect/batch", tags=["Detection"])
async def detect_batch(req: BatchDetectRequest):
    """Phân tích danh sách hàng loạt văn bản (Batch Processing)."""
    if detector_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Mô hình AI chưa sẵn sàng.",
        )

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


# ==============================================================================
# 3. HISTORY, STATISTICS & AUDIT REPORT ENDPOINTS
# ==============================================================================

@app.get("/api/v1/history", tags=["History"])
async def get_scan_history(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    """Lấy danh sách các lần thẩm định trước đó từ CSDL SQLite."""
    items = db.get_history(limit=limit, offset=offset)
    return {"total": len(items), "history": items}


@app.get("/api/v1/history/{scan_id}", tags=["History"])
async def get_scan_detail(scan_id: int):
    """Lấy thông tin chi tiết một lượt quét theo ID."""
    item = db.get_scan_by_id(scan_id)
    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Không tìm thấy lượt quét này.")
    return item


@app.delete("/api/v1/history/{scan_id}", tags=["History"])
async def delete_scan_record(scan_id: int):
    """Xóa một bản ghi lịch sử thẩm định khỏi CSDL."""
    success = db.delete_scan(scan_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Không tìm thấy mục để xóa.")
    return {"success": True, "deleted_id": scan_id}


@app.get("/api/v1/stats", tags=["History"])
async def get_dashboard_stats():
    """Lấy các chỉ số KPI thống kê tổng hợp phục vụ hiển thị trên Dashboard."""
    return db.get_statistics()


@app.get("/api/v1/report/pdf/{scan_id}", tags=["Audit Report"])
async def export_pdf_report(scan_id: int):
    """
    Sinh và tải về Báo Cáo Giám Định Nguồn Gốc Văn Bản dạng PDF (AI Audit Certificate)
    có đánh số trang, đóng dấu phân rã mô hình và bảng cấp câu AI Hotspots.
    """
    scan_data = db.get_scan_by_id(scan_id)
    if not scan_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Không tìm thấy dữ liệu lượt quét.")

    try:
        pdf_bytes = generate_pdf_report(scan_data)
        filename = f"AI_Audit_Report_{scan_id:06d}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"inline; filename={filename}",
                "Content-Type": "application/pdf",
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khởi tạo tệp PDF: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"\n[*] Đang khởi chạy Uvicorn Server tại http://{host}:{port} ...")
    uvicorn.run("api:app", host=host, port=port, reload=False)
