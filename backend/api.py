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
from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form, Query, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from inference import AITextDetector
import database as db
import auth
from auth import get_current_user, get_current_user_optional, require_role
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

    # Tự động tìm thư mục chứa models_dir ('output')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if os.environ.get("MODELS_DIR"):
        models_dir = os.environ.get("MODELS_DIR")
    elif os.path.exists(os.path.join(base_dir, "output")):
        models_dir = os.path.join(base_dir, "output")
    elif os.path.exists(os.path.join(base_dir, "..", "output")):
        models_dir = os.path.join(base_dir, "..", "output")
    else:
        models_dir = os.path.join(base_dir, "output")
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSSIBLE_WEB_DIRS = [
    os.path.abspath(os.path.join(BASE_DIR, "..", "frontend", "web")),
    os.path.abspath(os.path.join(BASE_DIR, "..", "frontend")),
    os.path.abspath(os.path.join(BASE_DIR, "web")),
    os.path.abspath(os.path.join(BASE_DIR, "..", "web")),
]
WEB_DIR = None
for d in POSSIBLE_WEB_DIRS:
    if os.path.exists(os.path.join(d, "index.html")):
        WEB_DIR = d
        break
if WEB_DIR is None:
    WEB_DIR = POSSIBLE_WEB_DIRS[0]

if os.path.exists(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


# ==============================================================================
# SCHEMAS (PYDANTIC MODELS)
# ==============================================================================

class DetectRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        description="Đoạn văn bản cần kiểm tra (tiếng Anh hoặc tiếng Việt).",
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
    language: Optional[str] = Field(
        default=None,
        description="Tùy chọn chỉ định ngôn ngữ ('en' hoặc 'vi'). Mặc định None sẽ tự động nhận diện."
    )


class WebDetectRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Văn bản gửi từ Web Dashboard")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    title: Optional[str] = Field(default=None, description="Tiêu đề hoặc tên tài liệu")
    language: Optional[str] = Field(default=None, description="Tùy chọn chỉ định ngôn ngữ ('en' hoặc 'vi')")


class SentenceAnalysisRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=5,
        description="Đoạn văn bản cần phân tích và bóc tách từng câu để tô màu.",
    )
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    language: Optional[str] = Field(default=None, description="Tùy chọn chỉ định ngôn ngữ ('en' hoặc 'vi')")


class BatchDetectRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Danh sách các đoạn văn bản cần phân tích (tối đa 50 mẫu).",
    )
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class RegisterRequest(BaseModel):
    email: str = Field(..., description="Email đăng ký", examples=["sinhvien@veritext.ai"])
    password: str = Field(..., min_length=6, description="Mật khẩu tối thiểu 6 ký tự", examples=["123456"])
    full_name: str = Field(..., min_length=2, description="Họ và tên đầy đủ", examples=["Nguyễn Văn A"])
    role: str = Field(default="student", description="Vai trò: 'student' (Sinh viên) hoặc 'teacher' (Giảng viên)", examples=["student"])


class LoginRequest(BaseModel):
    email: str = Field(..., description="Email đăng nhập", examples=["sinhvien@veritext.ai"])
    password: str = Field(..., description="Mật khẩu", examples=["123456"])


# ==============================================================================
# 0. AUTHENTICATION & USER MANAGEMENT (SQL SERVER + JWT)
# ==============================================================================

@app.post("/api/v1/auth/register", tags=["Authentication"])
async def register_account(req: RegisterRequest):
    """Đăng ký tài khoản người dùng mới (Sinh viên hoặc Giảng viên) vào Microsoft SQL Server."""
    try:
        user = db.create_user(
            email=req.email,
            password=req.password,
            full_name=req.full_name,
            role=req.role,
        )
        token = auth.create_access_token({
            "id": user["id"],
            "email": user["email"],
            "full_name": user["full_name"],
            "role": user["role"],
        })
        return {
            "success": True,
            "message": "Đăng ký tài khoản thành công!",
            "access_token": token,
            "token_type": "bearer",
            "user": user,
        }
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Lỗi đăng ký: {str(e)}")


@app.post("/api/v1/auth/login", tags=["Authentication"])
async def login_account(req: LoginRequest):
    """Đăng nhập tài khoản, xác thực mật khẩu qua bcrypt và trả về JWT Access Token."""
    user = db.authenticate_user(req.email, req.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email hoặc mật khẩu không chính xác!",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = auth.create_access_token({
        "id": user["id"],
        "email": user["email"],
        "full_name": user["full_name"],
        "role": user["role"],
    })
    return {
        "success": True,
        "message": "Đăng nhập thành công!",
        "access_token": token,
        "token_type": "bearer",
        "user": user,
    }


@app.get("/api/v1/auth/me", tags=["Authentication"])
async def get_my_profile(current_user: Dict[str, Any] = Depends(auth.get_current_user)):
    """Lấy thông tin tài khoản đang đăng nhập từ JWT Token."""
    user = db.get_user_by_id(current_user["id"])
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Người dùng không tồn tại.")
    return {"user": user}


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
@app.get("/api/v1/health", tags=["General"])
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
        "languages_supported": ["en", "vi"],
        "models_loaded": {
            "bert": is_ready and detector_instance.model_bert is not None,
            "roberta": is_ready and detector_instance.model_roberta is not None,
            "longformer": is_ready and detector_instance.model_longformer is not None,
            "phobert": is_ready and detector_instance.model_phobert is not None,
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
    và điểm số đóng góp của từng mô hình (BERT, RoBERTa, Longformer hoặc PhoBERT).
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
            force_language=req.language,
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi trong quá trình suy luận mô hình: {str(e)}",
        )


@app.post("/api/v1/detect/sentences", tags=["Detection"])
async def detect_sentences(
    req: SentenceAnalysisRequest,
    current_user: Optional[Dict[str, Any]] = Depends(auth.get_current_user_optional),
):
    """
    Phân tích diễn giải cấp câu (AI Hotspots) phục vụ Chrome Extension & Web Dashboard:
    Bóc tách đoạn văn bản thành từng câu độc lập, tính toán xác suất AI
    và gán nhãn màu cảnh báo (🟢 Xanh: Người viết, 🟡 Vàng: Nghi vấn, 🔴 Đỏ: Chắc chắn AI).
    Hỗ trợ tự động nhận diện Song ngữ Anh - Việt.
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
        overall_pred = detector_instance.predict(
            clean_text,
            threshold=req.threshold,
            detailed=True,
            force_language=req.language,
        )
        sentences = detector_instance.analyze_sentences(
            clean_text,
            threshold=req.threshold,
            force_language=req.language,
        )
        total_time_ms = round((time.time() - t0) * 1000, 2)

        red_count = sum(1 for s in sentences if s["highlight_color"] == "red")
        yellow_count = sum(1 for s in sentences if s["highlight_color"] == "yellow")
        green_count = sum(1 for s in sentences if s["highlight_color"] == "green")

        # Lưu tự động vào CSDL Microsoft SQL Server (VeriTextAI)
        ind_models = overall_pred.get("model_breakdown", {}).get("individual_models", {})
        score_bert = float(ind_models.get("bert", {}).get("probability", overall_pred["ai_probability"]))
        score_roberta = float(ind_models.get("roberta", {}).get("probability", overall_pred["ai_probability"]))
        score_longformer = float(ind_models.get("longformer", {}).get("probability", overall_pred["ai_probability"]))

        scan_id = db.save_scan(
            title=clean_text[:40] + "...",
            full_text=clean_text,
            word_count=len(clean_text.split()),
            char_count=len(clean_text),
            ai_probability=overall_pred["ai_probability"],
            verdict=overall_pred["label"],
            confidence=overall_pred["confidence"],
            inference_time_ms=total_time_ms,
            score_bert=score_bert,
            score_roberta=score_roberta,
            score_longformer=score_longformer,
            details_json={
                "language": overall_pred.get("language", "en"),
                "language_label": overall_pred.get("language_label", "Tiếng Anh"),
                "detected_model": overall_pred.get("detected_model", ""),
                "sentences": sentences,
                "model_breakdown": overall_pred.get("model_breakdown"),
            },
            user_id=current_user["id"] if current_user else None,
            language=overall_pred.get("language", "en"),
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
async def detect_web(
    req: WebDetectRequest,
    current_user: Optional[Dict[str, Any]] = Depends(auth.get_current_user_optional),
):
    """
    Endpoint tối ưu cho Web Dashboard: Phân tích toàn diện cả đoạn văn và từng câu,
    tự động lưu vào CSDL và trả về ID để hỗ trợ xuất báo cáo PDF.
    """
    return await detect_sentences(
        SentenceAnalysisRequest(
            text=req.text,
            threshold=req.threshold,
            language=req.language,
        ),
        current_user=current_user,
    )


@app.post("/api/v1/detect/upload", tags=["Web Dashboard"])
async def detect_uploaded_file(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    current_user: Optional[Dict[str, Any]] = Depends(auth.get_current_user_optional),
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

    ind_models = overall_pred.get("model_breakdown", {}).get("individual_models", {})
    score_bert = float(ind_models.get("bert", {}).get("probability", overall_pred["ai_probability"]))
    score_roberta = float(ind_models.get("roberta", {}).get("probability", overall_pred["ai_probability"]))
    score_longformer = float(ind_models.get("longformer", {}).get("probability", overall_pred["ai_probability"]))

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
        score_bert=score_bert,
        score_roberta=score_roberta,
        score_longformer=score_longformer,
        details_json={
            "language": overall_pred.get("language", "en"),
            "language_label": overall_pred.get("language_label", "Tiếng Anh"),
            "detected_model": overall_pred.get("detected_model", ""),
            "sentences": sentences,
            "model_breakdown": overall_pred.get("model_breakdown"),
        },
        user_id=current_user["id"] if current_user else None,
        language=overall_pred.get("language", "en"),
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
async def get_scan_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: Optional[Dict[str, Any]] = Depends(auth.get_current_user_optional),
):
    """Lấy danh sách các lần thẩm định trước đó từ CSDL Microsoft SQL Server theo phân quyền."""
    user_id = current_user["id"] if current_user else None
    role = current_user["role"] if current_user else None
    items = db.get_history(limit=limit, offset=offset, user_id=user_id, role=role)
    return {"total": len(items), "history": items, "current_user": current_user}


@app.get("/api/v1/history/{scan_id}", tags=["History"])
async def get_scan_detail(
    scan_id: int,
    current_user: Optional[Dict[str, Any]] = Depends(auth.get_current_user_optional),
):
    """Lấy thông tin chi tiết một lượt quét theo ID."""
    user_id = current_user["id"] if current_user else None
    role = current_user["role"] if current_user else None
    item = db.get_scan_by_id(scan_id, user_id=user_id, role=role)
    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Không tìm thấy lượt quét này hoặc bạn không có quyền xem.")
    return item


@app.delete("/api/v1/history/{scan_id}", tags=["History"])
async def delete_scan_record(
    scan_id: int,
    current_user: Optional[Dict[str, Any]] = Depends(auth.get_current_user_optional),
):
    """Xóa một bản ghi lịch sử thẩm định khỏi CSDL."""
    user_id = current_user["id"] if current_user else None
    role = current_user["role"] if current_user else None
    success = db.delete_scan(scan_id, user_id=user_id, role=role)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Không tìm thấy mục để xóa hoặc bạn không có quyền.")
    return {"success": True, "deleted_id": scan_id}


@app.get("/api/v1/stats", tags=["History"])
async def get_dashboard_stats(
    current_user: Optional[Dict[str, Any]] = Depends(auth.get_current_user_optional),
):
    """Lấy các chỉ số KPI thống kê tổng hợp phục vụ hiển thị trên Dashboard."""
    user_id = current_user["id"] if current_user else None
    role = current_user["role"] if current_user else None
    return db.get_statistics(user_id=user_id, role=role)


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
