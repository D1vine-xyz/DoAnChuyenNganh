"""
Module OCR Ngoại Tuyến Song Ngữ (Tiếng Việt & Tiếng Anh) cho Ảnh Chụp & PDF Scan
Đồ án Chuyên ngành Kỹ thuật Phần mềm (Chủ đề 6): VeriText AI

Sử dụng PyMuPDF (render trang PDF thành ảnh chất lượng cao) kết hợp EasyOCR (CRAFT + CRNN Latin/Việt)
chạy hoàn toàn cục bộ trên CPU/GPU mà không cần API trả phí bên ngoài.
"""

import io
import os
import re
import threading
import unicodedata
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageOps

MAX_OCR_PDF_PAGES = int(os.environ.get("MAX_OCR_PDF_PAGES", "15"))
MAX_IMAGE_SIDE_PX = 2600
MIN_IMAGE_SIDE_PX = 800

_reader_lock = threading.Lock()
_easyocr_reader = None


def _get_easyocr_reader():
    """Khởi tạo Lazy Singleton EasyOCR(['vi', 'en']) an toàn đa luồng."""
    global _easyocr_reader
    if _easyocr_reader is not None:
        return _easyocr_reader

    with _reader_lock:
        if _easyocr_reader is None:
            import torch
            import easyocr

            use_gpu = bool(torch.cuda.is_available())
            _easyocr_reader = easyocr.Reader(["vi", "en"], gpu=use_gpu, verbose=False)
    return _easyocr_reader


def _prepare_pil_for_ocr(img: Image.Image) -> np.ndarray:
    """
    Chuẩn hóa ảnh đầu vào trước khi OCR:
    - Tự xoay đúng chiều theo EXIF (ảnh chụp điện thoại).
    - Ghép nền trắng nếu ảnh PNG có kênh Alpha trong suốt.
    - Phóng nhẹ ảnh chữ nhỏ hoặc thu gọn ảnh quá lớn để tối ưu tốc độ CPU & độ nét dấu tiếng Việt.
    """
    img = ImageOps.exif_transpose(img)

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        white_bg = Image.new("RGB", rgba.size, (255, 255, 255))
        white_bg.paste(rgba, mask=rgba.split()[3])
        img = white_bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    max_side = max(w, h)
    if max_side > MAX_IMAGE_SIDE_PX:
        scale = MAX_IMAGE_SIDE_PX / float(max_side)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    elif max_side < MIN_IMAGE_SIDE_PX and max_side > 0:
        scale = min(2.0, MIN_IMAGE_SIDE_PX / float(max_side))
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)

    return np.asarray(img)


def _clean_ocr_text(chunks: List[str]) -> str:
    """Chuẩn hóa Unicode NFC và làm sạch khoảng trắng dư từ đầu ra OCR."""
    joined = "\n".join(str(c).strip() for c in chunks if str(c).strip())
    normalized = unicodedata.normalize("NFC", joined)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def ocr_image_bytes(image_bytes: bytes) -> Dict[str, Any]:
    """Trích xuất văn bản tiếng Việt / tiếng Anh từ dữ liệu nhị phân của tệp ảnh."""
    if not image_bytes:
        raise ValueError("Tệp hình ảnh rỗng (0 bytes).")

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img_arr = _prepare_pil_for_ocr(img)
    except Exception as e:
        raise ValueError(f"Không thể đọc tệp hình ảnh hợp lệ: {e}") from e

    reader = _get_easyocr_reader()
    raw_lines = reader.readtext(img_arr, detail=0, paragraph=True)
    text = _clean_ocr_text(raw_lines)
    if not text:
        raise ValueError("Không nhận diện được nội dung chữ rõ ràng trong hình ảnh.")

    return {
        "text": text,
        "ocr_used": True,
        "ocr_pages": 1,
        "ocr_total_pages": 1,
        "ocr_note": "Đã nhận diện văn bản từ hình ảnh bằng OCR ngoại tuyến (EasyOCR Song ngữ Anh-Việt).",
    }


def pdf_has_images(pdf_bytes: bytes) -> bool:
    """Kiểm tra nhanh tệp PDF có chứa hình ảnh nhúng hay không (phân biệt PDF scan)."""
    import pymupdf

    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            for idx in range(min(len(doc), 3)):
                if doc[idx].get_images(full=False):
                    return True
    except Exception:
        return False
    return False


def ocr_pdf_bytes(pdf_bytes: bytes, max_pages: int = MAX_OCR_PDF_PAGES) -> Dict[str, Any]:
    """
    Render từng trang PDF ảnh scan thành ảnh 144 DPI qua PyMuPDF rồi trích xuất chữ bằng EasyOCR.
    Giới hạn mặc định tối đa `max_pages` trang để tránh quá tải CPU với tài liệu dài.
    """
    import pymupdf

    reader = _get_easyocr_reader()
    page_texts: List[str] = []

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        total_pages = len(doc)
        if total_pages == 0:
            raise ValueError("Tệp PDF không có trang nào.")

        pages_to_scan = min(total_pages, max_pages)
        matrix = pymupdf.Matrix(2.0, 2.0)  # ~144 DPI: cân bằng độ nét dấu tiếng Việt và tốc độ CPU

        for page_idx in range(pages_to_scan):
            page = doc[page_idx]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_arr = _prepare_pil_for_ocr(pil_img)
            lines = reader.readtext(img_arr, detail=0, paragraph=True)
            cleaned = _clean_ocr_text(lines)
            if cleaned:
                page_texts.append(cleaned)

    full_text = "\n\n".join(page_texts).strip()
    if not full_text:
        raise ValueError(
            "Không nhận diện được chữ từ tệp PDF ảnh scan. Vui lòng kiểm tra độ nét của bản scan."
        )

    note = f"Đã trích xuất bằng OCR ngoại tuyến từ {pages_to_scan}/{total_pages} trang PDF ảnh scan."
    if total_pages > pages_to_scan:
        note += f" (Giới hạn tối đa {pages_to_scan} trang đầu để tối ưu tốc độ)."

    return {
        "text": full_text,
        "ocr_used": True,
        "ocr_pages": pages_to_scan,
        "ocr_total_pages": total_pages,
        "ocr_note": note,
    }
