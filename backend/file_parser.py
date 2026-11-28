"""
Module Tiền xử lý & Trích xuất Văn bản từ File Upload (.txt, .docx, .doc, .pdf, và Ảnh/PDF Scan qua OCR)
Đồ án Chuyên ngành Kỹ thuật Phần mềm (Chủ đề 6): VeriText AI
"""

import io
import os
import re
from typing import Any, Dict, Tuple

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def extract_text_with_meta(filename: str, content_bytes: bytes) -> Dict[str, Any]:
    """
    Trích xuất văn bản sạch kèm siêu dữ liệu OCR từ nội dung nhị phân của tệp tin.
    Hỗ trợ: .txt, .docx, .doc, .pdf (cả bản chữ gốc lẫn PDF ảnh scan qua OCR), .png, .jpg, .jpeg, .webp, .bmp, .tif.
    """
    fn_lower = (filename or "").lower()

    if fn_lower.endswith(".txt"):
        # Ưu tiên utf-8-sig trước để tự động loại bỏ BOM hợp lệ của Windows Notepad (tránh báo nhầm Anti-Bypass)
        try:
            text = content_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                text = content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = content_bytes.decode("latin-1", errors="replace")
        return {
            "text": text.strip(),
            "extension": "txt",
            "ocr_used": False,
            "ocr_pages": 0,
            "ocr_total_pages": 0,
            "ocr_note": "",
        }

    elif fn_lower.endswith(".docx"):
        import docx

        doc_stream = io.BytesIO(content_bytes)
        doc = docx.Document(doc_stream)
        full_text = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                full_text.append(text)
        # Bổ sung trích xuất nội dung nằm trong bảng Word (rất phổ biến ở mẫu báo cáo/tiểu luận)
        seen_cells = set()
        for table in doc.tables:
            for row in table.rows:
                row_parts = []
                for cell in row.cells:
                    cell_id = id(cell._tc)
                    if cell_id in seen_cells:
                        continue
                    seen_cells.add(cell_id)
                    c_text = cell.text.strip()
                    if c_text:
                        row_parts.append(c_text)
                if row_parts:
                    full_text.append(" ".join(row_parts))
        return {
            "text": "\n\n".join(full_text).strip(),
            "extension": "docx",
            "ocr_used": False,
            "ocr_pages": 0,
            "ocr_total_pages": 0,
            "ocr_note": "",
        }

    elif fn_lower.endswith(".pdf"):
        import pypdf
        import ocr_engine

        pdf_stream = io.BytesIO(content_bytes)
        reader = pypdf.PdfReader(pdf_stream)
        if reader.is_encrypted:
            try:
                if reader.decrypt("") == 0:
                    raise ValueError("Tệp PDF đang được bảo vệ bằng mật khẩu, không thể trích xuất văn bản.")
            except ValueError:
                raise
            except Exception:
                raise ValueError("Tệp PDF được mã hóa bằng mật khẩu, vui lòng gỡ mật khẩu trước khi tải lên.")

        total_pages = len(reader.pages)
        full_text = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                full_text.append(t.strip())
        extracted = "\n\n".join(full_text).strip()

        # Tự động kích hoạt OCR nếu PDF không có lớp text hoặc chỉ là chữ/số trang rời trên trang ảnh scan
        needs_ocr = (not extracted) or (
            len(extracted.split()) < 8 and ocr_engine.pdf_has_images(content_bytes)
        )
        if needs_ocr:
            ocr_res = ocr_engine.ocr_pdf_bytes(content_bytes)
            return {
                "text": ocr_res["text"],
                "extension": "pdf",
                "ocr_used": True,
                "ocr_pages": ocr_res["ocr_pages"],
                "ocr_total_pages": ocr_res["ocr_total_pages"],
                "ocr_note": ocr_res["ocr_note"],
            }

        return {
            "text": extracted,
            "extension": "pdf",
            "ocr_used": False,
            "ocr_pages": total_pages,
            "ocr_total_pages": total_pages,
            "ocr_note": "",
        }

    elif fn_lower.endswith(IMAGE_EXTENSIONS):
        import ocr_engine

        ext = os.path.splitext(fn_lower)[1].lstrip(".") or "image"
        ocr_res = ocr_engine.ocr_image_bytes(content_bytes)
        return {
            "text": ocr_res["text"],
            "extension": ext,
            "ocr_used": True,
            "ocr_pages": ocr_res["ocr_pages"],
            "ocr_total_pages": ocr_res["ocr_total_pages"],
            "ocr_note": ocr_res["ocr_note"],
        }

    elif fn_lower.endswith(".doc"):
        import olefile

        try:
            ole_stream = io.BytesIO(content_bytes)
            with olefile.OleFileIO(ole_stream) as ole:
                if ole.exists("WordDocument"):
                    stream = ole.openstream("WordDocument").read()
                    decoded = stream.decode("utf-16-le", errors="ignore")
                    paragraphs = [p.strip() for p in re.split(r"[\r\n\x00-\x1f\x7f-\x9f]+", decoded) if len(p.strip()) > 10]
                    # Lọc bỏ các đoạn chứa ký tự rác nhị phân (binary junk)
                    cleaned = []
                    for p in paragraphs:
                        valid_chars = sum(
                            1 for c in p if c.isalnum() or c.isspace() or c in ".,;:!?()-_'\"/\\@#%&*+=<>[]{}"
                        )
                        if len(p) > 0 and (valid_chars / len(p)) >= 0.70:
                            cleaned.append(p)
                    if cleaned:
                        return {
                            "text": "\n\n".join(cleaned).strip(),
                            "extension": "doc",
                            "ocr_used": False,
                            "ocr_pages": 0,
                            "ocr_total_pages": 0,
                            "ocr_note": "",
                        }
        except Exception:
            pass

        # Fallback đọc text thô nếu olefile không trích xuất được
        text = content_bytes.decode("latin-1", errors="ignore")
        clean = re.findall(r'[\w\s\.,;:!?\(\)\-\'\"]{10,}', text)
        return {
            "text": "\n\n".join(clean).strip(),
            "extension": "doc",
            "ocr_used": False,
            "ocr_pages": 0,
            "ocr_total_pages": 0,
            "ocr_note": "",
        }

    else:
        raise ValueError(
            f"Định dạng tệp '{filename}' không được hỗ trợ. "
            "Chỉ chấp nhận .txt, .docx, .doc, .pdf (cả bản scan) và ảnh .png, .jpg, .jpeg, .webp, .bmp."
        )


def extract_text_from_file(filename: str, content_bytes: bytes) -> Tuple[str, str]:
    """
    Trích xuất chuỗi văn bản sạch từ nội dung nhị phân của tệp tin.
    Giữ nguyên chữ ký (extracted_text, detected_extension) để tương thích ngược toàn bộ hệ thống.
    """
    meta = extract_text_with_meta(filename, content_bytes)
    return meta["text"], meta["extension"]
