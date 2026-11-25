"""
Module Tiền xử lý & Trích xuất Văn bản từ File Upload (.txt, .docx, .pdf)
Đồ án Chuyên ngành Kỹ thuật Phần mềm: Hệ thống Phát hiện Văn bản do LLM sinh (Ensemble PLMs)
"""

import io
from typing import Tuple


def extract_text_from_file(filename: str, content_bytes: bytes) -> Tuple[str, str]:
    """
    Trích xuất chuỗi văn bản sạch từ nội dung nhị phân của tệp tin.
    Hỗ trợ các định dạng: .txt, .docx, .pdf
    :return: (extracted_text, detected_extension)
    """
    fn_lower = filename.lower()

    if fn_lower.endswith(".txt"):
        # Ưu tiên utf-8-sig trước để tự động loại bỏ BOM hợp lệ của Windows Notepad (tránh báo nhầm Anti-Bypass)
        try:
            text = content_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                text = content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = content_bytes.decode("latin-1", errors="replace")
        return text.strip(), "txt"

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
        return "\n\n".join(full_text).strip(), "docx"

    elif fn_lower.endswith(".pdf"):
        import pypdf
        pdf_stream = io.BytesIO(content_bytes)
        reader = pypdf.PdfReader(pdf_stream)
        full_text = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                full_text.append(t.strip())
        return "\n\n".join(full_text).strip(), "pdf"

    elif fn_lower.endswith(".doc"):
        import olefile
        import re
        try:
            ole_stream = io.BytesIO(content_bytes)
            ole = olefile.OleFileIO(ole_stream)
            if ole.exists("WordDocument"):
                stream = ole.openstream("WordDocument").read()
                decoded = stream.decode("utf-16-le", errors="ignore")
                paragraphs = [p.strip() for p in re.split(r'[\r\n\x00-\x1f\x7f-\x9f]+', decoded) if len(p.strip()) > 10]
                # Lọc bỏ các đoạn chứa ký tự rác nhị phân (binary junk)
                cleaned = []
                for p in paragraphs:
                    valid_chars = sum(1 for c in p if c.isalnum() or c.isspace() or c in ".,;:!?()-_'\"/\\@#%&*+=<>[]{}")
                    if len(p) > 0 and (valid_chars / len(p)) >= 0.70:
                        cleaned.append(p)
                if cleaned:
                    return "\n\n".join(cleaned).strip(), "doc"
        except Exception:
            pass

        # Fallback đọc text thô nếu olefile không trích xuất được
        import re
        text = content_bytes.decode("latin-1", errors="ignore")
        clean = re.findall(r'[\w\s\.,;:!?\(\)\-\'\"]{10,}', text)
        return "\n\n".join(clean).strip(), "doc"

    else:
        raise ValueError(f"Định dạng tệp '{filename}' không được hỗ trợ. Chỉ chấp nhận .txt, .docx, .doc, .pdf")
