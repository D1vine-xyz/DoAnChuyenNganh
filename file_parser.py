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
        try:
            text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content_bytes.decode("utf-8-sig")
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

    else:
        raise ValueError(f"Định dạng tệp '{filename}' không được hỗ trợ. Chỉ chấp nhận .txt, .docx, .pdf")
