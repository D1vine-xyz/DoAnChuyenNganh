"""
Module Xử Lý Chấm Bài Hàng Loạt Lớp Học (.ZIP / Multi-File) & Xuất Bảng Điểm Excel/CSV
Đồ án Chuyên ngành Kỹ thuật Phần mềm (Chủ đề 6): VeriText AI

Tính năng:
1. Giải nén an toàn tệp .ZIP bài nộp từ LMS (Moodle / Google Classroom / Canvas):
   - Chống lỗi bảo mật Zip Slip (path traversal) & Zip Bomb (giới hạn dung lượng + số tệp).
   - Tự động bỏ qua tệp rác hệ thống (__MACOSX, .DS_Store, ~$ temp Office).
   - Hỗ trợ đồng thời kéo thả 1 file .ZIP hoặc chọn nhiều tệp (.docx, .doc, .pdf, .txt).
2. Trích xuất nhãn sinh viên/tệp sạch đẹp và phân tầng rủi ro (High / Moderate / Safe).
3. Xuất Bảng điểm kiểm định chuyên nghiệp dạng Excel (.xlsx) có tô màu Đỏ/Vàng/Xanh
   và dạng CSV chuẩn UTF-8 BOM (mở thẳng bằng Microsoft Excel không lỗi font tiếng Việt).
"""

import csv
import io
import os
import posixpath
import re
import zipfile
from datetime import datetime
from typing import Dict, Any, List, Tuple

from file_parser import extract_text_from_file

SUPPORTED_DOC_EXTENSIONS = (".txt", ".docx", ".doc", ".pdf")
MAX_BATCH_FILES = 100
MAX_SINGLE_FILE_BYTES = 15 * 1024 * 1024     # 15 MB / file
MAX_TOTAL_UNCOMPRESSED_BYTES = 100 * 1024 * 1024  # 100 MB / đợt chấm

TIER_LABELS_VI = {
    "AI_PURE": "AI sinh toàn phần",
    "HYBRID_MIXED": "Văn bản lai (Người + AI)",
    "AI_ASSISTED": "Nghi vấn AI biên tập",
    "HUMAN_PURE": "Tự viết hoàn toàn",
}

RISK_LABELS_VI = {
    "high": "Nguy cơ cao",
    "moderate": "Cần xem xét",
    "safe": "An toàn",
}


def classify_risk_level(ai_prob: float, has_bypass_warning: bool = False) -> str:
    """Phân loại mức độ rủi ro cho bảng xếp hạng giảng viên."""
    if has_bypass_warning or ai_prob >= 0.75:
        return "high"
    if ai_prob >= 0.45:
        return "moderate"
    return "safe"


def parse_student_label(filename: str) -> str:
    """Chuẩn hóa tên file thành tên hiển thị Sinh viên / Tài liệu gọn gàng."""
    base = os.path.basename(filename or "Khong_ten")
    stem, _ = os.path.splitext(base)
    clean = re.sub(r"[_\-]+", " ", stem).strip()
    clean = re.sub(r"\s{2,}", " ", clean)
    return clean[:120] if clean else base


def _decode_zip_member_name(info: zipfile.ZipInfo) -> str:
    """Xử lý tên file tiếng Việt trong ZIP chuẩn UTF-8 hoặc CP437 fallback."""
    name = info.filename
    if info.flag_bits & 0x800:
        return name
    try:
        return name.encode("cp437").decode("utf-8")
    except Exception:
        return name


def _is_safe_zip_member(member_name: str) -> bool:
    """Chặn Zip Slip path traversal và lọc rác hệ điều hành/Office."""
    norm = posixpath.normpath(member_name.replace("\\", "/"))
    if norm.startswith("/") or norm.startswith("..") or "/../" in f"/{norm}/":
        return False
    parts = [p for p in norm.split("/") if p]
    if not parts:
        return False
    base = parts[-1]
    if any(p == "__MACOSX" or p.startswith(".") for p in parts):
        return False
    if base.startswith("~$"):
        return False
    return base.lower().endswith(SUPPORTED_DOC_EXTENSIONS)


def extract_files_from_uploads(
    uploaded_items: List[Tuple[str, bytes]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Nhận danh sách (filename, bytes) gồm tệp tài liệu lẻ hoặc file .ZIP,
    trả về (valid_documents, skipped_files).
    """
    valid_docs: List[Dict[str, Any]] = []
    skipped_files: List[Dict[str, Any]] = []
    total_bytes = 0

    def _process_single_doc(doc_name: str, raw_bytes: bytes):
        nonlocal total_bytes
        if len(valid_docs) >= MAX_BATCH_FILES:
            skipped_files.append({"filename": doc_name, "reason": f"Vượt giới hạn tối đa {MAX_BATCH_FILES} tệp/đợt."})
            return
        if not raw_bytes:
            skipped_files.append({"filename": doc_name, "reason": "Tệp rỗng (0 bytes)."})
            return
        if len(raw_bytes) > MAX_SINGLE_FILE_BYTES:
            skipped_files.append({"filename": doc_name, "reason": "Vượt quá 15 MB."})
            return
        total_bytes += len(raw_bytes)
        if total_bytes > MAX_TOTAL_UNCOMPRESSED_BYTES:
            skipped_files.append({"filename": doc_name, "reason": "Vượt tổng dung lượng giải nén 100 MB."})
            return

        try:
            text, ext = extract_text_from_file(doc_name, raw_bytes)
        except Exception as e:
            skipped_files.append({"filename": doc_name, "reason": f"Lỗi đọc tệp: {str(e)}"})
            return

        words = text.split()
        if len(words) < 5:
            skipped_files.append({"filename": doc_name, "reason": "Văn bản quá ngắn (< 5 từ)."})
            return

        valid_docs.append({
            "filename": os.path.basename(doc_name),
            "student_label": parse_student_label(doc_name),
            "extension": ext,
            "text": text,
        })

    for fname, content in uploaded_items:
        lower_name = (fname or "").lower()
        if lower_name.endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        member_name = _decode_zip_member_name(info)
                        if not _is_safe_zip_member(member_name):
                            continue
                        if info.file_size > MAX_SINGLE_FILE_BYTES:
                            skipped_files.append({"filename": os.path.basename(member_name), "reason": "Tệp trong ZIP > 15 MB."})
                            continue
                        raw = zf.read(info)
                        _process_single_doc(os.path.basename(member_name), raw)
            except zipfile.BadZipFile:
                skipped_files.append({"filename": fname, "reason": "Tệp ZIP không hợp lệ hoặc bị hỏng."})
            except Exception as e:
                skipped_files.append({"filename": fname, "reason": f"Lỗi giải nén ZIP: {str(e)}"})
        elif lower_name.endswith(SUPPORTED_DOC_EXTENSIONS):
            _process_single_doc(fname, content)
        else:
            skipped_files.append({
                "filename": fname,
                "reason": "Định dạng không hỗ trợ (chỉ chấp nhận .zip, .docx, .doc, .pdf, .txt).",
            })

    return valid_docs, skipped_files


def generate_batch_csv_bytes(batch_detail: Dict[str, Any]) -> bytes:
    """Sinh file bảng điểm CSV chuẩn UTF-8 BOM (mở bằng Excel không lỗi tiếng Việt)."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "STT",
        "Sinh viên / Tên tệp",
        "Ngôn ngữ",
        "Số từ",
        "Xác suất AI (%)",
        "Mức độ Rủi ro",
        "Phân loại 4 mức",
        "Cảnh báo Bypass",
        "Burstiness (0-100)",
        "Perplexity Proxy (0-100)",
        "Kết luận",
        "Thời gian xử lý (ms)",
        "Mã kiểm định",
    ])

    for idx, s in enumerate(batch_detail.get("submissions", []), start=1):
        lang_txt = "Tiếng Việt" if s.get("language") == "vi" else "Tiếng Anh"
        tier_txt = TIER_LABELS_VI.get(s.get("authorship_tier", ""), s.get("verdict", ""))
        risk_txt = RISK_LABELS_VI.get(s.get("risk_level", "safe"), "An toàn")
        bypass_txt = "CÓ - Phát hiện lách luật" if s.get("has_bypass_warning") else "Không"
        verdict_txt = "AI-generated" if s.get("verdict") == "AI-generated" else "Human-written"

        writer.writerow([
            idx,
            s.get("student_label") or s.get("title") or "",
            lang_txt,
            s.get("word_count", 0),
            f"{float(s.get('ai_probability', 0.0)) * 100:.2f}%",
            risk_txt,
            tier_txt,
            bypass_txt,
            s.get("burstiness_score") if s.get("burstiness_score") is not None else "",
            s.get("perplexity_score") if s.get("perplexity_score") is not None else "",
            verdict_txt,
            s.get("inference_time_ms", 0.0),
            f"AUDIT-{int(s.get('scan_id', 0)):06d}",
        ])

    return ("\ufeff" + output.getvalue()).encode("utf-8")


def generate_batch_excel_bytes(batch_detail: Dict[str, Any]) -> bytes:
    """
    Sinh bảng điểm kiểm định Excel (.xlsx) chuyên nghiệp bằng openpyxl:
    - Sheet 1: Bảng điểm từng sinh viên có tô màu Đỏ / Vàng / Xanh theo mức rủi ro.
    - Sheet 2: Tổng quan thống kê lớp học dành cho báo cáo giảng viên.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "Bang_Diem_Kiem_Dinh"

    # Styles
    title_font = Font(name="Arial", size=14, bold=True, color="0F172A")
    meta_font = Font(name="Arial", size=10, italic=True, color="475569")
    header_font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
    body_font = Font(name="Arial", size=10, color="1E293B")
    bold_font = Font(name="Arial", size=10, bold=True, color="0F172A")

    header_fill = PatternFill(start_color="0F172A", end_color="0F172A", fill_type="solid")
    fill_high = PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid")
    fill_mod = PatternFill(start_color="FEF3C7", end_color="FEF3C7", fill_type="solid")
    fill_safe = PatternFill(start_color="DCFCE7", end_color="DCFCE7", fill_type="solid")
    fill_bypass = PatternFill(start_color="FFEDD5", end_color="FFEDD5", fill_type="solid")

    thin_side = Side(style="thin", color="CBD5E1")
    border_all = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)

    # Header Metadata
    ws.merge_cells("A1:M1")
    ws["A1"] = f"BẢNG ĐIỂM KIỂM ĐỊNH LIÊM CHÍNH HỌC THUẬT — {batch_detail.get('batch_name', 'Lớp học').upper()}"
    ws["A1"].font = title_font
    ws["A1"].alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 28

    ws.merge_cells("A2:M2")
    ws["A2"] = (
        f"Giảng viên phụ trách: {batch_detail.get('teacher_name', 'Giảng viên')}   |   "
        f"Thời gian chấm: {batch_detail.get('created_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}   |   "
        f"Tổng số bài: {batch_detail.get('total_files', 0)}   |   "
        f"Nguy cơ cao: {batch_detail.get('high_risk_count', 0)}   |   "
        f"Nghi vấn: {batch_detail.get('moderate_risk_count', 0)}   |   "
        f"An toàn: {batch_detail.get('safe_count', 0)}   |   "
        f"Cảnh báo Bypass: {batch_detail.get('bypass_count', 0)}"
    )
    ws["A2"].font = meta_font
    ws.row_dimensions[2].height = 20

    headers = [
        "STT",
        "Sinh viên / Tên tệp",
        "Ngôn ngữ",
        "Số từ",
        "Xác suất AI (%)",
        "Mức độ Rủi ro",
        "Phân loại 4 mức",
        "Cảnh báo Bypass",
        "Burstiness",
        "Perplexity Proxy",
        "Kết luận",
        "Thời gian (ms)",
        "Mã kiểm định",
    ]
    header_row_idx = 4
    ws.row_dimensions[header_row_idx].height = 24
    for col_idx, text in enumerate(headers, start=1):
        cell = ws.cell(row=header_row_idx, column=col_idx, value=text)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = border_all

    # Data rows
    for idx, s in enumerate(batch_detail.get("submissions", []), start=1):
        row_idx = header_row_idx + idx
        risk = s.get("risk_level", "safe")
        row_fill = fill_high if risk == "high" else fill_mod if risk == "moderate" else fill_safe

        lang_txt = "Tiếng Việt (VI)" if s.get("language") == "vi" else "Tiếng Anh (EN)"
        tier_txt = TIER_LABELS_VI.get(s.get("authorship_tier", ""), s.get("verdict", ""))
        risk_txt = RISK_LABELS_VI.get(risk, "An toàn")
        bypass_txt = "CÓ (Lách luật)" if s.get("has_bypass_warning") else "Không"

        row_values = [
            idx,
            s.get("student_label") or s.get("title") or "",
            lang_txt,
            int(s.get("word_count", 0)),
            round(float(s.get("ai_probability", 0.0)) * 100, 2),
            risk_txt,
            tier_txt,
            bypass_txt,
            s.get("burstiness_score") if s.get("burstiness_score") is not None else "-",
            s.get("perplexity_score") if s.get("perplexity_score") is not None else "-",
            s.get("verdict", ""),
            round(float(s.get("inference_time_ms", 0.0)), 1),
            f"AUDIT-{int(s.get('scan_id', 0)):06d}",
        ]

        ws.row_dimensions[row_idx].height = 20
        for col_idx, val in enumerate(row_values, start=1):
            cell = ws.cell(row=row_idx, column=col_idx, value=val)
            cell.font = bold_font if col_idx in (5, 6) else body_font
            cell.fill = fill_bypass if (col_idx == 8 and s.get("has_bypass_warning")) else row_fill
            cell.border = border_all
            if col_idx in (1, 3, 4, 5, 6, 8, 9, 10, 12, 13):
                cell.alignment = Alignment(horizontal="center", vertical="center")
            else:
                cell.alignment = Alignment(horizontal="left", vertical="center")
            if col_idx == 5:
                cell.number_format = '0.00"%"'

    # Auto-fit column widths
    col_widths = [6, 34, 16, 10, 16, 15, 25, 16, 12, 16, 16, 14, 15]
    for col_idx, width in enumerate(col_widths, start=1):
        ws.column_dimensions[get_column_letter(col_idx)].width = width

    ws.freeze_panes = "A5"

    # Sheet 2: Tổng quan lớp học
    ws2 = wb.create_sheet(title="Tong_Quan_Lop_Hoc")
    summary_rows = [
        ("THỐNG KÊ TỔNG HỢP ĐỢT CHẤM LỚP HỌC", ""),
        ("Tên đợt chấm / Môn học", batch_detail.get("batch_name", "")),
        ("Giảng viên phụ trách", batch_detail.get("teacher_name", "")),
        ("Thời gian thực hiện", batch_detail.get("created_at", "")),
        ("Tổng số bài đã chấm", batch_detail.get("total_files", 0)),
        ("Số bài Nguy cơ cao (Đỏ - >=75% AI hoặc Bypass)", batch_detail.get("high_risk_count", 0)),
        ("Số bài Nghi vấn / Cần xem xét (Vàng - 45-75% AI)", batch_detail.get("moderate_risk_count", 0)),
        ("Số bài An toàn (Xanh - <45% AI)", batch_detail.get("safe_count", 0)),
        ("Số bài phát hiện tiểu xảo Anti-Bypass", batch_detail.get("bypass_count", 0)),
        ("Xác suất AI trung bình toàn lớp (%)", round(float(batch_detail.get("avg_ai_probability", 0.0)) * 100, 2)),
    ]
    for r_idx, (k, v) in enumerate(summary_rows, start=1):
        c1 = ws2.cell(row=r_idx, column=1, value=k)
        c2 = ws2.cell(row=r_idx, column=2, value=v)
        if r_idx == 1:
            ws2.merge_cells("A1:B1")
            c1.font = title_font
        else:
            c1.font = bold_font
            c2.font = body_font
            c1.border = border_all
            c2.border = border_all
    ws2.column_dimensions["A"].width = 44
    ws2.column_dimensions["B"].width = 32

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()
