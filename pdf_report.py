"""
Module Sinh Báo Cáo Thẩm Định Văn Bản PDF (AI Inspection Audit Report)
Đồ án Chuyên ngành Kỹ thuật Phần mềm: Hệ thống Phát hiện Văn bản do LLM sinh (Ensemble PLMs)
"""

import os
import io
from datetime import datetime
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Đăng ký phông chữ Arial hỗ trợ tiếng Việt trên Windows
FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"

try:
    if os.path.exists(r"C:\Windows\Fonts\arial.ttf"):
        pdfmetrics.registerFont(TTFont("ArialVN", r"C:\Windows\Fonts\arial.ttf"))
        FONT_REGULAR = "ArialVN"
    if os.path.exists(r"C:\Windows\Fonts\arialbd.ttf"):
        pdfmetrics.registerFont(TTFont("ArialVN-Bold", r"C:\Windows\Fonts\arialbd.ttf"))
        FONT_BOLD = "ArialVN-Bold"
except Exception as e:
    print(f"[!] Cảnh báo nạp font Arial: {e}, dùng font mặc định.")


class NumberedCanvas(canvas.Canvas):
    """Canvas đánh số trang tự động dạng Trang X / Y."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont(FONT_REGULAR, 8.5)
        self.setFillColor(colors.HexColor("#64748B"))
        # Header line
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(40, 800, 555, 800)
        self.drawString(40, 805, "AI Text Detector • Hệ thống Giám định Văn bản Ensemble PLMs")

        # Footer
        self.line(40, 45, 555, 45)
        page_text = f"Trang {self._pageNumber} / {page_count}"
        self.drawRightString(555, 32, page_text)
        self.drawString(40, 32, "Đồ án Chuyên ngành Kỹ thuật Phần mềm • Bản quyền thuộc Nhóm tác giả")
        self.restoreState()


def generate_pdf_report(scan_data: Dict[str, Any]) -> bytes:
    """
    Sinh tài liệu PDF Báo cáo Giám định Văn bản từ dữ liệu lượt quét.
    :param scan_data: Dictionary chứa thông tin quét, verdict, confidence, sentences, breakdown.
    :return: bytes của tệp PDF sẵn sàng gửi về client.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=50,
        bottomMargin=50,
    )

    styles = getSampleStyleSheet()

    # Định nghĩa các styles chuẩn thẩm mỹ
    title_style = ParagraphStyle(
        "DocTitle",
        fontName=FONT_BOLD,
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
        alignment=1,  # Center
    )

    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        fontName=FONT_REGULAR,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        alignment=1,
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        fontName=FONT_BOLD,
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=10,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "Body",
        fontName=FONT_REGULAR,
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )

    body_bold = ParagraphStyle(
        "BodyBold",
        fontName=FONT_BOLD,
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#0F172A"),
    )

    story = []

    # 1. Header & Title
    story.append(Spacer(1, 10))
    story.append(Paragraph("BÁO CÁO GIÁM ĐỊNH NGUỒN GỐC VĂN BẢN", title_style))
    story.append(Spacer(1, 3))
    story.append(Paragraph("AI-GENERATED TEXT INSPECTION & AUDIT CERTIFICATE", subtitle_style))
    story.append(Paragraph("Hệ thống kết hợp mô hình ngôn ngữ lớn: BERT, RoBERTa và Longformer", subtitle_style))
    story.append(Spacer(1, 12))

    # 2. Thông tin thẩm định (Metadata Table)
    scan_id = scan_data.get("id", "N/A")
    timestamp_str = scan_data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    doc_title = scan_data.get("title") or scan_data.get("filename") or "Văn bản phân tích trực tiếp"
    word_count = scan_data.get("word_count", 0)
    char_count = scan_data.get("char_count", 0)
    proc_time = scan_data.get("inference_time_ms", 0.0)

    meta_table_data = [
        [
            Paragraph("<b>Mã thẩm định:</b>", body_style),
            Paragraph(f"AUDIT-{scan_id:06d}" if isinstance(scan_id, int) else str(scan_id), body_bold),
            Paragraph("<b>Thời gian thực hiện:</b>", body_style),
            Paragraph(str(timestamp_str), body_style),
        ],
        [
            Paragraph("<b>Tên tài liệu:</b>", body_style),
            Paragraph(str(doc_title)[:45], body_bold),
            Paragraph("<b>Độ dài văn bản:</b>", body_style),
            Paragraph(f"{word_count:,} từ ({char_count:,} ký tự)", body_style),
        ],
        [
            Paragraph("<b>Phương pháp:</b>", body_style),
            Paragraph("Weighted Ensemble (3 PLMs)", body_style),
            Paragraph("<b>Thời gian phản hồi:</b>", body_style),
            Paragraph(f"{proc_time} ms", body_style),
        ],
    ]

    meta_table = Table(meta_table_data, colWidths=[100, 160, 110, 145])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # 3. Khung kết luận chính thức (Official Verdict Box)
    is_ai = scan_data.get("verdict") == "AI-generated" or scan_data.get("ai_probability", 0) >= 0.5
    ai_prob = float(scan_data.get("ai_probability", 0.0))
    human_prob = 1.0 - ai_prob
    confidence = scan_data.get("confidence", "Cao")

    box_bg = colors.HexColor("#FEF2F2") if is_ai else colors.HexColor("#F0FDF4")
    box_border = colors.HexColor("#EF4444") if is_ai else colors.HexColor("#10B981")
    verdict_text_color = colors.HexColor("#B91C1C") if is_ai else colors.HexColor("#047857")
    verdict_title = "KẾT LUẬN: VĂN BẢN DO TRÍ TUỆ NHÂN TẠO SINH (AI-GENERATED)" if is_ai else "KẾT LUẬN: VĂN BẢN DO CON NGƯỜI TỰ NHIÊN VIẾT (HUMAN-WRITTEN)"

    verdict_p_style = ParagraphStyle(
        "VerdictTitle",
        fontName=FONT_BOLD,
        fontSize=13,
        leading=16,
        textColor=verdict_text_color,
        alignment=1,
    )

    verdict_sub_style = ParagraphStyle(
        "VerdictSub",
        fontName=FONT_REGULAR,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1E293B"),
        alignment=1,
    )

    verdict_content = [
        [Paragraph(f"🛡️ <b>{verdict_title}</b>", verdict_p_style)],
        [Spacer(1, 4)],
        [Paragraph(
            f"Xác suất AI sinh: <b>{ai_prob*100:.2f}%</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Xác suất Người viết: <b>{human_prob*100:.2f}%</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Độ tin cậy: <b>{confidence}</b>",
            verdict_sub_style
        )],
    ]

    verdict_table = Table(verdict_content, colWidths=[515])
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), box_bg),
        ("BOX", (0, 0), (-1, -1), 1.5, box_border),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 14))

    # 4. Bảng phân rã điểm số các mô hình thành phần (Ensemble Decomposition)
    story.append(Paragraph("📊 <b>Đóng Góp Của Các Mô Hình Trong Kiến Trúc Học Kết Hợp</b>", section_heading))

    score_longformer = scan_data.get("score_longformer", 0.0)
    score_bert = scan_data.get("score_bert", 0.0)
    score_roberta = scan_data.get("score_roberta", 0.0)

    # Nếu có breakdown trong details
    details = scan_data.get("details", {})
    if "model_breakdown" in details:
        mb = details["model_breakdown"].get("individual_models", {})
        score_longformer = mb.get("longformer", {}).get("probability", score_longformer)
        score_bert = mb.get("bert", {}).get("probability", score_bert)
        score_roberta = mb.get("roberta", {}).get("probability", score_roberta)

    breakdown_data = [
        [
            Paragraph("<b>Mô hình nền tảng (PLM)</b>", body_bold),
            Paragraph("<b>Đặc tả kỹ thuật</b>", body_bold),
            Paragraph("<b>Trọng số đóng góp</b>", body_bold),
            Paragraph("<b>Xác suất dự đoán AI</b>", body_bold),
        ],
        [
            Paragraph("<b>Longformer</b> (longformer-base-4096)", body_style),
            Paragraph("Ngữ cảnh 1024 tokens, Sliding Window Attention", body_style),
            Paragraph("<b>48.65%</b> (Trụ cột)", body_style),
            Paragraph(f"<b>{score_longformer*100:.2f}%</b>", body_style),
        ],
        [
            Paragraph("<b>BERT</b> (bert-base-uncased)", body_style),
            Paragraph("Ngữ cảnh 512 tokens, Tối ưu câu đơn và tin ngắn", body_style),
            Paragraph("<b>33.96%</b>", body_style),
            Paragraph(f"<b>{score_bert*100:.2f}%</b>", body_style),
        ],
        [
            Paragraph("<b>RoBERTa</b> (roberta-base)", body_style),
            Paragraph("Ngữ cảnh 512 tokens, Nhạy bén biến thể AI", body_style),
            Paragraph("<b>17.38%</b>", body_style),
            Paragraph(f"<b>{score_roberta*100:.2f}%</b>", body_style),
        ],
        [
            Paragraph("<b>Ensemble (Weighted Soft Voting)</b>", body_bold),
            Paragraph("Dung hợp đa mô hình tối ưu theo hàm Log Loss", body_bold),
            Paragraph("<b>100.0%</b>", body_bold),
            Paragraph(f"<b>{ai_prob*100:.2f}% AI</b>", body_bold),
        ],
    ]

    breakdown_table = Table(breakdown_data, colWidths=[165, 180, 85, 85])
    breakdown_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#94A3B8")),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#F1F5F9")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(breakdown_table)
    story.append(Spacer(1, 14))

    # 5. Phân tích chi tiết cấp câu (Sentence-Level AI Hotspots)
    sentences = details.get("sentences", [])
    if sentences:
        story.append(Paragraph("📑 <b>Phân Tích Chi Tiết Cấp Câu (AI Hotspots)</b>", section_heading))
        story.append(Paragraph("Hệ thống bóc tách văn bản thành từng câu độc lập để xác định cụ thể các đoạn văn có sự can thiệp của AI:", body_style))
        story.append(Spacer(1, 6))

        sent_rows = [
            [
                Paragraph("<b>STT</b>", body_bold),
                Paragraph("<b>% AI</b>", body_bold),
                Paragraph("<b>Mức độ</b>", body_bold),
                Paragraph("<b>Nội dung câu phân tích</b>", body_bold),
            ]
        ]

        for s in sentences[:30]:  # Tối đa 30 câu đầu tiên trong báo cáo PDF để tránh tràn trang
            color_hex = "#EF4444" if s["highlight_color"] == "red" else "#EAB308" if s["highlight_color"] == "yellow" else "#10B981"
            level_text = "Chắc chắn AI" if s["highlight_color"] == "red" else "Nghi vấn" if s["highlight_color"] == "yellow" else "Tự nhiên"

            row = [
                Paragraph(str(s["sentence_index"]), body_style),
                Paragraph(f"<b>{s['ai_percentage']}</b>", body_style),
                Paragraph(f"<font color='{color_hex}'><b>{level_text}</b></font>", body_style),
                Paragraph(s["sentence"], body_style),
            ]
            sent_rows.append(row)

        sent_table = Table(sent_rows, colWidths=[30, 50, 75, 360])
        sent_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#94A3B8")),
            ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(sent_table)

    # 6. Đoạn văn bản mẫu thẩm định (Text Excerpt)
    full_text = scan_data.get("full_text", "")
    if full_text:
        story.append(Spacer(1, 10))
        story.append(Paragraph("📄 <b>Trích Đoạn Văn Bản Được Thẩm Định</b>", section_heading))
        excerpt = full_text[:600] + ("..." if len(full_text) > 600 else "")
        quote_p = Paragraph(f"<i>\"{excerpt}\"</i>", body_style)
        quote_table = Table([[quote_p]], colWidths=[515])
        quote_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(quote_table)

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()
