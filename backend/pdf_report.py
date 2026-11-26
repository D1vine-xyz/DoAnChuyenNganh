"""
Module Sinh Báo Cáo Thẩm Định Văn Bản PDF (AI Inspection Audit Report)
Đồ án Chuyên ngành Kỹ thuật Phần mềm: Hệ thống Phát hiện Văn bản do LLM sinh (Ensemble PLMs)
"""

import os
import io
from datetime import datetime
from typing import Dict, Any, Optional
from xml.sax.saxutils import escape as xml_escape

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
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.barcode import qr
import auth

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
        self.drawString(40, 805, "VeriText AI • Hệ thống Giám định Văn bản Ensemble PLMs")

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
    story.append(Paragraph("AI-GENERATED TEXT INSPECTION &amp; AUDIT CERTIFICATE", subtitle_style))
    story.append(Paragraph("Hệ thống Song ngữ: PhoBERT base v2 (Tiếng Việt) &amp; Ensemble PLMs (Tiếng Anh)", subtitle_style))
    story.append(Spacer(1, 12))

    # 2. Thông tin thẩm định & Mã QR Chống Giả Mạo (Metadata + Cryptographic QR Verification)
    scan_id = scan_data.get("id", "N/A")
    timestamp_str = scan_data.get("timestamp") or scan_data.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    doc_title = scan_data.get("title") or scan_data.get("filename") or "Văn bản phân tích trực tiếp"
    author_name = scan_data.get("author_name") or "Khách vãng lai"
    lang_code = scan_data.get("language", "en")
    is_vi = lang_code == "vi"
    word_count = int(scan_data.get("word_count", 0))
    char_count = int(scan_data.get("char_count", 0))
    proc_time = scan_data.get("inference_time_ms", 0.0)
    ai_prob_meta = float(scan_data.get("ai_probability", 0.0))
    method_label = "PhoBERT v2 Sliding-Window (VI)" if is_vi else "Weighted Ensemble 3 PLMs (EN)"

    audit_code = scan_data.get("audit_code")
    if not audit_code and isinstance(scan_id, int):
        audit_code = auth.compute_audit_code(scan_id=scan_id, ai_probability=ai_prob_meta, word_count=word_count)
    elif not audit_code:
        audit_code = f"AUDIT-{scan_id}"

    base_verify_url = scan_data.get("verify_base_url") or "http://localhost:8000"
    verify_url = f"{base_verify_url.rstrip('/')}/?verify={audit_code}"

    # Tạo mã QR vector sắc nét nhúng trực tiếp vào biên bản PDF
    qr_widget = qr.QrCodeWidget(verify_url)
    qr_bounds = qr_widget.getBounds()
    qr_w = max(qr_bounds[2] - qr_bounds[0], 1.0)
    qr_h = max(qr_bounds[3] - qr_bounds[1], 1.0)
    qr_size = 64.0
    qr_drawing = Drawing(qr_size, qr_size, transform=[qr_size / qr_w, 0, 0, qr_size / qr_h, 0, 0])
    qr_drawing.add(qr_widget)

    meta_inner_data = [
        [
            Paragraph("<b>Mã xác thực:</b>", body_style),
            Paragraph(xml_escape(str(audit_code)), body_bold),
            Paragraph("<b>Thời gian:</b>", body_style),
            Paragraph(xml_escape(str(timestamp_str)), body_style),
        ],
        [
            Paragraph("<b>Tên tài liệu:</b>", body_style),
            Paragraph(xml_escape(str(doc_title)[:40]), body_bold),
            Paragraph("<b>Độ dài:</b>", body_style),
            Paragraph(f"{word_count:,} từ ({char_count:,} ký tự)", body_style),
        ],
        [
            Paragraph("<b>Phương pháp:</b>", body_style),
            Paragraph(method_label, body_style),
            Paragraph("<b>Người quét:</b>", body_style),
            Paragraph(f"{xml_escape(str(author_name)[:26])} ({proc_time} ms)", body_style),
        ],
    ]
    meta_inner_table = Table(meta_inner_data, colWidths=[78, 145, 72, 136])
    meta_inner_table.setStyle(TableStyle([
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))

    qr_caption = ParagraphStyle(
        "QrCaption",
        fontName=FONT_REGULAR,
        fontSize=6.8,
        leading=8.5,
        textColor=colors.HexColor("#475569"),
        alignment=1,
    )
    qr_cell_table = Table(
        [[qr_drawing], [Paragraph("Quét QR xác thực<br/>chống giả mạo PDF", qr_caption)]],
        colWidths=[80],
    )
    qr_cell_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
    ]))

    meta_table = Table([[meta_inner_table, qr_cell_table]], colWidths=[431, 84])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#CBD5E1")),
        ("LINEBEFORE", (1, 0), (1, 0), 0.5, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # 3. Khung kết luận chính thức (Official Verdict Box)
    raw_verdict = scan_data.get("verdict") or scan_data.get("label") or ""
    ai_prob = float(scan_data.get("ai_probability", 0.0))
    is_ai = raw_verdict == "AI-generated" or (not raw_verdict and ai_prob >= 0.5)
    human_prob = 1.0 - ai_prob
    confidence = xml_escape(str(scan_data.get("confidence", "Cao")))

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

    tier_map_pdf = {
        "AI_PURE": "AI sinh toàn phần (Pure AI)",
        "HYBRID_MIXED": "Văn bản lai: Trộn Người + AI (Hybrid)",
        "AI_ASSISTED": "Nghi vấn AI biên tập / Paraphrase",
        "HUMAN_PURE": "Tự viết hoàn toàn (Human-Written)",
    }
    raw_tier = scan_data.get("authorship_tier") or scan_data.get("details", {}).get("authorship_tier")
    tier_code = raw_tier.get("code") if isinstance(raw_tier, dict) else str(raw_tier or "")
    tier_display = tier_map_pdf.get(tier_code, "")

    verdict_content = [
        [Paragraph(f"<b>{verdict_title}</b>", verdict_p_style)],
        [Spacer(1, 4)],
        [Paragraph(
            f"Xác suất AI sinh: <b>{ai_prob*100:.2f}%</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Xác suất Người viết: <b>{human_prob*100:.2f}%</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Độ tin cậy: <b>{confidence}</b>",
            verdict_sub_style
        )],
    ]
    if tier_display:
        verdict_content.append([Paragraph(f"Phân loại mức độ can thiệp: <b>{xml_escape(tier_display)}</b>", verdict_sub_style)])

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
    story.append(Spacer(1, 10))

    # 3b. Khung Cảnh báo Gian lận Kỹ thuật (Anti-Bypass Alert) nếu phát hiện
    has_bypass = bool(scan_data.get("has_bypass_warning"))
    bypass_report = scan_data.get("bypass_report") or scan_data.get("details", {}).get("bypass_report") or {}
    if has_bypass or bypass_report.get("has_bypass_warning"):
        bypass_summary = xml_escape(str(bypass_report.get("summary") or "Phát hiện dấu hiệu chèn ký tự ẩn hoặc ký tự đồng hình nhằm qua mặt bộ lọc AI."))
        flagged = bypass_report.get("flagged_samples") or []
        flagged_str = f"<br/><b>Từ nghi vấn bị can thiệp:</b> {xml_escape(', '.join(flagged[:6]))}" if flagged else ""
        bypass_alert_table = Table(
            [[Paragraph(f"<b>[!] CẢNH BÁO TIỂU XẢO QUA MẶT AI (ANTI-BYPASS ALERT)</b><br/>{bypass_summary}{flagged_str}", body_style)]],
            colWidths=[515],
        )
        bypass_alert_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF7ED")),
            ("BOX", (0, 0), (-1, -1), 1.2, colors.HexColor("#EA580C")),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(bypass_alert_table)
        story.append(Spacer(1, 10))

    # 3c. Bảng Bằng Chứng Giải Thích Ngôn Ngữ Học (Explainable AI - XAI)
    xai = scan_data.get("xai_metrics") or scan_data.get("details", {}).get("xai_metrics")
    if isinstance(xai, dict) and xai.get("burstiness_score") is not None:
        story.append(Paragraph("<b>1. Bằng Chứng Giải Thích Ngôn Ngữ Học (Explainable AI - XAI)</b>", section_heading))
        b_score = xai.get("burstiness_score", 0.0)
        p_score = xai.get("perplexity_score", 0.0)
        lex_score = xai.get("lexical_diversity", 0.0)
        cliches = xai.get("ai_cliches") or []
        cliche_text = ", ".join(f'"{c["phrase"]}" (x{c["count"]})' for c in cliches[:6]) if cliches else "Không phát hiện cụm từ văn mẫu AI"

        xai_rows = [
            [
                Paragraph("<b>Chỉ số ngôn ngữ học</b>", body_bold),
                Paragraph("<b>Điểm số</b>", body_bold),
                Paragraph("<b>Diễn giải chuyên môn (Đối chứng)</b>", body_bold),
            ],
            [
                Paragraph("<b>Burstiness</b> (Độ biến thiên nhịp câu)", body_style),
                Paragraph(f"<b>{b_score}/100</b>", body_style),
                Paragraph(xml_escape(str(xai.get("burstiness_label", ""))), body_style),
            ],
            [
                Paragraph("<b>Perplexity Proxy</b> (Độ bất ngờ)", body_style),
                Paragraph(f"<b>{p_score}/100</b>", body_style),
                Paragraph(xml_escape(str(xai.get("perplexity_label", ""))), body_style),
            ],
            [
                Paragraph("<b>MATTR</b> (Độ đa dạng từ vựng)", body_style),
                Paragraph(f"<b>{lex_score}%</b>", body_style),
                Paragraph(f"TB {xai.get('sentence_stats', {}).get('avg_length', 0)} từ/câu (Độ lệch chuẩn ±{xai.get('sentence_stats', {}).get('std_length', 0)})", body_style),
            ],
            [
                Paragraph("<b>Cụm từ mẫu AI</b> (LLM Clichés)", body_style),
                Paragraph(f"<b>{xai.get('ai_cliche_count', 0)} lượt</b>", body_style),
                Paragraph(xml_escape(cliche_text), body_style),
            ],
        ]
        xai_table = Table(xai_rows, colWidths=[175, 70, 270])
        xai_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#94A3B8")),
            ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(xai_table)
        story.append(Spacer(1, 12))
    else:
        story.append(Spacer(1, 4))

    # 4. Bảng phân rã điểm số các mô hình thành phần (Ensemble Decomposition)
    story.append(Paragraph("<b>2. Đóng Góp Của Mô Hình Đánh Giá (Model Breakdown)</b>", section_heading))

    details = scan_data.get("details", {})
    mb = details.get("model_breakdown", {}).get("individual_models", {})

    if is_vi:
        p_phobert = mb.get("phobert", {}).get("probability", ai_prob)
        breakdown_data = [
            [
                Paragraph("<b>Mô hình nền tảng (PLM)</b>", body_bold),
                Paragraph("<b>Đặc tả kỹ thuật</b>", body_bold),
                Paragraph("<b>Trọng số đóng góp</b>", body_bold),
                Paragraph("<b>Xác suất dự đoán AI</b>", body_bold),
            ],
            [
                Paragraph("<b>PhoBERT v2</b> (vinai/phobert-base-v2)", body_style),
                Paragraph("Ngữ cảnh 256 tokens, Huấn luyện 8.000 mẫu Pre-2022 12 thể loại (Acc 99.08%)", body_style),
                Paragraph("<b>100.0%</b> (Chuyên biệt VN)", body_style),
                Paragraph(f"<b>{p_phobert*100:.2f}%</b>", body_style),
            ],
            [
                Paragraph("<b>PyVi Word Segmenter</b>", body_style),
                Paragraph("Tách từ ghép tiếng Việt chuẩn hóa BPE 64.000 tokens", body_style),
                Paragraph("Tiền xử lý", body_style),
                Paragraph("Hoàn tất", body_style),
            ],
        ]
    else:
        score_longformer = mb.get("longformer", {}).get("probability", scan_data.get("score_longformer", ai_prob))
        score_bert = mb.get("bert", {}).get("probability", scan_data.get("score_bert", ai_prob))
        score_roberta = mb.get("roberta", {}).get("probability", scan_data.get("score_roberta", ai_prob))
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
        story.append(Paragraph("<b>3. Phân Tích Chi Tiết Cấp Câu (AI Hotspots)</b>", section_heading))
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
                Paragraph(f"<b>{xml_escape(str(s['ai_percentage']))}</b>", body_style),
                Paragraph(f"<font color='{color_hex}'><b>{level_text}</b></font>", body_style),
                Paragraph(xml_escape(str(s["sentence"])), body_style),
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
    full_text = scan_data.get("full_text") or scan_data.get("text_snippet") or ""
    if full_text:
        story.append(Spacer(1, 10))
        story.append(Paragraph("<b>4. Trích Đoạn Văn Bản Được Thẩm Định</b>", section_heading))
        excerpt = xml_escape(full_text[:700] + ("..." if len(full_text) > 700 else ""))
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
