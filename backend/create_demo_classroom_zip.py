"""
Tạo sẵn gói bài nộp lớp học mẫu đa định dạng (Lop_K16_TieuLuan_Demo.zip) phục vụ Demo Vấn Đáp Hội Đồng.
Gồm 7 bài nộp thực tế (.DOCX có bảng biểu, .PDF văn bản số, .TXT) mô phỏng đầy đủ các tình huống:
1. 22110001_Nguyen_Van_An_TuViet.docx        (Tiếng Việt .DOCX - Người tự viết + Tài liệu tham khảo -> An toàn)
2. 22110002_Tran_Thi_Binh_ChatGPT.docx       (Tiếng Việt .DOCX - AI sinh thuần túy + Tài liệu tham khảo -> Nguy cơ cao)
3. 22110003_Le_Hoang_Cuong_LachLuat.docx     (Tiếng Việt .DOCX - Cài ký tự ẩn Zero-Width + Cyrillic -> Nguy cơ cao + Bypass)
4. 22110004_Pham_Minh_Duc_BaiLai.docx        (Tiếng Việt .DOCX - Văn bản lai trộn Người + AI -> Nguy cơ cao / Hybrid)
5. 22110005_Vo_Quoc_Huy_Essay_Human.pdf      (Tiếng Anh .PDF   - Người tự viết tự nhiên + References -> An toàn)
6. 22110006_Dang_Khanh_Linh_Essay_AI.pdf     (Tiếng Anh .PDF   - LLM sinh chuẩn khuôn mẫu + References -> Nguy cơ cao)
7. 22110007_Hoang_Minh_Tam_BienTapAI.txt     (Tiếng Anh .TXT   - Tự viết có AI biên tập/chèn ý -> Cần xem xét)
"""

import io
import os
import zipfile

import docx
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


def _make_docx(student_id: str, student_name: str, paragraphs: list[str]) -> bytes:
    doc = docx.Document()
    tbl = doc.add_table(rows=1, cols=2)
    tbl.cell(0, 0).text = f"MSSV: {student_id}"
    tbl.cell(0, 1).text = f"Sinh viên: {student_name}"
    for para in paragraphs:
        doc.add_paragraph(para)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _make_pdf_en(title: str, paragraphs: list[str]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=50, rightMargin=50, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"<b>{title}</b>", styles["Heading3"]), Spacer(1, 10)]
    for para in paragraphs:
        story.append(Paragraph(para, styles["BodyText"]))
        story.append(Spacer(1, 8))
    doc.build(story)
    return buf.getvalue()


def build_demo_zip() -> str:
    files_payload: dict[str, bytes] = {}

    # 1. Tiếng Việt - Người tự viết (.DOCX + Tài liệu tham khảo) -> An toàn
    files_payload["22110001_Nguyen_Van_An_TuViet.docx"] = _make_docx(
        "22110001",
        "Nguyễn Văn An",
        [
            "Tôi cũng nhiều lúc rơi vào tình trạng cuồng chân. "
            "Đi du lịch không được, về quê không xong, gặp gỡ, thăm nom ai đó cũng phải hỏi kỹ càng bởi chỗ nào cũng bị cảnh báo. "
            "Bố mẹ, anh chị, hàng xóm của tôi đều là F2, F3, con phố tôi ở cũng có người F0 rồi. "
            "Đành bó chân ở nhà viết bài, đọc sách, nghiên cứu tài liệu, soạn bài giảng, dạy con học và tập thể dục qua ngày.",
            "Tài liệu tham khảo:",
            "[1] Nhật ký ghi chép thực tế năm 2021.",
        ],
    )

    # 2. Tiếng Việt - ChatGPT thuần túy (.DOCX + Tài liệu tham khảo) -> Nguy cơ cao / AI toàn phần
    files_payload["22110002_Tran_Thi_Binh_ChatGPT.docx"] = _make_docx(
        "22110002",
        "Trần Thị Bình",
        [
            "Trí tuệ nhân tạo ngày nay đang đóng vai trò then chốt trong sự phát triển của nền giáo dục đại học toàn cầu. "
            "Không thể phủ nhận rằng các mô hình ngôn ngữ lớn mang lại nhiều lợi ích vượt trội trong việc hỗ trợ sinh viên tra cứu tài liệu và tổng hợp kiến thức. "
            "Bên cạnh đó, sự phổ biến nhanh chóng của các công cụ tạo sinh văn bản cũng đặt ra nhiều thách thức nghiêm trọng đối với tính liêm chính học thuật. "
            "Tóm lại, việc xây dựng hệ thống kiểm định văn bản minh bạch và đáng tin cậy là một hướng đi vô cùng cấp thiết trong bối cảnh hiện nay.",
            "Tài liệu tham khảo:",
            "[1] Vaswani et al. (2017). Attention Is All You Need. NeurIPS.",
            "[2] Nguyen & Nguyen (2020). PhoBERT: Pre-trained language models for Vietnamese.",
        ],
    )

    # 3. Tiếng Việt - Cài ký tự ẩn Zero-Width + Cyrillic Homoglyph (.DOCX) -> Nguy cơ cao + Anti-Bypass
    files_payload["22110003_Le_Hoang_Cuong_LachLuat.docx"] = _make_docx(
        "22110003",
        "Lê Hoàng Cường",
        [
            "Trí\u200b tuệ\u200b nhân\u200b t\u0430o đóng v\u0430i trò then chốt tr\u043eng bối c\u0430nh chuyển đổi số hiện n\u0430y. "
            "Không thể phủ nhận rằng công\u200b nghệ\u200b học sâu m\u0430ng lại nhiều lợi ích vượt trội ch\u043e ngành kỹ thuật phần mềm. "
            "Mặt khác, việc lạm dụng c\u0430c công cụ tự động sinh mã nguồn và tiểu luận đòi hỏi những giải ph\u0430p giám định chặt chẽ. "
            "Nhìn chung, sự kết hợp giữa mô hình học máy và quy trình đánh giá của giảng viên là giải pháp tối ưu.",
        ],
    )

    # 4. Tiếng Việt - Văn bản lai trộn Người + AI (.DOCX) -> Nguy cơ cao / HYBRID_MIXED
    files_payload["22110004_Pham_Minh_Duc_BaiLai.docx"] = _make_docx(
        "22110004",
        "Phạm Minh Đức",
        [
            "Công tác tiết kiệm, tiết giảm chi phí, hạ giá thành sản phẩm còn được đơn vị quan tâm đặc biệt và tổ chức phân giao cụ thể kế hoạch đến từng phòng ban. "
            "Trong năm, công ty đã tiết kiệm trên 260 tỷ đồng, đồng thời công tác nghiên cứu khoa học, phát huy sáng kiến, sáng chế cũng không ngừng được đẩy mạnh.",
            "Tuy nhiên, không thể phủ nhận rằng các mô hình ngôn ngữ lớn ngày nay sở hữu khả năng tổng hợp thông tin vượt trội và tạo ra các đoạn văn mạch lạc một cách ấn tượng. "
            "Hơn nữa, việc triển khai các khung đánh giá kết hợp đa mô hình tiền huấn luyện đóng vai trò then chốt trong việc bảo đảm tính xác thực và liêm chính học thuật.",
        ],
    )

    # 5. Tiếng Anh - Người tự viết (.PDF chuẩn chữ số + References) -> An toàn
    files_payload["22110005_Vo_Quoc_Huy_Essay_Human.pdf"] = _make_pdf_en(
        "22110005 - Vo Quoc Huy | Knowledge Representation Notes",
        [
            "Knowledge representation and reasoning is the field of artificial intelligence dedicated to representing information about the world in a form that a computer system can use to solve complex tasks such as diagnosing a medical condition or having a dialog in a natural language.",
            "Knowledge representation incorporates findings from psychology about how humans solve problems and represent knowledge in order to design formalisms that will make complex systems easier to design and build.",
            "Examples of knowledge representation formalisms include semantic nets, systems architecture, frames, rules, and ontologies. Examples of automated reasoning engines include inference engines, theorem provers, and classifiers.",
            "References",
            "[1] Brachman, R. & Levesque, H. (2004). Knowledge Representation and Reasoning. Morgan Kaufmann.",
        ],
    )

    # 6. Tiếng Anh - LLM sinh chuẩn khuôn mẫu (.PDF + References) -> Nguy cơ cao
    files_payload["22110006_Dang_Khanh_Linh_Essay_AI.pdf"] = _make_pdf_en(
        "22110006 - Dang Khanh Linh | AI in Software Engineering",
        [
            "Furthermore, it is crucial to note that artificial intelligence plays a pivotal role in reshaping modern software engineering practices. These multifaceted neural architectures leverage deep contextual representations to generate coherent and syntactically refined discourse at scale.",
            "Consequently, safeguarding academic integrity in today's digital age requires resilient multi-model verification frameworks. In conclusion, fostering transparent explainable detection tools is of paramount importance for higher education institutions.",
            "References",
            "[1] Brown, T. et al. (2020). Language Models are Few-Shot Learners. NeurIPS.",
        ],
    )

    # 7. Tiếng Anh - Người viết + AI biên tập nhẹ đoạn cuối (.TXT) -> Vùng Cần xem xét (Moderate)
    files_payload["22110007_Hoang_Minh_Tam_BienTapAI.txt"] = (
        "Knowledge representation and reasoning is the field of artificial intelligence dedicated to representing information about the world in a form that a computer system can use to solve complex tasks such as diagnosing a medical condition or having a dialog in a natural language. "
        "Knowledge representation incorporates findings from psychology about how humans solve problems and represent knowledge in order to design formalisms that will make complex systems easier to design and build. "
        "Examples of knowledge representation formalisms include semantic nets, systems architecture, frames, rules, and ontologies. "
        "Moreover, implementing these structured frameworks enhances analytical clarity and supports consistent decision-making across complex academic applications."
    ).encode("utf-8")

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "demo_samples"))
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, "Lop_K16_TieuLuan_Demo.zip")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, payload in files_payload.items():
            zf.writestr(fname, payload)

    print(f"[v] Đã tạo gói bài nộp mẫu demo đa định dạng tại: {zip_path} ({len(files_payload)} tệp)")
    return zip_path


if __name__ == "__main__":
    build_demo_zip()
