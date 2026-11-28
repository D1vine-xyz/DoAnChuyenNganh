"""
Unit Tests cho Module Chấm Bài Cả Lớp (.ZIP / Multi-file) & Xuất Excel/CSV (batch_processor.py)
"""

import io
import os
import sys
import unittest
import zipfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import batch_processor as bp


class TestBatchProcessor(unittest.TestCase):
    def test_01_safe_zip_extraction_and_zip_slip_guard(self):
        """Giải nén ZIP hợp lệ, chặn đường dẫn Zip Slip và bỏ qua rác __MACOSX / file quá ngắn."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("22110001_Nguyen_Van_A.txt", "Đây là một bài tiểu luận hợp lệ có nhiều hơn năm từ để kiểm tra.".encode("utf-8"))
            zf.writestr("__MACOSX/._hidden.txt", b"junk macos file with many words here ignored")
            zf.writestr("../../evil_path.txt", b"Attempted path traversal attack should be ignored safely")
            zf.writestr("short.txt", b"Too short")
            zf.writestr("image.png", b"\x89PNG\r\n")

        valid_docs, skipped = bp.extract_files_from_uploads([("class_batch.zip", buf.getvalue())])
        self.assertEqual(len(valid_docs), 1)
        self.assertEqual(valid_docs[0]["filename"], "22110001_Nguyen_Van_A.txt")
        self.assertEqual(valid_docs[0]["student_label"], "22110001 Nguyen Van A")
        self.assertGreaterEqual(len(skipped), 1)

    def test_02_excel_and_csv_export_generation(self):
        """Sinh file .XLSX hợp lệ (2 sheet) và .CSV chuẩn UTF-8 BOM không lỗi font."""
        sample_batch = {
            "id": 7,
            "batch_name": "K16 - Kiểm thử phần mềm",
            "teacher_name": "TS. Hoàng Văn Thầy",
            "created_at": "2026-11-26 00:15:00",
            "total_files": 2,
            "high_risk_count": 1,
            "moderate_risk_count": 0,
            "safe_count": 1,
            "bypass_count": 1,
            "avg_ai_probability": 0.55,
            "submissions": [
                {
                    "scan_id": 101,
                    "title": "sv1.txt",
                    "student_label": "22110001 Nguyen Van A",
                    "language": "vi",
                    "verdict": "AI-generated",
                    "ai_probability": 0.94,
                    "word_count": 150,
                    "burstiness_score": 22.5,
                    "perplexity_score": 31.0,
                    "has_bypass_warning": True,
                    "authorship_tier": "AI_PURE",
                    "risk_level": "high",
                    "inference_time_ms": 145.2,
                },
                {
                    "scan_id": 102,
                    "title": "sv2.txt",
                    "student_label": "22110002 Tran Thi B",
                    "language": "en",
                    "verdict": "Human-written",
                    "ai_probability": 0.16,
                    "word_count": 180,
                    "burstiness_score": 71.0,
                    "perplexity_score": 78.5,
                    "has_bypass_warning": False,
                    "authorship_tier": "HUMAN_PURE",
                    "risk_level": "safe",
                    "inference_time_ms": 210.4,
                },
            ],
        }

        xlsx_bytes = bp.generate_batch_excel_bytes(sample_batch)
        self.assertTrue(xlsx_bytes[:2] == b"PK")  # Chuẩn nén ZIP của định dạng .xlsx
        from openpyxl import load_workbook
        wb = load_workbook(io.BytesIO(xlsx_bytes))
        self.assertIn("Bang_Diem_Kiem_Dinh", wb.sheetnames)
        self.assertIn("Tong_Quan_Lop_Hoc", wb.sheetnames)
        ws = wb["Bang_Diem_Kiem_Dinh"]
        self.assertEqual(ws["B5"].value, "22110001 Nguyen Van A")
        self.assertEqual(ws["H5"].value, "CÓ (Lách luật)")

        csv_bytes = bp.generate_batch_csv_bytes(sample_batch)
        self.assertTrue(csv_bytes.startswith(b"\xef\xbb\xbf"))  # UTF-8 BOM
        csv_text = csv_bytes.decode("utf-8-sig")
        self.assertIn("22110001 Nguyen Van A", csv_text)
        self.assertRegex(csv_text, r"VT-000101-[A-F0-9]{8}")

    def test_03_docx_tables_utf8_bom_and_demo_zip(self):
        """Kiểm tra đọc bảng trong .DOCX, xử lý .TXT UTF-8 BOM, phân rủi ro theo ngưỡng và giải nén file Demo ZIP thật."""
        import docx
        from file_parser import extract_text_from_file

        # 1. File .TXT có UTF-8 BOM không giữ ký tự \ufeff ở đầu
        bom_txt = "\ufeffBài tiểu luận kiểm thử định dạng Windows Notepad hợp lệ.".encode("utf-8")
        txt_out, ext = extract_text_from_file("notepad_bom.txt", bom_txt)
        self.assertEqual(ext, "txt")
        self.assertFalse(txt_out.startswith("\ufeff"))

        # 1b. Kiểm tra bóc tách nhãn sinh viên từ thư mục Moodle/LMS ZIP
        moodle_lbl = bp.parse_student_label("22110001_NguyenVanA_88421_assignsubmission_file_/bai_nop.docx")
        self.assertIn("22110001 NguyenVanA", moodle_lbl)
        self.assertIn("bai nop", moodle_lbl)

        # 2. File .DOCX chỉ có nội dung trong bảng (Table) vẫn phải trích xuất được
        doc = docx.Document()
        tbl = doc.add_table(rows=1, cols=2)
        tbl.cell(0, 0).text = "Sinh viên: Nguyễn Văn A"
        tbl.cell(0, 1).text = "Nội dung bài làm nằm hoàn toàn trong bảng biểu học thuật."
        docx_buf = io.BytesIO()
        doc.save(docx_buf)
        docx_text, ext2 = extract_text_from_file("table_essay.docx", docx_buf.getvalue())
        self.assertEqual(ext2, "docx")
        self.assertIn("bảng biểu học thuật", docx_text)

        # 3. Phân rủi ro nhất quán theo ngưỡng strict (30%)
        self.assertEqual(bp.classify_risk_level(0.38, False, threshold=0.30), "moderate")
        self.assertEqual(bp.classify_risk_level(0.25, False, threshold=0.30), "safe")
        self.assertEqual(bp.classify_risk_level(0.20, True, threshold=0.50), "high")

        # 4. Giải nén thử bộ dữ liệu Demo vấn đáp thật (docs/demo_samples/Lop_K16_TieuLuan_Demo.zip)
        demo_zip_path = os.path.abspath(os.path.join(BASE_DIR, "..", "docs", "demo_samples", "Lop_K16_TieuLuan_Demo.zip"))
        if os.path.exists(demo_zip_path):
            with open(demo_zip_path, "rb") as f:
                valid_docs, _ = bp.extract_files_from_uploads([("Lop_K16_TieuLuan_Demo.zip", f.read())])
            self.assertGreaterEqual(len(valid_docs), 5)

    def test_04_ocr_image_and_scanned_pdf_extraction(self):
        """Kiểm tra trích xuất OCR ngoại tuyến từ ảnh (.PNG) và PDF ảnh scan (không có lớp text)."""
        from PIL import Image, ImageDraw, ImageFont
        import pymupdf
        from file_parser import extract_text_with_meta

        img = Image.new("RGB", (1300, 320), color="white")
        draw = ImageDraw.Draw(img)
        font_path = r"C:\Windows\Fonts\arial.ttf"
        font = ImageFont.truetype(font_path, 34) if os.path.exists(font_path) else ImageFont.load_default()
        draw.text((30, 50), "Trí tuệ nhân tạo đóng vai trò then chốt trong giáo dục đại học.", fill="black", font=font)
        draw.text((30, 140), "Artificial intelligence transforms academic research and software testing.", fill="black", font=font)

        png_buf = io.BytesIO()
        img.save(png_buf, format="PNG")
        png_bytes = png_buf.getvalue()

        # 1. Test OCR trực tiếp từ ảnh .PNG
        img_meta = extract_text_with_meta("anh_chup_bai_lam.png", png_bytes)
        self.assertTrue(img_meta["ocr_used"])
        self.assertEqual(img_meta["extension"], "png")
        self.assertIn("giáo dục", img_meta["text"].lower())
        self.assertIn("artificial intelligence", img_meta["text"].lower())

        # 2. Tạo PDF ảnh scan thuần (chỉ chèn raster image, không có lớp text)
        pdf_doc = pymupdf.open()
        page = pdf_doc.new_page(width=1300, height=320)
        page.insert_image(page.rect, stream=png_bytes)
        scanned_pdf_bytes = pdf_doc.tobytes()
        pdf_doc.close()

        pdf_meta = extract_text_with_meta("bai_nop_scan.pdf", scanned_pdf_bytes)
        self.assertTrue(pdf_meta["ocr_used"])
        self.assertEqual(pdf_meta["extension"], "pdf")
        self.assertEqual(pdf_meta["ocr_pages"], 1)
        self.assertIn("artificial intelligence", pdf_meta["text"].lower())


if __name__ == "__main__":
    unittest.main()
