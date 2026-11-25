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
        self.assertIn("AUDIT-000101", csv_text)


if __name__ == "__main__":
    unittest.main()
