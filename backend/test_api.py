"""
Unit & Integration Test Suite cho FastAPI Backend Server & Web Application
Đồ án Chuyên ngành Kỹ thuật Phần mềm (Chủ đề 6)
Kiểm thử toàn bộ endpoints:
- GET  /                           : Phục vụ Web Dashboard HTML
- GET  /health                    : Kiểm tra trạng thái server & models
- POST /api/v1/detect             : Dự đoán văn bản đơn lẻ
- POST /api/v1/detect/sentences   : Phân tích cấp câu (AI Hotspots)
- POST /api/v1/detect/web         : Phân tích từ Web Dashboard
- POST /api/v1/detect/upload      : Tải tệp tài liệu (.txt, .docx, .pdf)
- POST /api/v1/detect/batch       : Dự đoán hàng loạt
- GET  /api/v1/history            : Nhật ký lịch sử thẩm định
- GET  /api/v1/history/{id}       : Chi tiết lượt thẩm định
- GET  /api/v1/stats              : Chỉ số KPI Dashboard
- GET  /api/v1/report/pdf/{id}    : Tải báo cáo PDF
- DELETE /api/v1/history/{id}     : Xóa bản ghi lịch sử
"""

import os
import sys
import io
import unittest

# Đảm bảo đường dẫn backend luôn có trong sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi.testclient import TestClient
from api import app


class TestDetectionAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n[*] Đang khởi động TestClient và nạp mô hình vào Lifespan...")
        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        print("\n[*] Đang đóng TestClient...")
        cls.client_cm.__exit__(None, None, None)

    def test_01_root_endpoint_dashboard(self):
        """Kiểm tra endpoint gốc GET / phục vụ Web Dashboard"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        # Kỳ vọng trả về file HTML
        self.assertIn("text/html", response.headers.get("content-type", ""))
        self.assertIn("VeriText AI", response.text)

    def test_02_health_endpoint(self):
        """Kiểm tra endpoint sức khỏe GET /health"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["ready"])
        self.assertTrue(data["models_loaded"]["bert"])
        self.assertTrue(data["models_loaded"]["roberta"])
        self.assertTrue(data["models_loaded"]["longformer"])
        self.assertIsNotNone(data["ensemble_weights"])

    def test_03_detect_endpoint(self):
        """Kiểm tra phân tích đoạn văn POST /api/v1/detect"""
        ai_sample = (
            "Artificial intelligence models have demonstrated notable capabilities in generating sophisticated text. "
            "These pre-trained architectures capture intricate syntactic and semantic structures across diverse linguistic domains."
        )
        response = self.client.post("/api/v1/detect", json={
            "text": ai_sample,
            "threshold": 0.5,
            "detailed": True,
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("label", data)
        self.assertIn("ai_probability", data)
        self.assertIn("confidence", data)
        self.assertIn("model_breakdown", data)

    def test_04_detect_sentences_and_db_save(self):
        """Kiểm tra phân tích cấp câu POST /api/v1/detect/sentences và lưu vào SQLite"""
        paragraph = (
            "Yesterday afternoon, I went to the park and enjoyed walking my dog. "
            "Meanwhile, large language models are transforming modern digital content creation with high fluency. "
            "It was a sunny and pleasant day overall."
        )
        response = self.client.post("/api/v1/detect/sentences", json={
            "text": paragraph,
            "threshold": 0.5,
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("scan_id", data)
        self.assertIn("overall", data)
        self.assertIn("sentences", data)
        self.assertGreaterEqual(len(data["sentences"]), 2)

        # Lưu lại scan_id để test tải PDF
        self.__class__.last_scan_id = data["scan_id"]

    def test_05_upload_text_file(self):
        """Kiểm tra tiếp nhận tệp tải lên POST /api/v1/detect/upload"""
        content = (
            "The quick brown fox jumps over the lazy dog. "
            "Artificial neural networks are computational models inspired by biological neural systems in the brain. "
            "They learn complex representations through iterative backpropagation."
        )
        file_bytes = io.BytesIO(content.encode("utf-8"))
        response = self.client.post(
            "/api/v1/detect/upload",
            files={"file": ("test_essay.txt", file_bytes, "text/plain")},
            data={"threshold": 0.5},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("scan_id", data)
        self.assertEqual(data["filename"], "test_essay.txt")
        self.assertIn("overall", data)

    def test_06_get_history_and_stats(self):
        """Kiểm tra truy xuất lịch sử và thống kê KPI"""
        stats_res = self.client.get("/api/v1/stats")
        self.assertEqual(stats_res.status_code, 200)
        stats = stats_res.json()
        self.assertIn("total_scans", stats)
        self.assertGreaterEqual(stats["total_scans"], 1)

        hist_res = self.client.get("/api/v1/history?limit=10")
        self.assertEqual(hist_res.status_code, 200)
        hist = hist_res.json()
        self.assertIn("history", hist)
        self.assertGreaterEqual(len(hist["history"]), 1)

    def test_07_export_pdf_report(self):
        """Kiểm tra sinh và tải báo cáo PDF GET /api/v1/report/pdf/{scan_id}"""
        scan_id = getattr(self.__class__, "last_scan_id", 1)
        response = self.client.get(f"/api/v1/report/pdf/{scan_id}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("content-type"), "application/pdf")
        self.assertGreater(len(response.content), 1000)

    def test_08_batch_detect(self):
        """Kiểm tra phân tích hàng loạt POST /api/v1/detect/batch"""
        texts = [
            "This is the first simple short test sentence.",
            "Deep neural networks continue to excel in complex classification benchmarks across natural language processing."
        ]
        response = self.client.post("/api/v1/detect/batch", json={
            "texts": texts,
            "threshold": 0.5,
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_items"], 2)

    def test_09_vietnamese_detection(self):
        """Kiểm tra nhận diện và định tuyến văn bản Tiếng Việt qua PhoBERT"""
        vi_sample = (
            "Trí tuệ nhân tạo ngày nay đang đóng vai trò then chốt trong sự phát triển của công nghệ toàn cầu. "
            "Các mô hình xử lý ngôn ngữ tự nhiên hiện đại có khả năng tiếp nhận và xử lý lượng thông tin khổng lồ."
        )
        response = self.client.post("/api/v1/detect", json={
            "text": vi_sample,
            "threshold": 0.5,
            "detailed": True,
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["language"], "vi")
        self.assertIn("PhoBERT", data["detected_model"])
        self.assertIn("ai_probability", data)

    def test_10_vietnamese_sentence_analysis(self):
        """Kiểm tra bóc tách cấp câu Tiếng Việt và lưu CSDL"""
        vi_paragraph = (
            "Bộ Giáo dục và Đào tạo vừa công bố phương án thi tốt nghiệp trung học phổ thông mới. "
            "Học sinh sẽ thi hai môn bắt buộc và hai môn tự chọn theo định hướng nghề nghiệp cá nhân."
        )
        response = self.client.post("/api/v1/detect/sentences", json={
            "text": vi_paragraph,
            "threshold": 0.5,
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["overall"]["language"], "vi")
        self.assertGreaterEqual(len(data["sentences"]), 2)

        # Kiểm tra xuất PDF tiếng Việt có lưu full_text và XML escape an toàn
        vi_scan_id = data["scan_id"]
        pdf_res = self.client.get(f"/api/v1/report/pdf/{vi_scan_id}")
        self.assertEqual(pdf_res.status_code, 200)
        self.assertEqual(pdf_res.headers.get("content-type"), "application/pdf")

    def test_11_auth_rbac_pdf_and_language_routing_guard(self):
        """Kiểm tra đăng nhập RBAC, cô lập PDF/History, lưu full_text và chặn nhận nhầm ngôn ngữ"""
        # 1. Đăng nhập sinh viên & giảng viên
        s_res = self.client.post("/api/v1/auth/login", json={
            "email": "sinhvien@veritext.ai", "password": "123456"
        })
        self.assertEqual(s_res.status_code, 200)
        s_token = s_res.json()["access_token"]

        t_res = self.client.post("/api/v1/auth/login", json={
            "email": "giangvien@veritext.ai", "password": "123456"
        })
        self.assertEqual(t_res.status_code, 200)
        t_token = t_res.json()["access_token"]

        # 2. Quét văn bản tiếng Anh có chứa tên riêng tiếng Việt & ký tự XML (<tag> & symbol)
        en_with_vn_name = (
            "Research by Dr. Nguyễn shows that transformer models with <attention> & self-regulation "
            "produce remarkably consistent syntactic structures across academic essays."
        )
        scan_res = self.client.post(
            "/api/v1/detect/web",
            json={"text": en_with_vn_name, "threshold": 0.5, "title": "Paper <A> & B"},
            headers={"Authorization": f"Bearer {t_token}"},
        )
        self.assertEqual(scan_res.status_code, 200)
        scan_data = scan_res.json()
        # Phải giữ đúng tuyến Tiếng Anh dù có 1 từ 'Nguyễn'
        self.assertEqual(scan_data["overall"]["language"], "en")
        teacher_scan_id = scan_data["scan_id"]

        # 3. Sinh viên KHÔNG được xem chi tiết hoặc tải PDF của giảng viên
        s_detail = self.client.get(
            f"/api/v1/history/{teacher_scan_id}",
            headers={"Authorization": f"Bearer {s_token}"},
        )
        self.assertEqual(s_detail.status_code, 404)

        s_pdf = self.client.get(
            f"/api/v1/report/pdf/{teacher_scan_id}",
            headers={"Authorization": f"Bearer {s_token}"},
        )
        self.assertEqual(s_pdf.status_code, 404)

        # 4. Giảng viên xem được full_text và tải được PDF (kể cả chứa ký tự < > &)
        t_detail = self.client.get(
            f"/api/v1/history/{teacher_scan_id}",
            headers={"Authorization": f"Bearer {t_token}"},
        )
        self.assertEqual(t_detail.status_code, 200)
        detail_json = t_detail.json()
        self.assertEqual(detail_json["full_text"], en_with_vn_name)
        self.assertEqual(detail_json["title"], "Paper <A> & B")

        t_pdf = self.client.get(
            f"/api/v1/report/pdf/{teacher_scan_id}",
            headers={"Authorization": f"Bearer {t_token}"},
        )
        self.assertEqual(t_pdf.status_code, 200)
        self.assertEqual(t_pdf.headers.get("content-type"), "application/pdf")

        # 5. Dọn bản ghi test để giữ sạch DB
        del_res = self.client.delete(
            f"/api/v1/history/{teacher_scan_id}",
            headers={"Authorization": f"Bearer {t_token}"},
        )
        self.assertEqual(del_res.status_code, 200)

    def test_12_forensics_bypass_filters_and_xai_pdf(self):
        """Kiểm tra toàn luồng Giai đoạn 1: Anti-Bypass + Lọc Tài liệu tham khảo + XAI + Lưu SQL + Xuất PDF."""
        poisoned_vi = (
            "Trí\u200b tuệ\u200b nhân t\u0430o đóng v\u0430i trò then chốt trong bối cảnh hiện nay. "
            "Không thể phủ nhận rằng các mô hình ngôn ngữ lớn mang lại nhiều lợi ích vượt trội. "
            "Tóm lại, việc xây dựng hệ thống kiểm định là vô cùng cấp thiết.\n\n"
            "Tài liệu tham khảo:\n"
            "[1] Vaswani et al. (2017). Attention Is All You Need."
        )
        res = self.client.post("/api/v1/detect/web", json={
            "text": poisoned_vi,
            "threshold": 0.5,
            "strictness_preset": "strict",
            "filter_quotes": True,
            "filter_references": True,
            "title": "Test Phase 1 Bypass & XAI",
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()

        # 1. Anti-Bypass phải bật cảnh báo
        self.assertTrue(data["bypass_report"]["has_bypass_warning"])
        self.assertGreaterEqual(data["bypass_report"]["zero_width_count"], 2)
        self.assertGreaterEqual(data["bypass_report"]["homoglyph_count"], 2)

        # 2. Bộ lọc học thuật phải cắt mục Tài liệu tham khảo
        self.assertTrue(data["filter_report"]["references_stripped"])

        # 3. Preset strict phải đặt threshold = 0.30
        self.assertAlmostEqual(data["overall"]["threshold_used"], 0.30)

        # 4. XAI metrics & 4-tier authorship phải đầy đủ
        self.assertIn("burstiness_score", data["xai_metrics"])
        self.assertIn("perplexity_score", data["xai_metrics"])
        self.assertGreaterEqual(data["xai_metrics"]["ai_cliche_count"], 3)
        self.assertIn(data["authorship_tier"]["code"], ("AI_PURE", "HYBRID_MIXED", "AI_ASSISTED", "HUMAN_PURE"))

        # 5. Đọc lại từ SQL Server và xuất PDF có chứa XAI + Anti-Bypass
        scan_id = data["scan_id"]
        detail = self.client.get(f"/api/v1/history/{scan_id}").json()
        self.assertTrue(detail["has_bypass_warning"])
        self.assertIsNotNone(detail["burstiness_score"])
        self.assertIsNotNone(detail["perplexity_score"])
        self.assertIsNotNone(detail["xai_metrics"])

        pdf_res = self.client.get(f"/api/v1/report/pdf/{scan_id}")
        self.assertEqual(pdf_res.status_code, 200)
        self.assertEqual(pdf_res.headers.get("content-type"), "application/pdf")
        self.assertGreater(len(pdf_res.content), 2000)

        # Dọn bản ghi test
        self.client.delete(f"/api/v1/history/{scan_id}")


if __name__ == "__main__":
    unittest.main()
