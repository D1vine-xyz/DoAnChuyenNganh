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

import io
import unittest
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
        self.assertIn("AI Text Detector", response.text)

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


if __name__ == "__main__":
    unittest.main()
