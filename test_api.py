"""
Unit & Integration Test Suite cho FastAPI Backend Server (Ngày 14)
Kiểm thử toàn bộ các endpoints: /health, /api/v1/detect, /api/v1/detect/sentences, /api/v1/detect/batch.
"""

import unittest
from fastapi.testclient import TestClient
from api import app


class TestDetectionAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n[*] Đang khởi động TestClient và nạp mô hình vào Lifespan...")
        # Sử dụng TestClient với context manager để kích hoạt lifespan
        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        print("\n[*] Đang đóng TestClient...")
        cls.client_cm.__exit__(None, None, None)

    def test_01_root_endpoint(self):
        """Kiểm tra endpoint gốc GET /"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("docs_url", data)

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
        self.assertGreaterEqual(data["ai_probability"], 0.0)
        self.assertLessEqual(data["ai_probability"], 1.0)

    def test_04_detect_sentences_endpoint(self):
        """Kiểm tra phân tích cấp câu phục vụ Chrome Extension POST /api/v1/detect/sentences"""
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

        self.assertIn("overall", data)
        self.assertIn("sentence_summary", data)
        self.assertIn("sentences", data)
        self.assertGreaterEqual(len(data["sentences"]), 2)

        for s in data["sentences"]:
            self.assertIn("sentence_index", s)
            self.assertIn("sentence", s)
            self.assertIn("ai_probability", s)
            self.assertIn("highlight_color", s)
            self.assertIn(s["highlight_color"], ["green", "yellow", "red"])

    def test_05_batch_detect_endpoint(self):
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
        self.assertEqual(len(data["results"]), 2)

    def test_06_validation_error_empty_text(self):
        """Kiểm tra bắt lỗi validation khi văn bản quá ngắn"""
        response = self.client.post("/api/v1/detect", json={"text": "hi"})
        # Pydantic min_length=3 -> "hi" is 2 chars -> 422 Unprocessable Entity
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()

