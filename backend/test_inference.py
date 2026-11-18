"""
Unit Test & Integration Test cho Bộ Suy Luận AI Text Detector (Ngày 13)
Kiểm tra tính chính xác, tính toàn vẹn của kết quả và thời gian phản hồi.
"""

import os
import unittest
import pandas as pd
from inference import AITextDetector


class TestAITextDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n[*] Đang khởi tạo AITextDetector cho Unit Tests...")
        cls.detector = AITextDetector(models_dir="output")

        # Nạp dữ liệu kiểm thử thực tế từ test.csv
        test_csv_path = os.path.join("data", "test.csv")
        if os.path.exists(test_csv_path):
            df = pd.read_csv(test_csv_path)
            cls.human_sample = df[df["label"] == 0]["text"].iloc[0]
            cls.ai_sample = df[df["label"] == 1]["text"].iloc[0]
        else:
            cls.human_sample = "Yesterday evening, my family gathered for dinner and we watched a local football game. The weather was unusually chilly for this time of year."
            cls.ai_sample = (
                "Artificial intelligence and deep learning algorithms have revolutionized natural language understanding. "
                "These sophisticated neural models exhibit unprecedented accuracy in contextual generation and stylistic synthesis."
            )

    def test_detector_initialization(self):
        """Kiểm tra các mô hình và trọng số được nạp đầy đủ."""
        self.assertIsNotNone(self.detector.model_bert, "Mô hình BERT chưa được nạp.")
        self.assertIsNotNone(self.detector.model_roberta, "Mô hình RoBERTa chưa được nạp.")
        self.assertIsNotNone(self.detector.model_longformer, "Mô hình Longformer chưa được nạp.")
        self.assertEqual(len(self.detector.weights), 3, "Trọng số Ensemble phải gồm 3 giá trị.")
        self.assertAlmostEqual(float(sum(self.detector.weights)), 1.0, places=4, msg="Tổng trọng số Ensemble phải bằng 1.0.")

    def test_predict_schema(self):
        """Kiểm tra cấu trúc dict trả về của phương thức predict."""
        text = "This is a brief test statement to verify output dictionary schema."
        result = self.detector.predict(text, detailed=True)

        required_keys = [
            "label", "label_vi", "ai_probability", "human_probability",
            "ai_percentage", "confidence", "confidence_level",
            "threshold_used", "text_stats", "inference_time_ms", "model_breakdown"
        ]
        for k in required_keys:
            self.assertIn(k, result, f"Thiếu key '{k}' trong kết quả trả về.")

        self.assertIn(result["label"], ["AI-generated", "Human-written"])
        self.assertGreaterEqual(result["ai_probability"], 0.0)
        self.assertLessEqual(result["ai_probability"], 1.0)
        self.assertAlmostEqual(result["ai_probability"] + result["human_probability"], 1.0, places=4)

    def test_analyze_sentences(self):
        """Kiểm tra chức năng phân tích cấp câu và highlight màu sắc."""
        paragraph = (
            "The movie premiere took place in Los Angeles last night. "
            "However, artificial intelligence systems continue to generate synthetic discourse with unparalleled grammatical coherence. "
            "Fans waited outside the cinema for several hours hoping to see the cast."
        )
        sentences = self.detector.analyze_sentences(paragraph)
        self.assertGreaterEqual(len(sentences), 2, "Phải phân tách được ít nhất 2 câu.")

        for s in sentences:
            self.assertIn("sentence_index", s)
            self.assertIn("sentence", s)
            self.assertIn("ai_probability", s)
            self.assertIn("highlight_color", s)
            self.assertIn(s["highlight_color"], ["green", "yellow", "red"])

    def test_empty_input(self):
        """Kiểm tra xử lý ngoại lệ khi văn bản rỗng."""
        result = self.detector.predict("   ")
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()

