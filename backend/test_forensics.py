"""
Unit Tests cho Module Pháp y Văn bản, Anti-Bypass, Bộ lọc Học thuật & XAI (text_forensics.py)
Đồ án Chuyên ngành Kỹ thuật Phần mềm (Chủ đề 6): VeriText AI
"""

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import text_forensics as tfx


class TestTextForensics(unittest.TestCase):
    def test_01_detect_zero_width_and_homoglyph_bypass(self):
        """Phát hiện ký tự tàng hình Zero-Width và Cyrillic Homoglyph giả Latin, đồng thời khôi phục văn bản sạch."""
        poisoned = (
            "Trí\u200b tuệ\u200b nhân t\u0430o (AI) đóng v\u0430i trò then chốt tr\u043eng bối c\u0430nh chuyển đổi số."
        )
        report = tfx.detect_bypass_attempts(poisoned)
        self.assertTrue(report["has_bypass_warning"])
        self.assertEqual(report["zero_width_count"], 2)
        self.assertEqual(report["homoglyph_count"], 4)
        self.assertIn(report["risk_level"], ("MEDIUM", "HIGH"))
        self.assertNotIn("\u200b", report["sanitized_text"])
        self.assertNotIn("\u0430", report["sanitized_text"])
        self.assertIn("nhân tao", report["sanitized_text"])
        self.assertIn("trong bối canh", report["sanitized_text"])
        self.assertGreaterEqual(len(report["flagged_samples"]), 2)

    def test_02_clean_text_has_no_bypass_warning(self):
        """Văn bản sạch bình thường không bị báo động nhầm Anti-Bypass."""
        clean = "Hôm nay trời mát mẻ, chúng tôi cùng nhau chuẩn bị báo cáo đồ án chuyên ngành kỹ thuật phần mềm."
        report = tfx.detect_bypass_attempts(clean)
        self.assertFalse(report["has_bypass_warning"])
        self.assertEqual(report["risk_level"], "NONE")
        self.assertEqual(report["total_anomalies"], 0)

    def test_03_academic_filters_quotes_and_references(self):
        """Bộ lọc học thuật bỏ đúng trích dẫn dài trong ngoặc kép và mục Tài liệu tham khảo cuối bài."""
        essay = (
            "Trong nghiên cứu này chúng tôi đánh giá hiệu năng của các mô hình ngôn ngữ lớn. "
            'Theo tác giả Nguyễn Văn A: "Trí tuệ nhân tạo đang tạo ra sự thay đổi mang tính cách mạng trong toàn bộ ngành giáo dục đại học hiện đại ngày nay." '
            "Kết quả thực nghiệm của nhóm cho thấy độ chính xác cải thiện rõ rệt trên tập kiểm thử độc lập.\n\n"
            "Tài liệu tham khảo:\n"
            "[1] Vaswani et al. (2017). Attention Is All You Need.\n"
            "[2] Nguyen & Nguyen (2020). PhoBERT."
        )
        res = tfx.apply_academic_filters(essay, filter_quotes=True, filter_references=True)
        self.assertTrue(res["filter_applied"])
        self.assertEqual(res["quotes_removed_count"], 1)
        self.assertTrue(res["references_stripped"])
        self.assertNotIn("Vaswani", res["filtered_text"])
        self.assertNotIn("cách mạng trong toàn bộ", res["filtered_text"])
        self.assertIn("Kết quả thực nghiệm của nhóm", res["filtered_text"])

    def test_04_linguistic_xai_vi_and_en(self):
        """Tính Burstiness, MATTR, Perplexity Proxy và bắt cụm từ văn mẫu AI song ngữ."""
        vi_ai = (
            "Trí tuệ nhân tạo đóng vai trò then chốt trong bối cảnh hiện nay. "
            "Không thể phủ nhận rằng công nghệ mang lại nhiều lợi ích vượt trội. "
            "Bên cạnh đó, hệ thống cũng đặt ra nhiều thách thức về bản quyền. "
            "Tóm lại, việc ứng dụng giải pháp kiểm định là vô cùng cấp thiết."
        )
        xai_vi = tfx.analyze_linguistic_xai(vi_ai, language="vi")
        self.assertGreaterEqual(xai_vi["ai_cliche_count"], 4)
        self.assertGreater(xai_vi["burstiness_score"], 0.0)
        self.assertGreater(xai_vi["perplexity_score"], 0.0)
        self.assertGreater(xai_vi["lexical_diversity"], 0.0)
        self.assertGreaterEqual(len(xai_vi["explanation_bullets"]), 2)

        en_ai = (
            "Furthermore, it is crucial to note that artificial intelligence plays a crucial role in modern education. "
            "In conclusion, developing multifaceted detection systems is of paramount importance."
        )
        xai_en = tfx.analyze_linguistic_xai(en_ai, language="en")
        self.assertGreaterEqual(xai_en["ai_cliche_count"], 4)

    def test_05_authorship_tier_classification(self):
        """Kiểm tra phân loại 4 mức độ can thiệp (Pure AI / Hybrid / Assisted / Human)."""
        # Hybrid: 2 câu đỏ xen kẽ 2 câu xanh
        hybrid_sents = [
            {"highlight_color": "green"},
            {"highlight_color": "red"},
            {"highlight_color": "red"},
            {"highlight_color": "green"},
        ]
        tier_hybrid = tfx.classify_authorship_tier(0.65, hybrid_sents, threshold=0.5)
        self.assertEqual(tier_hybrid["code"], "HYBRID_MIXED")

        # Pure AI
        ai_sents = [{"highlight_color": "red"}] * 4
        tier_ai = tfx.classify_authorship_tier(0.95, ai_sents, threshold=0.5)
        self.assertEqual(tier_ai["code"], "AI_PURE")

        # Pure Human
        human_sents = [{"highlight_color": "green"}] * 4
        tier_human = tfx.classify_authorship_tier(0.12, human_sents, threshold=0.5)
        self.assertEqual(tier_human["code"], "HUMAN_PURE")

    def test_06_resolve_threshold_presets(self):
        """Kiểm tra chuyển đổi preset Khóa luận / Tiểu luận / Tham khảo."""
        th_strict, _ = tfx.resolve_threshold(0.5, "strict")
        th_bal, _ = tfx.resolve_threshold(0.5, "balanced")
        th_len, _ = tfx.resolve_threshold(0.5, "lenient")
        self.assertAlmostEqual(th_strict, 0.30)
        self.assertAlmostEqual(th_bal, 0.50)
        self.assertAlmostEqual(th_len, 0.70)

    def test_07_leading_utf8_bom_not_false_positive(self):
        """UTF-8 BOM hợp lệ ở đầu file Notepad không bị báo oan là Anti-Bypass, nhưng BOM giữa văn bản vẫn bị bắt."""
        clean_bom = "\ufeffĐây là một bài tiểu luận sạch được lưu từ Windows Notepad bằng UTF-8 BOM."
        rep_clean = tfx.detect_bypass_attempts(clean_bom)
        self.assertFalse(rep_clean["has_bypass_warning"])
        self.assertEqual(rep_clean["zero_width_count"], 0)

        mid_bom = "Đây là văn bản cố tình chèn\ufeff ký tự ẩn ở giữa câu."
        rep_mid = tfx.detect_bypass_attempts(mid_bom)
        self.assertTrue(rep_mid["has_bypass_warning"])
        self.assertEqual(rep_mid["zero_width_count"], 1)

    def test_08_smart_sentence_split_preserves_numbered_headings(self):
        """Bộ tách câu XAI tách chính xác theo xuống dòng mà vẫn giữ nguyên tiền tố đánh số tiêu đề mục."""
        text = "1. Mở đầu\nHôm qua trời nắng nhẹ. Cả nhóm đi bộ dọc công viên.\n2. Kết luận\nBài làm hoàn toàn tự nhiên."
        sents = tfx.split_sentences_smart(text)
        self.assertEqual(sents[0], "1. Mở đầu")
        self.assertEqual(sents[1], "Hôm qua trời nắng nhẹ.")
        self.assertEqual(sents[3], "2. Kết luận")
        self.assertEqual(len(sents), 5)


if __name__ == "__main__":
    unittest.main()
