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
        """Kiểm tra phân tích cấp câu POST /api/v1/detect/sentences và lưu vào SQL Server"""
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

        # 3. Sinh viên & Khách (chưa đăng nhập) KHÔNG được xem chi tiết, tải PDF hoặc xóa bài riêng của giảng viên
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

        g_detail = self.client.get(f"/api/v1/history/{teacher_scan_id}")
        self.assertEqual(g_detail.status_code, 404)

        g_pdf = self.client.get(f"/api/v1/report/pdf/{teacher_scan_id}")
        self.assertEqual(g_pdf.status_code, 404)

        g_del = self.client.delete(f"/api/v1/history/{teacher_scan_id}")
        self.assertEqual(g_del.status_code, 404)

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

    def test_13_classroom_batch_zip_rbac_and_excel_export(self):
        """Kiểm tra toàn luồng Giai đoạn 2: Chặn Sinh viên (403), Giảng viên chấm .ZIP, lưu SQL & Xuất Excel/CSV."""
        import zipfile

        # 1. Lấy token Sinh viên và Giảng viên
        s_token = self.client.post("/api/v1/auth/login", json={
            "email": "sinhvien@veritext.ai", "password": "123456"
        }).json()["access_token"]
        t_token = self.client.post("/api/v1/auth/login", json={
            "email": "giangvien@veritext.ai", "password": "123456"
        }).json()["access_token"]

        # Tạo 1 file .ZIP mẫu 2 bài nộp (1 bài cài ký tự ẩn Bypass, 1 bài tự viết)
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr(
                "22110001_SV_Bypass.txt",
                "Trí\u200b tuệ\u200b nhân t\u0430o đóng v\u0430i trò then chốt trong bối cảnh hiện nay. Không thể phủ nhận lợi ích vượt trội.".encode("utf-8"),
            )
            zf.writestr(
                "22110002_SV_Human.txt",
                "Last weekend my friends and I went hiking near the lake and cooked instant noodles by the trail.".encode("utf-8"),
            )
        zip_bytes = zip_buf.getvalue()

        # 2. Sinh viên gọi API chấm lớp học phải bị từ chối 403 Forbidden
        s_resp = self.client.post(
            "/api/v1/classroom/batch-scan",
            files=[("files", ("test_lop.zip", io.BytesIO(zip_bytes), "application/zip"))],
            data={"batch_name": "Unauthorized Batch"},
            headers={"Authorization": f"Bearer {s_token}"},
        )
        self.assertEqual(s_resp.status_code, 403)

        # 3. Giảng viên chấm file .ZIP thành công
        t_resp = self.client.post(
            "/api/v1/classroom/batch-scan",
            files=[("files", ("test_lop.zip", io.BytesIO(zip_bytes), "application/zip"))],
            data={"batch_name": "K16 Test Batch", "strictness_preset": "balanced", "filter_references": "true"},
            headers={"Authorization": f"Bearer {t_token}"},
        )
        self.assertEqual(t_resp.status_code, 200)
        b_data = t_resp.json()["batch"]
        batch_id = b_data["id"]
        self.assertEqual(b_data["total_files"], 2)
        self.assertGreaterEqual(b_data["bypass_count"], 1)
        self.assertEqual(len(b_data["submissions"]), 2)
        # Bài dính bypass phải được xếp lên đầu danh sách rủi ro
        self.assertTrue(b_data["submissions"][0]["has_bypass_warning"])

        # 4. Xuất Excel (.xlsx) và CSV (.csv) thành công với token Giảng viên
        xlsx_res = self.client.get(
            f"/api/v1/classroom/batches/{batch_id}/export?format=xlsx",
            headers={"Authorization": f"Bearer {t_token}"},
        )
        self.assertEqual(xlsx_res.status_code, 200)
        self.assertTrue(xlsx_res.content[:2] == b"PK")

        csv_res = self.client.get(
            f"/api/v1/classroom/batches/{batch_id}/export?format=csv",
            headers={"Authorization": f"Bearer {t_token}"},
        )
        self.assertEqual(csv_res.status_code, 200)
        self.assertTrue(csv_res.content.startswith(b"\xef\xbb\xbf"))

        # 5. Sinh viên không được xuất bảng điểm Excel (403)
        s_export = self.client.get(
            f"/api/v1/classroom/batches/{batch_id}/export?format=xlsx",
            headers={"Authorization": f"Bearer {s_token}"},
        )
        self.assertEqual(s_export.status_code, 403)

        # 6. Dọn sạch đợt test và 2 bản ghi scan con
        for sub in b_data["submissions"]:
            self.client.delete(f"/api/v1/history/{sub['scan_id']}", headers={"Authorization": f"Bearer {t_token}"})
        del_batch = self.client.delete(
            f"/api/v1/classroom/batches/{batch_id}",
            headers={"Authorization": f"Bearer {t_token}"},
        )
        self.assertEqual(del_batch.status_code, 200)

    def test_14_complete_auth_profile_password_and_google_oauth(self):
        """Kiểm tra toàn diện Đăng ký (confirm_password, email validation), Đổi mật khẩu, Cập nhật hồ sơ & Google OAuth."""
        from unittest.mock import patch
        import database as db_mod

        # 1. Kiểm tra /api/v1/auth/config
        cfg = self.client.get("/api/v1/auth/config")
        self.assertEqual(cfg.status_code, 200)
        self.assertIn("google_enabled", cfg.json())

        # 2. Đăng ký sai confirm_password -> 400
        bad_reg = self.client.post("/api/v1/auth/register", json={
            "email": "test_auth_user@veritext.ai",
            "password": "Password123!",
            "confirm_password": "MismatchPassword",
            "full_name": "Test Auth User",
            "role": "student",
        })
        self.assertEqual(bad_reg.status_code, 400)

        # 3. Đăng ký hợp lệ
        test_email = "test_auth_complete@veritext.ai"
        with db_mod.get_connection() as conn:
            conn.cursor().execute("DELETE FROM users WHERE email IN (?, ?);", (test_email, "google_demo@gmail.com"))
            conn.commit()

        reg = self.client.post("/api/v1/auth/register", json={
            "email": test_email,
            "password": "OldPassword123",
            "confirm_password": "OldPassword123",
            "full_name": "Nguyen Test User",
            "role": "student",
        })
        self.assertEqual(reg.status_code, 200)
        token = reg.json()["access_token"]

        # 4. Cập nhật hồ sơ & đổi vai trò -> trả JWT mới
        upd = self.client.put(
            "/api/v1/auth/profile",
            json={"full_name": "TS. Nguyen Updated", "role": "teacher"},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(upd.status_code, 200)
        self.assertEqual(upd.json()["user"]["full_name"], "TS. Nguyen Updated")
        self.assertEqual(upd.json()["user"]["role"], "teacher")
        token = upd.json()["access_token"]

        # 5. Đổi mật khẩu -> sai mật khẩu cũ phải báo 400, đúng mật khẩu cũ phải thành công
        bad_pwd = self.client.post(
            "/api/v1/auth/change-password",
            json={"old_password": "WrongOldPassword", "new_password": "NewPassword456!"},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(bad_pwd.status_code, 400)

        ok_pwd = self.client.post(
            "/api/v1/auth/change-password",
            json={"old_password": "OldPassword123", "new_password": "NewPassword456!"},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(ok_pwd.status_code, 200)

        # Đăng nhập lại bằng mật khẩu mới
        relogin = self.client.post("/api/v1/auth/login", json={
            "email": test_email, "password": "NewPassword456!"
        })
        self.assertEqual(relogin.status_code, 200)

        # 6. Kiểm tra luồng Google OAuth 2.0 (mock verify_google_id_token)
        mock_google_payload = {
            "google_id": "google-sub-123456789",
            "email": "google_demo@gmail.com",
            "full_name": "Google Demo User",
            "avatar_url": "https://lh3.googleusercontent.com/a/demo-avatar",
        }
        with patch("auth.verify_google_id_token", return_value=mock_google_payload):
            g_res = self.client.post("/api/v1/auth/google", json={
                "credential": "fake.valid.google.jwt.token",
                "role": "teacher",
            })
            self.assertEqual(g_res.status_code, 200)
            g_user = g_res.json()["user"]
            self.assertEqual(g_user["email"], "google_demo@gmail.com")
            self.assertEqual(g_user["auth_provider"], "google")
            self.assertTrue(g_user["is_google_linked"])
            self.assertEqual(g_user["avatar_url"], mock_google_payload["avatar_url"])

        # Dọn 2 tài khoản test khỏi SQL Server
        with db_mod.get_connection() as conn:
            conn.cursor().execute("DELETE FROM users WHERE email IN (?, ?);", (test_email, "google_demo@gmail.com"))
            conn.commit()

    def test_15_extension_jwt_sync_and_demo_zip_e2e(self):
        """Kiểm tra toàn luồng Giai đoạn 3 (Extension JWT sync thật/giả) và chấm thử bộ Demo ZIP thật của Giai đoạn 2."""
        t_token = self.client.post("/api/v1/auth/login", json={
            "email": "giangvien@veritext.ai", "password": "123456"
        }).json()["access_token"]

        # 1. Chrome Extension gửi token hợp lệ -> saved_by_user = True
        ext_ok = self.client.post(
            "/api/v1/detect/sentences",
            json={
                "text": "Trí tuệ nhân tạo đang hỗ trợ rất lớn trong phân tích văn bản học thuật hiện đại.",
                "threshold": 0.5,
                "title": "[LMS/Web] Bài nộp Moodle K16",
            },
            headers={"Authorization": f"Bearer {t_token}"},
        )
        self.assertEqual(ext_ok.status_code, 200)
        ok_data = ext_ok.json()
        self.assertTrue(ok_data["saved_by_user"])
        self.assertIn("Hoàng Văn Thầy", ok_data["saved_user_name"])
        self.client.delete(f"/api/v1/history/{ok_data['scan_id']}", headers={"Authorization": f"Bearer {t_token}"})

        # 2. Chrome Extension gửi token giả/hết hạn -> fallback khách an toàn & saved_by_user = False
        ext_expired = self.client.post(
            "/api/v1/detect/sentences",
            json={
                "text": "This paragraph verifies expired token fallback behavior for Chrome Extension.",
                "threshold": 0.5,
            },
            headers={"Authorization": "Bearer invalid.expired.jwt.token"},
        )
        self.assertEqual(ext_expired.status_code, 200)
        exp_data = ext_expired.json()
        self.assertFalse(exp_data["saved_by_user"])
        self.client.delete(f"/api/v1/history/{exp_data['scan_id']}")

        # 3. Chấm toàn bộ file Demo ZIP thật (docs/demo_samples/Lop_K16_TieuLuan_Demo.zip)
        demo_zip_path = os.path.abspath(os.path.join(BASE_DIR, "..", "docs", "demo_samples", "Lop_K16_TieuLuan_Demo.zip"))
        if os.path.exists(demo_zip_path):
            with open(demo_zip_path, "rb") as f:
                z_res = self.client.post(
                    "/api/v1/classroom/batch-scan",
                    files=[("files", ("Lop_K16_TieuLuan_Demo.zip", f, "application/zip"))],
                    data={"batch_name": "E2E Demo ZIP Audit", "strictness_preset": "balanced"},
                    headers={"Authorization": f"Bearer {t_token}"},
                )
            self.assertEqual(z_res.status_code, 200)
            batch = z_res.json()["batch"]
            self.assertGreaterEqual(batch["total_files"], 5)
            self.assertGreaterEqual(batch["bypass_count"], 1)
            # Dọn sạch sau test
            for sub in batch["submissions"]:
                self.client.delete(f"/api/v1/history/{sub['scan_id']}", headers={"Authorization": f"Bearer {t_token}"})
            self.client.delete(f"/api/v1/classroom/batches/{batch['id']}", headers={"Authorization": f"Bearer {t_token}"})

    def test_16_sliding_window_long_essay_qr_verify_and_history_filters(self):
        """Kiểm tra Sliding-Window bài dài (>256 tokens), chống chấm oan tiêu đề ngắn, xác thực HMAC QR & bộ lọc lịch sử."""
        long_hybrid_vi = (
            "1. Mở đầu\n"
            "Hôm qua tôi đi bộ dọc bờ hồ lúc trời vừa hửng nắng sớm, nghe tiếng rao bánh khúc quen thuộc ở góc phố nhỏ "
            "và nhớ lại những buổi chiều tụ tập ăn chè cùng đám bạn thời sinh viên. Hồi đó cả nhóm hay rủ nhau đạp xe "
            "vòng quanh khu tập thể cũ, vừa ăn xôi xéo vừa bàn chuyện thi cử, chẳng mấy khi nghĩ ngợi sâu xa về công nghệ. "
            "Bà cụ bán nước chè đầu ngõ vẫn nhớ tên từng đứa, cứ thấy mặt là mỉm cười rót ngay cốc trà đá mát lạnh. "
            "Những mảnh ký ức bình dị ấy khiến bài ghi chép tay hôm nay mang đậm cảm xúc đời thường rất tự nhiên. "
            "Khi mùa thu ghé qua, lá bàng rơi rụng đầy khoảng sân trường cũ, đám học trò lại í ới gọi nhau chụp vài tấm ảnh kỷ niệm. "
            "Tôi vẫn giữ thói quen ghi chép mọi chuyện vặt vãnh vào cuốn sổ tay bìa sờn cũ để mỗi lúc rảnh rỗi lật ra đọc lại. "
            "Cảm giác viết từng dòng chữ bằng mực xanh lên trang giấy thơm mùi gỗ luôn đem tới sự thư thái kỳ lạ mà không màn hình nào thay thế nổi.\n\n"
            "2. Phân tích AI\n"
            "Trí tuệ nhân tạo ngày nay đang đóng vai trò then chốt trong sự phát triển vượt bậc của nền công nghệ số toàn cầu. "
            "Không thể phủ nhận rằng các mô hình ngôn ngữ lớn hiện đại có khả năng tiếp nhận, tổng hợp và tái cấu trúc "
            "lượng thông tin khổng lồ với độ mạch lạc và tính nhất quán cú pháp đáng kinh ngạc. "
            "Bên cạnh đó, sự phổ biến rộng rãi của các công cụ sinh văn bản tự động cũng đặt ra hàng loạt thách thức cấp bách "
            "đối với tính trung thực học thuật, quyền sở hữu trí tuệ và công tác kiểm định chất lượng đào tạo tại các trường đại học. "
            "Do đó, việc nghiên cứu và triển khai các giải pháp phát hiện nội dung do máy sinh dựa trên kiến trúc Transformer kết hợp "
            "là một hướng đi chiến lược vô cùng cần thiết trong kỷ nguyên chuyển đổi số hiện đại. "
            "Tóm lại, việc kết hợp phân tích thống kê ngôn ngữ học với các mô hình học sâu sẽ đảm bảo tính minh bạch và công bằng."
        )

        res = self.client.post("/api/v1/detect/web", json={
            "text": long_hybrid_vi,
            "threshold": 0.5,
            "title": "KiemThu_SlidingWindow_QR_Verify",
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()

        # 1. Sliding-Window phải chia >= 2 đoạn bao phủ 100% văn bản > 256 tokens
        stats = data["overall"]["text_stats"]
        self.assertGreaterEqual(stats["chunk_count"], 2)
        self.assertIn("Sliding-Window", stats["context_coverage"])

        # 2. Tiêu đề ngắn (<= 4 từ: "1. Mở đầu") phải được bảo vệ ở trạng thái xanh trung tính
        first_sent = data["sentences"][0]
        self.assertLessEqual(first_sent["word_count"], 4)
        self.assertEqual(first_sent["highlight_color"], "green")
        self.assertIn("Tiêu đề", first_sent["assessment"])

        # 3. Audit Code HMAC-SHA256 hợp lệ & PDF QR export thành công
        scan_id = data["scan_id"]
        audit_code = data.get("audit_code", "")
        self.assertRegex(audit_code, r"^VT-\d{6}-[A-F0-9]{8}$")

        pdf_res = self.client.get(f"/api/v1/report/pdf/{scan_id}")
        self.assertEqual(pdf_res.status_code, 200)
        self.assertEqual(pdf_res.headers.get("content-type"), "application/pdf")
        self.assertGreater(len(pdf_res.content), 2000)

        # 4. Cổng xác thực chống giả mạo GET /api/v1/verify/{audit_code}
        ok_verify = self.client.get(f"/api/v1/verify/{audit_code}").json()
        self.assertTrue(ok_verify["valid"])
        self.assertEqual(ok_verify["certificate"]["id"], scan_id)

        last_ch = "0" if audit_code[-1] != "0" else "1"
        tampered_code = audit_code[:-1] + last_ch
        bad_verify = self.client.get(f"/api/v1/verify/{tampered_code}").json()
        self.assertFalse(bad_verify["valid"])

        # 5. Bộ lọc lịch sử đa tiêu chí (language, verdict, q)
        verdict = data["overall"]["label"]
        flt_res = self.client.get(
            "/api/v1/history",
            params={"language": "vi", "verdict": verdict, "q": "KiemThu_SlidingWindow_QR_Verify"},
        )
        self.assertEqual(flt_res.status_code, 200)
        matched_ids = [it["id"] for it in flt_res.json().get("history", [])]
        self.assertIn(scan_id, matched_ids)

        # Dọn bản ghi test khỏi SQL Server
        self.client.delete(f"/api/v1/history/{scan_id}")


if __name__ == "__main__":
    unittest.main()
