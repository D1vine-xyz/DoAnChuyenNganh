# Frontend - VeriText AI PRO

Thư mục này chứa toàn bộ mã nguồn giao diện người dùng (Client-side) của nền tảng **VeriText AI PRO**:

## 1. `web/` - Web Application Dashboard (SPA)
- **Công nghệ:** HTML5, Modern CSS (Minimalist Paper & Graphite Dark/Light Mode với View Transitions API, CSS Variables), Vanilla JavaScript (ES6+), Google Identity Services (OAuth 2.0).
- **Tính năng cốt lõi:**
  - **Thẩm định Văn bản Song ngữ & Đa định dạng:** Nhập trực tiếp hoặc kéo thả tệp `.txt` (hỗ trợ UTF-8 BOM), `.docx` (bóc tách cả nội dung trong bảng), `.doc`, `.pdf`; chọn 3 chế độ ngưỡng học thuật (`Khóa luận 30%`, `Tiểu luận 50%`, `Tham khảo 70%`) và Bộ lọc học thuật thông minh (tự động bỏ trích dẫn `"..."` $\ge 8$ từ và mục *Tài liệu tham khảo*).
  - **Pháp y Văn bản (Anti-Bypass) & Explainable AI (XAI):** Cảnh báo ký tự tàng hình Zero-Width & đồng hình Cyrillic/Hy Lạp giả Latin; hiển thị chỉ số Burstiness, Perplexity Proxy, MATTR, cụm từ khuôn mẫu LLM, phân loại 4 mức độ can thiệp và mạch văn gốc tô màu kiểu Turnitin.
  - **Chấm Bài Lớp Học Hàng Loạt (`RBAC Teacher Only`):** Kéo thả gói `.ZIP` tải từ Moodle/Google Classroom hoặc nhiều tệp cùng lúc (tối đa 100 tệp/đợt), tự động xếp hạng rủi ro toàn lớp và xuất bảng điểm `.XLSX` (2 sheet tô màu) hoặc `.CSV` (UTF-8 BOM).
  - **Biên Bản PDF + Chữ Ký Số HMAC-SHA256 & QR Code:** Xuất chứng nhận PDF A4 kèm QR Code và Cổng tra cứu xác thực chống giả mạo (`VT-XXXXXX-XXXXXXXX`).
  - **Quản trị Tài khoản & Lịch sử SQL Server:** Đăng ký/Đăng nhập JWT + Bcrypt hoặc 1-Click Google Sign-In, đổi vai trò/mật khẩu và lọc nhật ký đa tiêu chí (`Ngôn ngữ`, `Kết luận`, `Chỉ bài dính Bypass`, tìm kiếm từ khóa).
- **Cách sử dụng:** Được máy chủ FastAPI tự động phục vụ tại `http://localhost:8000/`.

## 2. `extension/` - Chrome Extension (Manifest v3 — v3.0.0 PRO)
- **Công nghệ:** Chrome Extensions Manifest V3 (Service Worker, Content Script, Context Menus, `chrome.storage.local`).
- **Tính năng cốt lõi:**
  - **Đồng bộ Tài khoản JWT với SQL Server:** Đăng nhập ngay trong Popup để lưu mọi lượt quét trên trình duyệt thẳng về nhật ký tài khoản trên `VeriTextAI`.
  - **Quét Toàn Trang LMS / Classroom / Blog 1-Click:** Bóc tách thông minh nội dung bài viết chính trên trang hiện tại kèm Bộ lọc học thuật, cảnh báo Anti-Bypass, huy hiệu 4 mức độ và dải chỉ số XAI rút gọn.
  - **Quét Bôi Đen Tức Thì (`In-Page Selection`):** Hiển thị nút nổi `🤖 Quét AI` và Context Menu chuột phải với Modal nổi trực quan ngay trên trang web đang đọc.
- **Cách cài đặt:**
  1. Khởi chạy Backend tại `http://localhost:8000` (`start_server.bat` hoặc `backend/run_server.py`).
  2. Mở Chrome, truy cập `chrome://extensions/` và bật **Developer mode**.
  3. Bấm **Load unpacked** và chọn thư mục `c:\DoAn\frontend\extension`.
