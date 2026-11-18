# Frontend - VeriText AI

Thư mục này chứa toàn bộ mã nguồn giao diện người dùng (Client-side) của dự án **VeriText AI**:

## 1. `web/` - Web Application Dashboard
- **Công nghệ:** HTML5, Modern CSS (Glassmorphism, Dark/Light Mode, CSS Variables), Vanilla JavaScript (ES6+), Chart.js, Lucide Icons.
- **Tính năng:**
  - Giao diện thẩm định văn bản AI trực tiếp & Tải tệp tài liệu (`.txt`, `.docx`, `.doc`, `.pdf`).
  - Phân tích trực quan từng câu (AI Hotspot sentence highlighting).
  - Tải báo cáo PDF chuyên nghiệp một click.
  - Quản lý nhật ký thẩm định & thống kê KPI.
  - Chế độ Dark Mode siêu mượt với View Transitions API.
- **Cách sử dụng:** Được máy chủ FastAPI backend tự động phục vụ tại `http://localhost:8000/`.

## 2. `extension/` - Chrome Extension
- **Công nghệ:** Chrome Extensions Manifest V3 (Service Worker, Content Script, Context Menus).
- **Tính năng:**
  - Quét văn bản bôi đen trên bất kỳ trang web nào qua menu chuột phải.
  - Cửa sổ Popup nhanh kiểm tra tính xác thực của nội dung trực tuyến.
  - Gửi yêu cầu phân tích trực tiếp đến backend API `http://localhost:8000/api/v1/detect`.
- **Cách cài đặt:**
  1. Mở trình duyệt Chrome, truy cập `chrome://extensions/`.
  2. Bật công tắc **Developer mode** ở góc trên bên phải.
  3. Bấm **Load unpacked** và chọn thư mục `frontend/extension`.
