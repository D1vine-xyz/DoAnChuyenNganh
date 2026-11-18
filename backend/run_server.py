"""
Script Khởi chạy Máy chủ và Tự động Bật Trình duyệt khi Mô hình Đã Sẵn Sàng
Đồ án Chuyên ngành Kỹ thuật Phần mềm: Hệ thống Thẩm định Văn bản AI (Ensemble PLMs)
"""

import os
import sys
import time
import threading
import webbrowser
import urllib.request
import uvicorn

# Đảm bảo in tiếng Việt UTF-8 chuẩn trên Windows Console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Đảm bảo thư mục chứa script luôn có trong sys.path và làm working directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
os.chdir(CURRENT_DIR)


def wait_and_open_browser(url: str = "http://localhost:8000", max_wait_sec: int = 60):
    """
    Tiến trình ngầm liên tục kiểm tra endpoint /health.
    Chỉ khi nào 3 mô hình PLMs đã nạp xong vào RAM và server phản hồi 200 OK,
    trình duyệt mới được tự động mở lên, tránh lỗi 'This site can't be reached'.
    """
    health_url = f"{url}/health"
    start_time = time.time()
    opened = False

    while time.time() - start_time < max_wait_sec:
        time.sleep(1.0)
        try:
            with urllib.request.urlopen(health_url, timeout=1.5) as response:
                if response.status == 200:
                    time.sleep(0.5)
                    print("\n" + "=" * 70)
                    print(f"  [v] SERVER ĐÃ SẴN SÀNG! ĐANG TỰ ĐỘNG MỞ TRÌNH DUYỆT...")
                    print(f"  🌐 Truy cập Web Dashboard tại: {url}")
                    print(f"  📑 Tài liệu API Swagger tại    : {url}/docs")
                    print("=" * 70 + "\n")
                    webbrowser.open(url)
                    opened = True
                    break
        except Exception:
            # Server đang trong quá trình nạp mô hình, tiếp tục chờ
            pass

    if not opened:
        print(f"\n[!] Lưu ý: Bạn có thể tự mở trình duyệt và truy cập: {url}")


def main():
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    print("\n" + "=" * 70)
    print("  KHỞI ĐỘNG HỆ THỐNG THẨM ĐỊNH VĂN BẢN AI (FASTAPI + ENSEMBLE PLMs)")
    print("======================================================================")
    print("[*] Đang khởi tạo và nạp 3 mô hình BERT, RoBERTa, Longformer vào RAM...")
    print("[*] Quá trình này mất khoảng 10 - 15 giây. Vui lòng chờ...")
    print("======================================================================\n")

    # Kích hoạt thread kiểm tra và mở trình duyệt khi server sẵn sàng
    threading.Thread(target=wait_and_open_browser, args=(f"http://localhost:{port}",), daemon=True).start()

    # Khởi chạy Uvicorn server
    uvicorn.run("api:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
