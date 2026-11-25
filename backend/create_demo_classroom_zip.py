"""
Tạo sẵn gói bài nộp lớp học mẫu (Lop_K16_TieuLuan_Demo.zip) phục vụ Demo Vấn Đáp Hội Đồng
Gồm 6 bài nộp mô phỏng đầy đủ các tình huống thực tế của sinh viên:
1. 22110001_Nguyen_Van_An_TuViet.txt         (Tiếng Việt - Người tự viết)
2. 22110002_Tran_Thi_Binh_ChatGPT.txt        (Tiếng Việt - AI sinh thuần túy + Tài liệu tham khảo)
3. 22110003_Le_Hoang_Cuong_LachLuat.txt      (Tiếng Việt - AI bị cài ký tự ẩn Zero-Width + Cyrillic Homoglyph)
4. 22110004_Pham_Minh_Duc_BaiLai.txt         (Tiếng Việt - Văn bản lai trộn Người + AI)
5. 22110005_Vo_Quoc_Huy_Essay_Human.txt      (Tiếng Anh - Tự viết tự nhiên)
6. 22110006_Dang_Khanh_Linh_Essay_AI.txt     (Tiếng Anh - LLM sinh chuẩn khuôn mẫu)
"""

import os
import zipfile

DEMO_FILES = {
    "22110001_Nguyen_Van_An_TuViet.txt": (
        "Cuối tuần rồi nhóm mình có buổi họp làm đồ án ngay quán cà phê gần cổng trường vì phòng thư viện kín hết chỗ. "
        "Mấy đứa mang laptop ra chia nhau sửa lại mấy lỗi kết nối cơ sở dữ liệu mà hôm trước chạy mãi không lên. "
        "Tới trưa trời đổ mưa rào nên cả nhóm gọi cơm tấm ngồi ăn tại chỗ luôn, vừa ăn vừa vẽ lại sơ đồ lớp cho kịp thứ hai nộp thầy. "
        "Dù hơi mệt nhưng đến chiều lúc chạy thử chức năng đăng nhập thành công thì ai cũng thấy nhẹ nhõm hẳn."
    ),
    "22110002_Tran_Thi_Binh_ChatGPT.txt": (
        "Trí tuệ nhân tạo ngày nay đang đóng vai trò then chốt trong sự phát triển của nền giáo dục đại học toàn cầu. "
        "Không thể phủ nhận rằng các mô hình ngôn ngữ lớn mang lại nhiều lợi ích vượt trội trong việc hỗ trợ sinh viên tra cứu tài liệu và tổng hợp kiến thức. "
        "Bên cạnh đó, sự phổ biến nhanh chóng của các công cụ tạo sinh văn bản cũng đặt ra nhiều thách thức nghiêm trọng đối với tính liêm chính học thuật. "
        "Tóm lại, việc xây dựng hệ thống kiểm định văn bản minh bạch và đáng tin cậy là một hướng đi vô cùng cấp thiết trong bối cảnh hiện nay.\n\n"
        "Tài liệu tham khảo:\n"
        "[1] Vaswani et al. (2017). Attention Is All You Need. NeurIPS.\n"
        "[2] Nguyen & Nguyen (2020). PhoBERT: Pre-trained language models for Vietnamese."
    ),
    "22110003_Le_Hoang_Cuong_LachLuat.txt": (
        "Trí\u200b tuệ\u200b nhân\u200b t\u0430o đóng v\u0430i trò then chốt tr\u043eng bối c\u0430nh chuyển đổi số hiện n\u0430y. "
        "Không thể phủ nhận rằng công\u200b nghệ\u200b học sâu m\u0430ng lại nhiều lợi ích vượt trội ch\u043e ngành kỹ thuật phần mềm. "
        "Mặt khác, việc lạm dụng c\u0430c công cụ tự động sinh mã nguồn và tiểu luận đòi hỏi những giải ph\u0430p giám định chặt chẽ. "
        "Nhìn chung, sự kết hợp giữa mô hình học máy và quy trình đánh giá của giảng viên là giải pháp tối ưu."
    ),
    "22110004_Pham_Minh_Duc_BaiLai.txt": (
        "Hôm thứ tư tuần trước khi nghe thầy phổ biến đề tài tiểu luận, mình có hỏi thử mấy bạn cùng lớp xem nên chọn hướng nào cho dễ làm. "
        "Sau một hồi bàn bạc ở hành lang tòa B, mình quyết định tìm hiểu về việc sinh viên sử dụng công cụ AI khi viết báo cáo môn học. "
        "Tuy nhiên, không thể phủ nhận rằng các mô hình ngôn ngữ lớn ngày nay sở hữu khả năng tổng hợp thông tin vượt trội và tạo ra các đoạn văn mạch lạc một cách ấn tượng. "
        "Hơn nữa, việc triển khai các khung đánh giá kết hợp đa mô hình tiền huấn luyện đóng vai trò then chốt trong việc bảo đảm tính xác thực và liêm chính học thuật."
    ),
    "22110005_Vo_Quoc_Huy_Essay_Human.txt": (
        "Last Friday evening, my cousin dropped by our apartment with a broken old laptop he found in his closet. "
        "We spent almost two hours hunting for a tiny screwdriver just to pop the back cover open and check the battery cable. "
        "Turns out the ribbon connector had simply slipped loose after he dropped his backpack on the bus last winter. "
        "Once we clicked it back in and saw the login screen light up, we celebrated with instant noodles on the balcony."
    ),
    "22110006_Dang_Khanh_Linh_Essay_AI.txt": (
        "Furthermore, it is crucial to note that artificial intelligence plays a pivotal role in reshaping modern software engineering practices. "
        "These multifaceted neural architectures leverage deep contextual representations to generate coherent and syntactically refined discourse at scale. "
        "Consequently, safeguarding academic integrity in today's digital age requires resilient multi-model verification frameworks. "
        "In conclusion, fostering transparent explainable detection tools is of paramount importance for higher education institutions."
    ),
}


def build_demo_zip():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "demo_samples"))
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, "Lop_K16_TieuLuan_Demo.zip")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, content in DEMO_FILES.items():
            zf.writestr(fname, content.encode("utf-8"))

    print(f"[v] Đã tạo gói bài nộp mẫu demo tại: {zip_path} ({len(DEMO_FILES)} tệp)")


if __name__ == "__main__":
    build_demo_zip()
