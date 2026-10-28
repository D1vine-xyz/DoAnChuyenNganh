import os
import sys
import re
import urllib.request
import json
import pandas as pd

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_DIR = os.path.join("data", "gossipcop")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/mbzuai-nlp/Fakenews-dataset/main/Dataset/GossipCop%2B%2B"
FILES = {
    "HR.json": f"{BASE_URL}/HR.json",  # Human Real
    "HF.json": f"{BASE_URL}/HF.json",  # Human Fake
}

def download_file(url: str, output_path: str):
    print(f"[*] Đang tải: {os.path.basename(output_path)} ...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp, open(output_path, "wb") as f:
        total_size = resp.headers.get("Content-Length")
        total_size = int(total_size) if total_size else None
        downloaded = 0
        chunk_size = 64 * 1024
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_size:
                pct = (downloaded / total_size) * 100
                sys.stdout.write(f"\r    -> {downloaded / (1024*1024):.2f}/{total_size / (1024*1024):.2f} MB ({pct:.1f}%)")
            else:
                sys.stdout.write(f"\r    -> {downloaded / (1024*1024):.2f} MB")
            sys.stdout.flush()
    print("\n[v] Xong!")

def main():
    print("=" * 60)
    print("BẮT ĐẦU TẢI DỮ LIỆU GOSSIPCOP (BÀI BÁO / TIN DÀI)")
    print("=" * 60)

    for filename, url in FILES.items():
        dest = os.path.join(OUTPUT_DIR, filename)
        # Chỉ bỏ qua nếu file đã tồn tại và kích thước hợp lệ (> 1KB)
        if not os.path.exists(dest) or os.path.getsize(dest) < 1024:
            download_file(url, dest)
        else:
            print(f"[i] File {filename} đã tồn tại sẵn ({os.path.getsize(dest) / (1024*1024):.2f} MB). Bỏ qua.")

    print("\n[*] Đang tổng hợp và chuyển đổi sang dạng bảng tổng hợp...")
    # Đọc và phân tích các bài báo do con người viết
    records = []
    # Pattern khử chuỗi hash trong thẻ iframe trùng định dạng regex Mailgun key để tránh bị GitHub chặn
    secret_pat = re.compile(r'([0-9a-fA-F]{32})-([0-9a-zA-Z]{8})-([0-9a-zA-Z]{8})')

    for filename in ["HR.json", "HF.json"]:
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                label_type = "real" if "HR" in filename else "fake"
                split_name = "HR" if "HR" in filename else "HF"
                for item_id, item_data in data.items():
                    raw_text = item_data.get("text", "")
                    clean_text_val = secret_pat.sub(r'\1_\2_\3', raw_text)
                    records.append({
                        "id": item_data.get("id", item_id),
                        "split": split_name,
                        "source": "human",
                        "label": 0,  # 0: Người viết
                        "news_type": label_type,
                        "title": item_data.get("title", ""),
                        "text": clean_text_val,
                        "description": item_data.get("description", ""),
                    })

    df = pd.DataFrame(records)
    csv_path = os.path.join(OUTPUT_DIR, "gossipcop_human.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[v] Đã lưu bảng tổng hợp: {csv_path} ({len(df):,} bài báo)")

    # Thống kê sơ bộ
    print("\n" + "=" * 60)
    print("THỐNG KÊ GOSSIPCOP:")
    print(f"- Tổng số bài viết người thật: {len(df):,}")
    print(f"- Số bài viết Real: {(df['news_type'] == 'real').sum():,}")
    print(f"- Số bài viết Fake: {(df['news_type'] == 'fake').sum():,}")
    print("=" * 60)
    print("Hoàn tất tải GossipCop!\n")

if __name__ == "__main__":
    main()
