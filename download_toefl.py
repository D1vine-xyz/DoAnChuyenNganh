import os
import sys
import urllib.request
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_DIR = os.path.join("data", "toefl")
os.makedirs(OUTPUT_DIR, exist_ok=True)

GITHUB_API_BASE = "https://api.github.com/repos/rexshijaku/chatgpt-generated-text-detection-corpus/contents/full_texts"
QUESTIONS_RAW_URL = "https://raw.githubusercontent.com/rexshijaku/chatgpt-generated-text-detection-corpus/main/full_texts/questions.txt"

def fetch_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))

def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")

def parse_questions(text: str) -> dict:
    """Trích xuất danh sách câu hỏi đề bài TOEFL từ file questions.txt"""
    questions = {}
    current_id = None
    current_lines = []

    for line in text.splitlines():
        line_stripped = line.strip()
        if line_stripped and line_stripped[0].isdigit():
            parts = line_stripped.split(".", 1)
            if parts[0].isdigit() and len(parts) > 1:
                if current_id is not None:
                    questions[current_id] = " ".join(current_lines).strip()
                current_id = int(parts[0])
                current_lines = [parts[1].strip()]
                continue
        if current_id is not None and line_stripped:
            current_lines.append(line_stripped)

    if current_id is not None:
        questions[current_id] = " ".join(current_lines).strip()

    return questions

def download_file_content(item):
    name, download_url, source = item
    essay_id = int(name.replace(".txt", ""))
    try:
        content = fetch_text(download_url)
        return (essay_id, source, content.strip(), None)
    except Exception as e:
        return (essay_id, source, None, str(e))

def main():
    print("=" * 60)
    print("BẮT ĐẦU TẢI DỮ LIỆU TOEFL (126 HUMAN + 126 CHATGPT ESSAYS)")
    print("=" * 60)

    # 1. Tải đề bài questions.txt
    print("[*] Đang tải danh sách câu hỏi đề bài (questions.txt)...")
    questions_text = fetch_text(QUESTIONS_RAW_URL)
    questions_map = parse_questions(questions_text)
    print(f"[v] Đã phân tích được {len(questions_map)} đề bài câu hỏi.")

    # 2. Lấy danh sách chính xác các file từ GitHub API
    print("[*] Đang lấy danh sách file từ GitHub repository...")
    human_files = fetch_json(f"{GITHUB_API_BASE}/human")
    chatgpt_files = fetch_json(f"{GITHUB_API_BASE}/chatgpt")

    download_tasks = []
    for f in human_files:
        if f["name"].endswith(".txt"):
            download_tasks.append((f["name"], f["download_url"], "human"))
    for f in chatgpt_files:
        if f["name"].endswith(".txt"):
            download_tasks.append((f["name"], f["download_url"], "chatgpt"))

    print(f"[*] Tổng số bài luận cần tải: {len(download_tasks)} ({len(human_files)} Human + {len(chatgpt_files)} ChatGPT)")

    records = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_task = {executor.submit(download_file_content, task): task for task in download_tasks}
        completed = 0
        for future in as_completed(future_to_task):
            essay_id, source, text, err = future.result()
            completed += 1
            if err:
                print(f"\n[!] Lỗi tải {source}/{essay_id}: {err}")
            else:
                records.append({
                    "essay_id": essay_id,
                    "prompt_question": questions_map.get(essay_id, ""),
                    "source": source,
                    "label": 0 if source == "human" else 1,  # 0: Người viết, 1: AI (ChatGPT)
                    "text": text,
                    "word_count": len(text.split()),
                })
            sys.stdout.write(f"\r    -> Đã tải thành công: {completed}/{len(download_tasks)} bài luận")
            sys.stdout.flush()

    print("\n[v] Tải thành công toàn bộ bài luận!")

    # 3. Lưu vào DataFrame
    df = pd.DataFrame(records)
    df = df.sort_values(by=["essay_id", "source"]).reset_index(drop=True)

    csv_path = os.path.join(OUTPUT_DIR, "toefl_essays.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[v] Đã lưu bảng tổng hợp: {csv_path}")

    # 4. Thống kê
    print("\n" + "=" * 60)
    print("THỐNG KÊ TẬP DỮ LIỆU KIỂM THỬ TOEFL:")
    print(f"- Tổng số bài luận: {len(df)}")
    print(f"- Số bài do Người viết (label=0): {(df['label'] == 0).sum()}")
    print(f"- Số bài do ChatGPT viết (label=1): {(df['label'] == 1).sum()}")
    print(f"- Số từ trung bình (Người): {df[df['label'] == 0]['word_count'].mean():.1f} từ")
    print(f"- Số từ trung bình (ChatGPT): {df[df['label'] == 1]['word_count'].mean():.1f} từ")
    print("=" * 60)
    print("Hoàn tất tải TOEFL!\n")

if __name__ == "__main__":
    main()
