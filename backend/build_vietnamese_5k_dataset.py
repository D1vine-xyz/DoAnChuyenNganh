"""
Module Xây dựng Bộ Dữ liệu Phát hiện Văn bản AI Tiếng Việt 5.000 Mẫu (Vietnamese AI Benchmark 5K)
Đồ án Chuyên ngành Kỹ thuật Phần mềm (Chủ đề 6: Thẩm định Văn bản LLM - Ensemble PLMs)

Bộ dữ liệu kết hợp đa nguồn (Multi-source & Multi-LLM):
1. Nguồn Gốc (1.000 mẫu): Cặp đối sánh VnExpress (Human) & Gemini 3.5 Flash Paraphrase (AI).
2. Nguồn Mở rộng (4.000 mẫu): Benchmark tiếng Việt đa lĩnh vực (Báo chí, Đời sống, Công nghệ, Xã hội).
Tổng quy mô: Đúng 5.000 mẫu chuẩn chỉnh, cân bằng 50% Human (2.500) và 50% AI (2.500).
Phân chia: 70% Train (3.500) | 15% Val (750) | 15% Test (750).
"""

import os
import sys
import shutil
import re
import unicodedata
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from datasets import load_dataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def clean_text(text: str) -> str:
    """Làm sạch văn bản tiếng Việt Unicode NFC."""
    if not text or not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_5k_dataset():
    print("=" * 70)
    print("  KHỞI TẠO BỘ DỮ LIỆU TIẾNG VIỆT 5.000 MẪU (PHÂN HỆ CHUYÊN NGÀNH)")
    print("=" * 70)

    # 1. NẠP DỮ LIỆU CŨ (1.000 mẫu)
    old_full_path = os.path.join(DATA_DIR, "vietnamese_full_dataset.csv")
    records = []
    seen_texts = set()

    if os.path.exists(old_full_path):
        print(f"[*] Đang nạp dữ liệu gốc từ {old_full_path}...")
        df_old = pd.read_csv(old_full_path)
        for _, row in df_old.iterrows():
            c_text = clean_text(str(row["text"]))
            if c_text and c_text not in seen_texts:
                seen_texts.add(c_text)
                records.append({
                    "text": c_text,
                    "label": int(row["label"]),
                    "source": str(row.get("source", "gemini_vnexpress_paired")),
                    "word_count": len(c_text.split()),
                    "char_count": len(c_text),
                })
        print(f"    -> Đã nạp thành công {len(records)} mẫu gốc từ giai đoạn 1.")
    else:
        print("[!] Không tìm thấy vietnamese_full_dataset.csv, sẽ nạp toàn bộ từ nguồn mở rộng.")

    # Đếm số lượng hiện tại
    human_count = sum(1 for r in records if r["label"] == 0)
    ai_count = sum(1 for r in records if r["label"] == 1)
    print(f"    -> Hiện trạng: Người viết: {human_count} | AI sinh: {ai_count}")

    # 2. BỔ SUNG TỪ BENCHMARK TIẾNG VIỆT ĐA LĨNH VỰC (HUGGING FACE)
    NEEDED_HUMAN = 2500 - human_count
    NEEDED_AI = 2500 - ai_count

    print(f"[*] Cần bổ sung thêm: {NEEDED_HUMAN} mẫu Người viết + {NEEDED_AI} mẫu AI sinh...")
    print("[*] Đang tải benchmark tiếng Việt kngann2201/vietnamese-ai-detection từ Hugging Face...")

    ds = load_dataset("kngann2201/vietnamese-ai-detection", split="train")

    added_human = 0
    added_ai = 0

    # Duyệt lấy dữ liệu với độ dài lý tưởng cho PhoBERT (100 - 380 từ)
    for item in ds:
        lbl = int(item["label"])
        raw_text = clean_text(item.get("text", ""))
        words = raw_text.split()
        w_len = len(words)

        if w_len < 100 or w_len > 400:
            continue
        if raw_text in seen_texts:
            continue

        if lbl == 0 and added_human < NEEDED_HUMAN:
            seen_texts.add(raw_text)
            records.append({
                "text": raw_text,
                "label": 0,
                "source": "hf_vietnamese_human",
                "word_count": w_len,
                "char_count": len(raw_text),
            })
            added_human += 1

        elif lbl == 1 and added_ai < NEEDED_AI:
            seen_texts.add(raw_text)
            records.append({
                "text": raw_text,
                "label": 1,
                "source": "hf_vietnamese_ai",
                "word_count": w_len,
                "char_count": len(raw_text),
            })
            added_ai += 1

        if added_human >= NEEDED_HUMAN and added_ai >= NEEDED_AI:
            break

    print(f"[v] Đã bổ sung thành công:")
    print(f"    - Thêm Người viết : +{added_human} mẫu")
    print(f"    - Thêm AI sinh    : +{added_ai} mẫu")

    # 3. TẠO DATAFRAME TỔNG HỢP & GÁN ID CHUẨN HÓA
    df_all = pd.DataFrame(records)
    # Trộn ngẫu nhiên (Shuffle) với random seed cố định để tái lập kết quả
    df_all = df_all.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_all["id"] = [f"VN_5K_{i+1:04d}" for i in range(len(df_all))]
    df_all["label_name"] = df_all["label"].map({0: "Human", 1: "AI-generated"})

    # Đưa cột id lên đầu
    cols = ["id", "text", "label", "label_name", "source", "word_count", "char_count"]
    df_all = df_all[cols]

    # Sao lưu các file 1K cũ
    for fname in ["vietnamese_full_dataset.csv", "vietnamese_train.csv", "vietnamese_val.csv", "vietnamese_test.csv"]:
        src = os.path.join(DATA_DIR, fname)
        if os.path.exists(src):
            bak = os.path.join(DATA_DIR, fname.replace(".csv", "_1k_backup.csv"))
            if not os.path.exists(bak):
                shutil.copy2(src, bak)

    # 4. CHIA TẬP STRATIFIED SPLIT (70% Train - 15% Val - 15% Test)
    # Phân tầng chuẩn xác đảm bảo tỷ lệ nhãn 50/50 ở cả 3 tập
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_idx, test_idx = next(sss_test.split(df_all, df_all["label"]))

    df_train_val = df_all.iloc[train_val_idx].reset_index(drop=True)
    df_test = df_all.iloc[test_idx].reset_index(drop=True)

    # 15% của 5000 = 750 mẫu val -> test_size = 750 / 4250 ≈ 0.17647
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.17647, random_state=42)
    train_idx, val_idx = next(sss_val.split(df_train_val, df_train_val["label"]))

    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].reset_index(drop=True)

    # Lưu cả bản có đuôi _5k và bản chuẩn thay thế
    save_map = {
        "vietnamese_full_dataset.csv": df_all,
        "vietnamese_full_dataset_5k.csv": df_all,
        "vietnamese_train.csv": df_train,
        "vietnamese_train_5k.csv": df_train,
        "vietnamese_val.csv": df_val,
        "vietnamese_val_5k.csv": df_val,
        "vietnamese_test.csv": df_test,
        "vietnamese_test_5k.csv": df_test,
    }

    for fname, data_df in save_map.items():
        data_df.to_csv(os.path.join(DATA_DIR, fname), index=False, encoding="utf-8")

    # 5. BÁO CÁO KẾT QUẢ THỐNG KÊ
    print("\n" + "=" * 70)
    print("  KẾT QUẢ XÂY DỰNG BỘ DỮ LIỆU TIẾNG VIỆT 5.000 MẪU (CHUẨN CHUYÊN NGÀNH)")
    print("=" * 70)
    print(f"[*] TỔNG CỘNG       : {len(df_all):>5,} mẫu | Người: {(df_all['label']==0).sum():>4} | AI: {(df_all['label']==1).sum():>4}")
    print(f"    - Độ dài từ TB  : {df_all['word_count'].mean():.1f} từ (Min: {df_all['word_count'].min()}, Max: {df_all['word_count'].max()})")
    print(f"    - Số ký tự TB   : {df_all['char_count'].mean():.1f} ký tự")
    print("-" * 70)
    print(f"[*] TẬP TRAIN (70%) : {len(df_train):>5,} mẫu | Người: {(df_train['label']==0).sum():>4} ({(df_train['label']==0).mean()*100:.1f}%) | AI: {(df_train['label']==1).sum():>4} ({(df_train['label']==1).mean()*100:.1f}%)")
    print(f"[*] TẬP VAL   (15%) : {len(df_val):>5,} mẫu | Người: {(df_val['label']==0).sum():>4} ({(df_val['label']==0).mean()*100:.1f}%) | AI: {(df_val['label']==1).sum():>4} ({(df_val['label']==1).mean()*100:.1f}%)")
    print(f"[*] TẬP TEST  (15%) : {len(df_test):>5,} mẫu | Người: {(df_test['label']==0).sum():>4} ({(df_test['label']==0).mean()*100:.1f}%) | AI: {(df_test['label']==1).sum():>4} ({(df_test['label']==1).mean()*100:.1f}%)")
    print("=" * 70)
    print("🎉 BỘ DỮ LIỆU 5.000 MẪU ĐÃ SẴN SÀNG ĐỂ FINE-TUNE PHOBERT TRÊN KAGGLE GPU!\n")

if __name__ == "__main__":
    build_5k_dataset()
