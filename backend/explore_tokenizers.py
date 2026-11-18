import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# Tắt cảnh báo không cần thiết
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Đảm bảo in tiếng Việt / ký tự đặc biệt trên console Windows không lỗi mã hóa
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from transformers import AutoTokenizer

MODELS = {
    "BERT": "bert-base-uncased",
    "RoBERTa": "roberta-base",
    "Longformer": "allenai/longformer-base-4096",
}

def compute_percentiles(lengths):
    arr = np.array(lengths)
    return {
        "count": int(len(arr)),
        "min": int(np.min(arr)),
        "mean": float(round(np.mean(arr), 2)),
        "median": float(round(np.median(arr), 2)),
        "p75": float(round(np.percentile(arr, 75), 2)),
        "p90": float(round(np.percentile(arr, 90), 2)),
        "p95": float(round(np.percentile(arr, 95), 2)),
        "p99": float(round(np.percentile(arr, 99), 2)),
        "max": int(np.max(arr)),
        "pct_le_128": float(round(np.mean(arr <= 128) * 100, 2)),
        "pct_le_256": float(round(np.mean(arr <= 256) * 100, 2)),
        "pct_le_512": float(round(np.mean(arr <= 512) * 100, 2)),
        "pct_le_1024": float(round(np.mean(arr <= 1024) * 100, 2)),
        "pct_le_2048": float(round(np.mean(arr <= 2048) * 100, 2)),
        "pct_le_4096": float(round(np.mean(arr <= 4096) * 100, 2)),
    }

def print_table_row(name, stats):
    print(f"{name:<12} | {stats['count']:>6,} | {stats['mean']:>7.1f} | {stats['median']:>6.1f} | {stats['p75']:>6.1f} | {stats['p90']:>6.1f} | {stats['p95']:>6.1f} | {stats['max']:>6,}")

def main():
    parser = argparse.ArgumentParser(description="Khảo sát Tokenization và Phân phối độ dài Token cho BERT, RoBERTa, Longformer (Ngày 6)")
    parser.add_argument("--data", type=str, default="data/train.csv", help="Đường dẫn file dữ liệu cần khảo sát (mặc định: data/train.csv)")
    parser.add_argument("--sample_size", type=int, default=None, help="Số lượng mẫu khảo sát (None = toàn bộ tập)")
    parser.add_argument("--batch_size", type=int, default=512, help="Kích thước batch khi tokenize (mặc định: 512)")
    args = parser.parse_args()

    print("=" * 85)
    print("  NGÀY 6: KHẢO SÁT TOKENIZATION & PHÂN PHỐI ĐỘ DÀI TOKEN (ENSEMBLE BACKBONES)")
    print("=" * 85)

    if not os.path.exists(args.data):
        print(f"[!] Không tìm thấy file dữ liệu: {args.data}")
        print("    Vui lòng chạy 'python split_data.py' trước để tạo file này.")
        sys.exit(1)

    print(f"[*] Đang đọc dữ liệu từ: {args.data} ...")
    df = pd.read_csv(args.data)
    if args.sample_size and args.sample_size < len(df):
        df = df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
        print(f"[*] Đã lấy mẫu ngẫu nhiên {len(df):,} mẫu để khảo sát nhanh.")
    else:
        print(f"[v] Đã nạp toàn bộ {len(df):,} mẫu dữ liệu huấn luyện.")

    texts = df["text"].astype(str).tolist()

    # 1. Nạp và so sánh đặc tính kỹ thuật của 3 bộ Tokenizer
    print("\n" + "-" * 85)
    print("1. ĐẶC TÍNH KỸ THUẬT CỦA 3 BỘ TOKENIZER TRONG ENSEMBLE:")
    print("-" * 85)
    print(f"{'Mô hình':<12} | {'Tên HuggingFace':<28} | {'Kiến trúc Tokenizer':<20} | {'Vocab':<7} | {'Max Len':<7}")
    print("-" * 85)

    tokenizers = {}
    for name, model_id in MODELS.items():
        print(f"[*] Đang tải {name} tokenizer ({model_id})...", end="\r")
        tok = AutoTokenizer.from_pretrained(model_id)
        tokenizers[name] = tok
        algo = "WordPiece" if "bert-base" in model_id else "BPE (Byte-Pair)"
        v_size = f"{tok.vocab_size:,}"
        max_l = str(tok.model_max_length)
        print(f"{name:<12} | {model_id:<28} | {algo:<20} | {v_size:<7} | {max_l:<7}")

    # 2. Khảo sát định tính trên câu ví dụ
    print("\n" + "-" * 85)
    print("2. ĐỐI SÁNH ĐỊNH TÍNH TRÊN MẪU VĂN BẢN THỰC TẾ:")
    print("-" * 85)
    sample_text = "The AI-generated article highlights LLM capabilities in natural language understanding and synthesis."
    print(f"Câu mẫu: \"{sample_text}\"\n")
    for name, tok in tokenizers.items():
        tokens = tok.tokenize(sample_text)
        print(f"  [{name:<10}] ({len(tokens):>2} tokens): {' '.join(tokens[:15])}...")

    # 3. Tính toán phân phối độ dài token trên toàn bộ tập dữ liệu
    print("\n" + "-" * 85)
    print(f"3. TÍNH TOÁN PHÂN PHỐI ĐỘ DÀI TOKEN ({len(texts):,} VĂN BẢN)...")
    print("-" * 85)

    token_lengths = {}
    for name, tok in tokenizers.items():
        print(f"[*] Đang tokenize bằng {name} Fast Tokenizer...")
        lengths = []
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i : i + args.batch_size]
            encoded = tok(
                batch,
                truncation=False,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=True,
            )["input_ids"]
            lengths.extend([len(x) for x in encoded])
        token_lengths[name] = lengths
        df[f"{name.lower()}_token_len"] = lengths

    # Bảng phân phối tổng thể
    print("\n" + "=" * 85)
    print("BẢNG THỐNG KÊ PHÂN PHỐI ĐỘ DÀI TOKEN (TỔNG THỂ TẬP TRAIN)")
    print("=" * 85)
    print(f"{'Mô hình':<12} | {'Số mẫu':<6} | {'Mean':<7} | {'Median':<6} | {'P75':<6} | {'P90':<6} | {'P95':<6} | {'Max':<6}")
    print("-" * 85)
    summary_stats = {}
    for name in MODELS.keys():
        stats = compute_percentiles(token_lengths[name])
        summary_stats[name] = stats
        print_table_row(name, stats)
    print("=" * 85)

    # 4. Phân tích tác động cắt ngắn chuỗi (Truncation Impact Analysis)
    print("\n" + "=" * 85)
    print("TỶ LỆ VĂN BẢN ĐƯỢC GIỮ TRỌN VẸN (KHÔNG BỊ CẮT TRUNCATE) THEO NGƯỠNG ĐỘ DÀI:")
    print("=" * 85)
    print(f"{'Mô hình':<12} | {'<= 128 tok':<11} | {'<= 256 tok':<11} | {'<= 512 tok':<11} | {'<= 1024 tok':<11} | {'<= 2048 tok':<11} | {'<= 4096 tok':<11}")
    print("-" * 85)
    for name, stats in summary_stats.items():
        print(f"{name:<12} | {stats['pct_le_128']:>9.1f}% | {stats['pct_le_256']:>9.1f}% | {stats['pct_le_512']:>9.1f}% | {stats['pct_le_1024']:>10.1f}% | {stats['pct_le_2048']:>10.1f}% | {stats['pct_le_4096']:>10.1f}%")
    print("=" * 85)

    # 5. Phân tích phân vị theo Nguồn (Người vs AI)
    print("\n" + "-" * 85)
    print("5. SO SÁNH ĐỘ DÀI TOKEN: NGƯỜI THẬT (LABEL 0) VS AI SINH (LABEL 1)")
    print("-" * 85)
    print(f"{'Nhóm văn bản':<22} | {'Mô hình':<10} | {'Mean':<7} | {'Median':<6} | {'P90':<6} | {'P95':<6} | {'Max':<6}")
    print("-" * 85)
    group_stats = {}
    for label_val, group_name in [(0, "Người thật (Human)"), (1, "Máy sinh (AI/LLM)")]:
        sub_df = df[df["label"] == label_val]
        group_stats[group_name] = {}
        for name in MODELS.keys():
            lens = sub_df[f"{name.lower()}_token_len"].tolist()
            s = compute_percentiles(lens)
            group_stats[group_name][name] = s
            print(f"{group_name:<22} | {name:<10} | {s['mean']:>7.1f} | {s['median']:>6.1f} | {s['p90']:>6.1f} | {s['p95']:>6.1f} | {s['max']:>6,}")
    print("-" * 85)

    # 6. Đưa ra khuyến nghị thiết lập max_length cho Tuần 2
    print("\n" + "=" * 85)
    print("💡 KHUYẾN NGHỊ CẤU HÌNH MAX_LENGTH CHO MÔ HÌNH ENSEMBLE (TUẦN 2):")
    print("=" * 85)
    print(f"1. BERT & RoBERTa (Standard PLMs):")
    print(f"   - Giới hạn cứng của kiến trúc: 512 tokens.")
    print(f"   - Tỷ lệ dữ liệu bao phủ trọn vẹn: ~{summary_stats['BERT']['pct_le_512']:.1f}% (BERT) và ~{summary_stats['RoBERTa']['pct_le_512']:.1f}% (RoBERTa).")
    print(f"   - Khuyến nghị cấu hình: max_length = 512 với padding='max_length' và truncation=True.")
    print(f"2. Longformer (Long Document PLM):")
    print(f"   - Khả năng tối đa của kiến trúc: 4096 tokens.")
    print(f"   - Tỷ lệ bao phủ tại ngưỡng 1024 tokens: ~{summary_stats['Longformer']['pct_le_1024']:.1f}%.")
    print(f"   - Tỷ lệ bao phủ tại ngưỡng 2048 tokens: ~{summary_stats['Longformer']['pct_le_2048']:.1f}%.")
    print(f"   - Khuyến nghị cấu hình: max_length = 1024 hoặc 2048 để cân bằng tối ưu giữa bộ nhớ VRAM GPU và việc nắm bắt toàn diện bài báo dài.")
    print("=" * 85)

    # 7. Lưu file kết quả thống kê
    output_json = os.path.join("data", "token_length_stats.json")
    all_results = {
        "dataset_analyzed": args.data,
        "sample_count": len(df),
        "overall_summary": summary_stats,
        "group_summary": group_stats,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[v] Đã lưu báo cáo thống kê chi tiết vào: {output_json}\n")

if __name__ == "__main__":
    main()

