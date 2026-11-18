import os
import sys
import pandas as pd
from sklearn.utils import shuffle

# Đảm bảo in tiếng Việt / emoji trên Windows console không lỗi
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def clean_text_simple(text):
    if text is None or pd.isna(text):
        return ""
    return str(text).strip()

def main():
    print("=" * 70)
    print("  NGÀY 5: GẮN NHÃN PHÂN LOẠI NHỊ PHÂN & HỢP NHẤT DỮ LIỆU HUẤN LUYỆN")
    print("=" * 70)

    os.makedirs("data", exist_ok=True)
    records = []

    # -------------------------------------------------------------
    # NGUỒN 1: DỮ LIỆU TIN DÀI GOSSIPCOP (BÀI BÁO)
    # Đã có sẵn 10,918 bài người thật và 8,170 bài do máy (LLM) sinh ra
    # -------------------------------------------------------------
    gossipcop_file = os.path.join("data", "gossipcop_cleaned.csv")
    if os.path.exists(gossipcop_file):
        print(f"[*] Đang nạp dữ liệu tin dài từ: {gossipcop_file}...")
        df_g = pd.read_csv(gossipcop_file)

        # Lọc các bài người viết (Human - label 0)
        human_gossip = df_g[df_g["source"] == "human"].copy()
        for _, r in human_gossip.iterrows():
            txt = clean_text_simple(r.get("text", ""))
            if len(txt.split()) >= 15:
                records.append({
                    "text": txt,
                    "label": 0,
                    "domain": "gossipcop",
                    "text_type": "long_news",
                    "source": "human",
                    "word_count": len(txt.split())
                })

        # Lọc các bài do AI sinh (Machine - label 1)
        machine_gossip = df_g[df_g["source"] == "machine"].copy()
        for _, r in machine_gossip.iterrows():
            txt = clean_text_simple(r.get("text", ""))
            if len(txt.split()) >= 15:
                records.append({
                    "text": txt,
                    "label": 1,
                    "domain": "gossipcop",
                    "text_type": "long_news",
                    "source": "machine",
                    "word_count": len(txt.split())
                })
        print(f"    -> Đã thêm: {len(human_gossip):,} bài Người + {len(machine_gossip):,} bài AI từ GossipCop")

    # -------------------------------------------------------------
    # NGUỒN 2: DỮ LIỆU TIN DÀI GOSSIPCOP DO GEMINI PARAPHRASE (NGÀY 4)
    # -------------------------------------------------------------
    gp_file = os.path.join("data", "gossipcop_paired_sample.csv")
    if os.path.exists(gp_file):
        df_gp = pd.read_csv(gp_file)
        added_gp = 0
        for _, r in df_gp.iterrows():
            ai_txt = clean_text_simple(r.get("ai_text", ""))
            if len(ai_txt.split()) >= 15:
                records.append({
                    "text": ai_txt,
                    "label": 1,
                    "domain": "gossipcop",
                    "text_type": "long_news",
                    "source": "gemini_paraphrase",
                    "word_count": len(ai_txt.split())
                })
                added_gp += 1
        print(f"    -> Đã thêm: {added_gp} bài báo dài do Gemini sinh từ Ngày 4")

    # -------------------------------------------------------------
    # NGUỒN 3: DỮ LIỆU TIN NGẮN LIAR DO GEMINI PARAPHRASE (NGÀY 3)
    # -------------------------------------------------------------
    lp_file = os.path.join("data", "liar_train_paired_sample.csv")
    if os.path.exists(lp_file):
        df_lp = pd.read_csv(lp_file)
        added_lp_human = 0
        added_lp_ai = 0
        for _, r in df_lp.iterrows():
            h_txt = clean_text_simple(r.get("human_text", ""))
            a_txt = clean_text_simple(r.get("ai_text", ""))
            if h_txt and len(h_txt.split()) >= 4:
                records.append({
                    "text": h_txt,
                    "label": 0,
                    "domain": "liar",
                    "text_type": "short_statement",
                    "source": "human",
                    "word_count": len(h_txt.split())
                })
                added_lp_human += 1
            if a_txt and len(a_txt.split()) >= 4:
                records.append({
                    "text": a_txt,
                    "label": 1,
                    "domain": "liar",
                    "text_type": "short_statement",
                    "source": "gemini_paraphrase",
                    "word_count": len(a_txt.split())
                })
                added_lp_ai += 1
        print(f"    -> Đã thêm: {added_lp_human} câu Người + {added_lp_ai} câu AI từ tập mẫu LIAR Ngày 3")

    # -------------------------------------------------------------
    # HỢP NHẤT, LỌC TRÙNG LẶP & TRỘN NGẪU NHIÊN (SHUFFLE)
    # -------------------------------------------------------------
    print("\n[*] Đang tổng hợp, khử trùng lặp và trộn ngẫu nhiên dữ liệu...")
    df_merged = pd.DataFrame(records)

    initial_count = len(df_merged)
    df_merged = df_merged.drop_duplicates(subset=["text"]).reset_index(drop=True)
    dedup_count = len(df_merged)

    # Trộn ngẫu nhiên với random_state cố định để đảm bảo tính tái lập (reproducibility)
    df_merged = shuffle(df_merged, random_state=42).reset_index(drop=True)

    # Lưu thành file huấn luyện cuối cùng
    output_file = os.path.join("data", "final_train_dataset.csv")
    df_merged.to_csv(output_file, index=False, encoding="utf-8")

    human_count = (df_merged["label"] == 0).sum()
    ai_count = (df_merged["label"] == 1).sum()

    print("=" * 70)
    print("THỐNG KÊ TẬP DỮ LIỆU HUẤN LUYỆN CUỐI CÙNG (final_train_dataset.csv):")
    print(f"- Tổng số mẫu ban đầu: {initial_count:,}")
    print(f"- Tổng số mẫu sau lọc trùng: {dedup_count:,}")
    print(f"- Phân bố nhãn phân loại nhị phân:")
    print(f"  + Nhãn 0 (Người thật viết - Human): {human_count:,} mẫu ({human_count / dedup_count * 100:.1f}%)")
    print(f"  + Nhãn 1 (Mô hình AI sinh - Machine): {ai_count:,} mẫu ({ai_count / dedup_count * 100:.1f}%)")
    print(f"- Phân bố theo thể loại văn bản:")
    for t_type in df_merged["text_type"].unique():
        sub_cnt = (df_merged["text_type"] == t_type).sum()
        print(f"  + {t_type}: {sub_cnt:,} mẫu")
    print(f"- Độ dài từ trung bình:")
    print(f"  + Văn bản con người: {df_merged[df_merged['label'] == 0]['word_count'].mean():.1f} từ")
    print(f"  + Văn bản do AI tạo: {df_merged[df_merged['label'] == 1]['word_count'].mean():.1f} từ")
    print("=" * 70)
    print(f"🎉 Hoàn tất Ngày 5! File dữ liệu đã được lưu tại: {output_file}\n")

if __name__ == "__main__":
    main()

