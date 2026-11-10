"""
Script Đánh giá Khả năng Tổng quát hóa Ngoại miền (Cross-Domain Generalization) - Ngày 12
Bộ dữ liệu: TOEFL Essay Dataset (data/toefl_dataset.csv - 252 bài luận: 126 Người vs 126 ChatGPT)

Mục tiêu học thuật:
- Đánh giá khả năng suy luận zero-shot trên phong cách viết học thuật chưa từng học khi train
- Đo lường hiện tượng dịch chuyển miền (Domain Shift)
- So sánh tính bền vững (Robustness) của mô hình đơn lẻ vs Ensemble
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

from ensemble import EnsembleDetector

TOEFL_FILE = os.path.join("data", "toefl_dataset.csv")
OUTPUT_DIR = "output"
FIGURES_DIR = os.path.join("reports", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

IN_DOMAIN_TABLE = os.path.join("output", "ensemble_comparison_table.csv")
WEIGHTS_FILE = os.path.join("output", "ensemble_weights.json")
STACKING_MODEL_FILE = os.path.join("output", "stacking_meta_learner.joblib")


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    """Tính toán Accuracy, Precision, Recall, F1, AUC."""
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc_val = roc_auc_score(y_true, y_prob)
    except Exception:
        auc_val = 0.5
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc_val),
    }


def main():
    print("=" * 80)
    print("  ĐÁNH GIÁ TỔNG QUÁT HÓA NGOẠI MIỀN TRÊN TẬP BÀI LUẬN TOEFL (NGÀY 12)")
    print("=" * 80)

    if not os.path.exists(TOEFL_FILE):
        print(f"[!] LỖI: Không tìm thấy tệp {TOEFL_FILE}")
        sys.exit(1)

    toefl_df = pd.read_csv(TOEFL_FILE)
    print(f"[*] Đã nạp tập TOEFL: {len(toefl_df):,} mẫu.")
    print("    - Phân bố nhãn:", dict(toefl_df["label"].value_counts()))

    texts = toefl_df["text"].astype(str).tolist()
    labels = toefl_df["label"].values

    # Khởi tạo Ensemble Detector
    detector = EnsembleDetector(
        bert_path=os.path.join(OUTPUT_DIR, "bert_best.pt"),
        roberta_path=os.path.join(OUTPUT_DIR, "roberta_best.pt"),
        longformer_path=os.path.join(OUTPUT_DIR, "longformer_best.pt"),
        use_fp16=False,  # CPU inference
    )

    # Nạp trọng số tối ưu từ Ngày 11
    with open(WEIGHTS_FILE, "r", encoding="utf-8") as f:
        w_dict = json.load(f)["weights"]
    weights = np.array([w_dict["bert"], w_dict["roberta"], w_dict["longformer"]])
    detector.weights = weights
    print(f"[*] Đã nạp trọng số tối ưu Ensemble: BERT={weights[0]:.4f}, RoBERTa={weights[1]:.4f}, Longformer={weights[2]:.4f}")

    # Nạp mô hình Stacking
    if os.path.exists(STACKING_MODEL_FILE):
        detector.meta_classifier = joblib.load(STACKING_MODEL_FILE)
        print("  [v] Đã nạp mô hình Stacking Meta-Classifier thành công.")

    # Trích xuất dự đoán
    print("\n[*] Đang chạy trích xuất dự đoán trên tập TOEFL (252 bài luận)...")
    start_t = time.time()
    preds = detector.extract_predictions(texts, batch_size=8, show_progress=True)
    elapsed = time.time() - start_t
    print(f"  [v] Trích xuất hoàn tất trong {elapsed:.1f}s ({elapsed / len(texts):.2f}s / bài).")

    # Tính xác suất các chiến lược Ensemble
    p_bert = preds["bert_probs"]
    p_roberta = preds["roberta_probs"]
    p_longformer = preds["longformer_probs"]

    # 1. Simple Soft Voting
    p_simple = (p_bert + p_roberta + p_longformer) / 3.0

    # 2. Weighted Soft Voting
    p_weighted = (weights[0] * p_bert + weights[1] * p_roberta + weights[2] * p_longformer)

    # 3. Stacking
    if detector.meta_classifier:
        X_stack = np.column_stack([p_bert, p_roberta, p_longformer])
        p_stacking = detector.meta_classifier.predict_proba(X_stack)[:, 1]
    else:
        p_stacking = p_simple

    # Lưu file dự đoán chi tiết
    results_df = pd.DataFrame({
        "essay_id": toefl_df["essay_id"],
        "label": labels,
        "bert_prob": p_bert,
        "roberta_prob": p_roberta,
        "longformer_prob": p_longformer,
        "ensemble_simple_prob": p_simple,
        "ensemble_weighted_prob": p_weighted,
        "ensemble_stacking_prob": p_stacking,
    })
    pred_out_file = os.path.join(OUTPUT_DIR, "toefl_predictions.csv")
    results_df.to_csv(pred_out_file, index=False)
    print(f"  [v] Đã lưu dự đoán chi tiết: {pred_out_file}")

    # Đánh giá các chỉ số
    models_dict = {
        "BERT (bert-base-uncased)": p_bert,
        "RoBERTa (roberta-base)": p_roberta,
        "Longformer (longformer-base-4096)": p_longformer,
        "Ensemble (Simple Soft Voting)": p_simple,
        "Ensemble (Weighted Soft Voting)": p_weighted,
        "Ensemble (Stacking Meta-Classifier)": p_stacking,
    }

    toefl_metrics_list = []
    toefl_results_json = {}

    print("\n" + "=" * 80)
    print("  KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP NGOẠI MIỀN TOEFL (252 BÀI LUẬN)")
    print("=" * 80)
    print(f"{'Mô hình / Chiến lược':<35} | {'Acc':<8} | {'Prec':<8} | {'Recall':<8} | {'F1':<8} | {'AUC':<8}")
    print("-" * 80)

    for m_name, p_vals in models_dict.items():
        res = evaluate_predictions(labels, p_vals)
        toefl_results_json[m_name] = res
        toefl_metrics_list.append({
            "model": m_name,
            "accuracy": res["accuracy"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
            "auc": res["auc"],
        })
        print(f"{m_name:<35} | {res['accuracy']*100:<7.2f}% | {res['precision']*100:<7.2f}% | {res['recall']*100:<7.2f}% | {res['f1']*100:<7.2f}% | {res['auc']*100:<7.2f}%")

    toefl_table_df = pd.DataFrame(toefl_metrics_list)
    table_out_file = os.path.join(OUTPUT_DIR, "toefl_comparison_table.csv")
    toefl_table_df.to_csv(table_out_file, index=False)
    print("-" * 80)
    print(f"  [v] Đã lưu bảng kết quả TOEFL: {table_out_file}")

    # So sánh độ suy giảm hiệu năng miền (Domain Shift Degradation)
    if os.path.exists(IN_DOMAIN_TABLE):
        in_table_df = pd.read_csv(IN_DOMAIN_TABLE)
        shift_analysis = {}

        print("\n" + "=" * 80)
        print("  PHÂN TÍCH ĐỘ SUY GIẢM HIỆU NĂNG MIỀN (DOMAIN SHIFT DEGRADATION: TOEFL vs GOSSIPCOP)")
        print("=" * 80)
        print(f"{'Mô hình':<25} | {'In-Domain Acc':<14} | {'TOEFL Acc':<12} | {'Δ Accuracy':<12} | {'Δ F1-Score':<12}")
        print("-" * 80)

        for _, row in toefl_table_df.iterrows():
            m_name = row["model"]
            in_row = in_table_df[in_table_df["model"] == m_name]
            if not in_row.empty:
                in_acc = in_row["accuracy"].values[0]
                in_f1 = in_row["f1"].values[0]
                toefl_acc = row["accuracy"]
                toefl_f1 = row["f1"]

                diff_acc = toefl_acc - in_acc
                diff_f1 = toefl_f1 - in_f1

                shift_analysis[m_name] = {
                    "in_domain_accuracy": float(in_acc),
                    "toefl_accuracy": float(toefl_acc),
                    "delta_accuracy": float(diff_acc),
                    "in_domain_f1": float(in_f1),
                    "toefl_f1": float(toefl_f1),
                    "delta_f1": float(diff_f1),
                }

                diff_acc_str = f"{diff_acc*100:+.2f}%"
                diff_f1_str = f"{diff_f1*100:+.2f}%"
                print(f"{m_name[:24]:<25} | {in_acc*100:<13.2f}% | {toefl_acc*100:<11.2f}% | {diff_acc_str:<12} | {diff_f1_str:<12}")

        shift_file = os.path.join(OUTPUT_DIR, "toefl_domain_shift_analysis.json")
        with open(shift_file, "w", encoding="utf-8") as f:
            json.dump(shift_analysis, f, indent=2, ensure_ascii=False)
        print("-" * 80)
        print(f"  [v] Đã lưu phân tích domain shift: {shift_file}")

        # Vẽ biểu đồ đối sánh In-Domain vs TOEFL
        plot_domain_shift_comparison(in_table_df, toefl_table_df)


def plot_domain_shift_comparison(in_df: pd.DataFrame, out_df: pd.DataFrame):
    """Vẽ biểu đồ đối sánh hiệu năng Nội miền (GossipCop) vs Ngoại miền (TOEFL)."""
    short_names = ["BERT", "RoBERTa", "Longformer", "Simple Voting", "Weighted Voting\n(Best)", "Stacking (LR)"]
    in_accs = in_df["accuracy"].values * 100
    out_accs = out_df["accuracy"].values * 100

    in_f1s = in_df["f1"].values * 100
    out_f1s = out_df["f1"].values * 100

    x = np.arange(len(short_names))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # 1. So sánh Accuracy
    rects1 = ax1.bar(x - width/2, in_accs, width, label="Nội miền (GossipCop++)", color="#1f77b4", alpha=0.9, edgecolor="black")
    rects2 = ax1.bar(x + width/2, out_accs, width, label="Ngoại miền (TOEFL Essays)", color="#ff7f0e", alpha=0.9, edgecolor="black")
    ax1.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Cross-Domain Generalization: Accuracy\n(In-Domain vs Out-of-Domain)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, fontsize=9.5, fontweight="bold")
    ax1.legend(loc="lower left", fontsize=10)
    ax1.set_ylim([70, 102])

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#d62728")

    # 2. So sánh F1-Score
    rects3 = ax2.bar(x - width/2, in_f1s, width, label="Nội miền (GossipCop++)", color="#2ca02c", alpha=0.9, edgecolor="black")
    rects4 = ax2.bar(x + width/2, out_f1s, width, label="Ngoại miền (TOEFL Essays)", color="#d62728", alpha=0.9, edgecolor="black")
    ax2.set_ylabel("F1-Score (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Cross-Domain Generalization: F1-Score\n(In-Domain vs Out-of-Domain)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, fontsize=9.5, fontweight="bold")
    ax2.legend(loc="lower left", fontsize=10)
    ax2.set_ylim([70, 102])

    for rect in rects3:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    for rect in rects4:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#d62728")

    plt.tight_layout()
    chart_out = os.path.join(FIGURES_DIR, "toefl_domain_shift_comparison.png")
    plt.savefig(chart_out, dpi=300)
    plt.close()
    print(f"  [v] Đã lưu biểu đồ so sánh ngoại miền: {chart_out}")


if __name__ == "__main__":
    main()

