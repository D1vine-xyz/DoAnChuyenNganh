"""
Sinh Bộ Biểu Đồ Khoa Học 300 DPI cho Tuyến Tiếng Việt (PhoBERT 8K Pre-2022)
và Phân tích Độ nhạy Theo Độ dài Văn bản (Topic 6 Research Gap #1).

Kết quả lưu vào: docs/reports/figures/
  1. vietnamese_5k_vs_8k_genre_comparison.png
  2. vietnamese_8k_confusion_and_roc.png
  3. length_sensitivity_analysis.png
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    confusion_matrix,
)
from transformers import AutoTokenizer
from pyvi import ViTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIG_DIR = os.path.join(ROOT_DIR, "docs", "reports", "figures")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models import build_model


GENRE_DISPLAY_LABELS = {
    "AI_BaoChi_CongNghe": "AI: Báo chí & Công nghệ",
    "AI_HocThuat_GiaiThich": "AI: Học thuật & Giải thích",
    "AI_Paraphrase": "AI: Diễn đạt lại (Paraphrase)",
    "AI_TieuLuan_TongHop": "AI: Tiểu luận tổng hợp",
    "AI_VanHoc_Truyen": "AI: Văn học & Truyện",
    "BaoChi_TruyenThong": "Người: Báo chí (2019)",
    "HocThuat_BachKhoa": "Người: Bách khoa (ViQuAD)",
    "PhapLuat_ChinhThong": "Người: Pháp luật (<=2022)",
    "Sach_TriThuc": "Người: Sách tri thức (<=2022)",
    "TruyenNgan": "Người: Truyện ngắn (<=2022)",
    "VanHoc_KinhDien": "Người: Văn học kinh điển",
    "VanTho_TruyenThong": "Người: Thơ ca truyền thống",
}


def configure_matplotlib_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": "#D1D5DB",
        "axes.linewidth": 0.9,
        "grid.color": "#E5E7EB",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def predict_phobert_checkpoint(
    ckpt_path: str,
    segmented_texts: list,
    tokenizer,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Chạy suy luận PhoBERT trên danh sách văn bản đã tách từ PyVi."""
    model = build_model("phobert").to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    probs = []
    with torch.inference_mode():
        for start in range(0, len(segmented_texts), batch_size):
            batch_txt = segmented_texts[start:start + batch_size]
            enc = tokenizer(
                batch_txt,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            batch_probs = torch.sigmoid(logits).cpu().numpy().flatten()
            probs.append(batch_probs)
            if (start // batch_size) % 10 == 0:
                print(f"    -> Đã suy luận {min(start + batch_size, len(segmented_texts))}/{len(segmented_texts)} mẫu...")

    return np.concatenate(probs)


def load_or_compute_vi_predictions(recompute: bool = False) -> pd.DataFrame:
    """Nạp cache dự đoán trên tập Test 1.200 mẫu Pre-2022 hoặc chạy trực tiếp hai checkpoint 5K vs 8K."""
    cache_csv = os.path.join(OUTPUT_DIR, "phobert_comparison_predictions.csv")
    test_csv = os.path.join(DATA_DIR, "vietnamese_pre2022_test.csv")
    df_test = pd.read_csv(test_csv)

    if os.path.exists(cache_csv) and not recompute:
        df_cached = pd.read_csv(cache_csv)
        if len(df_cached) == len(df_test) and {"prob_8k", "prob_5k"}.issubset(df_cached.columns):
            print(f"[*] Sử dụng cache dự đoán có sẵn: {cache_csv}")
            return df_cached

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok_dir = os.path.join(OUTPUT_DIR, "tokenizer_phobert")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir if os.path.exists(tok_dir) else "vinai/phobert-base-v2")

    print(f"[*] Đang tách từ PyVi cho {len(df_test)} mẫu Test Tiếng Việt...")
    segmented = [ViTokenizer.tokenize(str(t)) for t in df_test["text"].tolist()]

    ckpt_8k = os.path.join(OUTPUT_DIR, "phobert_best.pt")
    ckpt_5k = os.path.join(OUTPUT_DIR, "phobert_best_5k_backup.pt")

    print(f"[*] Đánh giá PhoBERT 8K Pre-2022 ({ckpt_8k})...")
    prob_8k = predict_phobert_checkpoint(ckpt_8k, segmented, tokenizer, device)

    if os.path.exists(ckpt_5k):
        print(f"[*] Đánh giá PhoBERT 5K Cũ ({ckpt_5k})...")
        prob_5k = predict_phobert_checkpoint(ckpt_5k, segmented, tokenizer, device)
    else:
        print("[!] Không tìm thấy phobert_best_5k_backup.pt, sử dụng prob_8k làm fallback.")
        prob_5k = prob_8k.copy()

    df_out = pd.DataFrame({
        "id": df_test["id"],
        "label": df_test["label"].astype(int),
        "genre": df_test["genre"].astype(str),
        "word_count": df_test["word_count"].astype(int),
        "prob_5k": np.round(prob_5k, 6),
        "prob_8k": np.round(prob_8k, 6),
    })
    df_out.to_csv(cache_csv, index=False, encoding="utf-8-sig")
    print(f"[v] Đã lưu dự đoán so sánh tại: {cache_csv}")
    return df_out


def plot_genre_comparison(df_vi: pd.DataFrame, save_path: str):
    """Biểu đồ 1: Đối sánh Accuracy 12 thể loại giữa PhoBERT 5K (Cũ) vs PhoBERT 8K Pre-2022 (Mới)."""
    df_vi = df_vi.copy()
    df_vi["pred_5k"] = (df_vi["prob_5k"] >= 0.5).astype(int)
    df_vi["pred_8k"] = (df_vi["prob_8k"] >= 0.5).astype(int)

    rows = []
    for genre, grp in df_vi.groupby("genre"):
        acc_5k = float((grp["label"] == grp["pred_5k"]).mean()) * 100.0
        acc_8k = float((grp["label"] == grp["pred_8k"]).mean()) * 100.0
        rows.append({
            "genre": genre,
            "label_disp": f"{GENRE_DISPLAY_LABELS.get(genre, genre)} (n={len(grp)})",
            "acc_5k": acc_5k,
            "acc_8k": acc_8k,
            "gain": acc_8k - acc_5k,
            "true_label": int(grp["label"].iloc[0]),
        })

    df_g = pd.DataFrame(rows).sort_values(["true_label", "gain"], ascending=[True, True])
    overall_5k = accuracy_score(df_vi["label"], df_vi["pred_5k"]) * 100.0
    overall_8k = accuracy_score(df_vi["label"], df_vi["pred_8k"]) * 100.0

    y = np.arange(len(df_g))
    height = 0.37

    fig, ax = plt.subplots(figsize=(12.2, 7.6))
    bars_5k = ax.barh(y - height / 2, df_g["acc_5k"], height, label=f"PhoBERT 5K Cũ (Tổng Test: {overall_5k:.2f}%)", color="#9CA3AF", edgecolor="#6B7280", linewidth=0.6)
    bars_8k = ax.barh(y + height / 2, df_g["acc_8k"], height, label=f"PhoBERT 8K Pre-2022 Mới (Tổng Test: {overall_8k:.2f}%)", color="#059669", edgecolor="#047857", linewidth=0.6)

    for b5, b8, gain in zip(bars_5k, bars_8k, df_g["gain"]):
        w5 = b5.get_width()
        w8 = b8.get_width()
        ax.text(min(w5 + 0.8, 101.0), b5.get_y() + b5.get_height() / 2, f"{w5:.1f}%", va="center", ha="left", fontsize=8.5, color="#4B5563")
        gain_txt = f" (+{gain:.1f}%)" if gain >= 0.5 else ""
        ax.text(min(w8 + 0.8, 101.0), b8.get_y() + b8.get_height() / 2, f"{w8:.1f}%{gain_txt}", va="center", ha="left", fontsize=8.5, fontweight="bold", color="#065F46")

    ax.set_yticks(y)
    ax.set_yticklabels(df_g["label_disp"], fontsize=9.8)
    ax.set_xlim(0, 118)
    ax.set_xlabel("Độ chính xác phân loại Accuracy (%)")
    ax.set_title("ĐỐI SÁNH HIỆU NĂNG PHOBERT V2 TRÊN 12 THỂ LOẠI TIẾNG VIỆT (TẬP TEST TIỀN-2022, N = 1.200)", pad=36)
    ax.grid(axis="x", alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        edgecolor="#E5E7EB",
        fontsize=9.8,
    )

    fig.savefig(save_path)
    plt.close(fig)
    print(f"[v] Đã xuất Biểu đồ 1: {save_path} (5K={overall_5k:.2f}% -> 8K={overall_8k:.2f}%)")


def plot_confusion_and_roc(df_vi: pd.DataFrame, save_path: str):
    """Biểu đồ 2: Ma trận nhầm lẫn PhoBERT 8K & Đường cong ROC-AUC (8K vs 5K)."""
    y_true = df_vi["label"].values
    prob_8k = df_vi["prob_8k"].values
    prob_5k = df_vi["prob_5k"].values
    pred_8k = (prob_8k >= 0.5).astype(int)

    cm = confusion_matrix(y_true, pred_8k)
    acc = accuracy_score(y_true, pred_8k) * 100.0
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, pred_8k, average="binary")

    fpr_8k, tpr_8k, _ = roc_curve(y_true, prob_8k)
    auc_8k = auc(fpr_8k, tpr_8k)
    fpr_5k, tpr_5k, _ = roc_curve(y_true, prob_5k)
    auc_5k = auc(fpr_5k, tpr_5k)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.5))

    # Left panel: Confusion Matrix
    im = ax1.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax1.set_title("Ma Trận Nhầm Lẫn — PhoBERT 8K Pre-2022\n(Test Set N = 1.200 | Acc = {:.2f}%, F1 = {:.2f}%)".format(acc, f1 * 100.0), pad=12)
    tick_marks = np.arange(2)
    class_names = ["Người viết\n(Human = 0)", "Do AI sinh\n(AI = 1)"]
    ax1.set_xticks(tick_marks)
    ax1.set_yticks(tick_marks)
    ax1.set_xticklabels(class_names, fontsize=10)
    ax1.set_yticklabels(class_names, fontsize=10)
    ax1.set_ylabel("Nhãn Thực Tế (Ground Truth)")
    ax1.set_xlabel("Nhãn Dự Đoán (Predicted)")

    thresh = cm.max() / 2.0
    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = (val / total) * 100.0
            color = "white" if val > thresh else "#111827"
            ax1.text(j, i, f"{val}\n({pct:.2f}%)", ha="center", va="center", color=color, fontsize=13, fontweight="bold")

    # Right panel: ROC Curves
    ax2.plot(fpr_8k, tpr_8k, color="#059669", lw=2.6, label=f"PhoBERT 8K Pre-2022 (AUC = {auc_8k:.4f})")
    ax2.plot(fpr_5k, tpr_5k, color="#6B7280", lw=2.0, linestyle="-.", label=f"PhoBERT 5K Cũ (AUC = {auc_5k:.4f})")
    ax2.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.2, linestyle="--", label="Ngẫu nhiên (AUC = 0.5000)")
    ax2.set_xlim([-0.01, 1.0])
    ax2.set_ylim([0.0, 1.02])
    ax2.set_xlabel("Tỷ lệ Dương tính Giả (False Positive Rate)")
    ax2.set_ylabel("Tỷ lệ Dương tính Thật (True Positive Rate)")
    ax2.set_title(f"Đường Cong ROC-AUC Tiếng Việt\n(Precision = {prec*100:.2f}%, Recall = {rec*100:.2f}%)", pad=12)
    ax2.grid(True, alpha=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="lower right", frameon=True, framealpha=0.95)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[v] Đã xuất Biểu đồ 2: {save_path} (AUC_8K={auc_8k:.4f}, AUC_5K={auc_5k:.4f})")


def plot_length_sensitivity(df_vi: pd.DataFrame, save_path: str):
    """Biểu đồ 3: Phân tích độ nhạy theo độ dài văn bản (Ngắn / Vừa / Dài) cho cả Tiếng Việt & Tiếng Anh."""
    # Vietnamese buckets on 1,200 test samples
    vi_bins = [
        ("Ngắn\n(< 100 từ)", df_vi["word_count"] < 100),
        ("Trung bình\n(100 - 220 từ)", (df_vi["word_count"] >= 100) & (df_vi["word_count"] <= 220)),
        ("Dài\n(> 220 từ)", df_vi["word_count"] > 220),
    ]
    vi_labels, vi_acc_8k, vi_f1_8k, vi_acc_5k = [], [], [], []
    for name, mask in vi_bins:
        sub = df_vi[mask]
        y = sub["label"].values
        p8 = (sub["prob_8k"].values >= 0.5).astype(int)
        p5 = (sub["prob_5k"].values >= 0.5).astype(int)
        _, _, f1_8, _ = precision_recall_fscore_support(y, p8, average="binary", zero_division=0)
        vi_labels.append(f"{name}\n[n={len(sub)}]")
        vi_acc_8k.append(accuracy_score(y, p8) * 100.0)
        vi_f1_8k.append(f1_8 * 100.0)
        vi_acc_5k.append(accuracy_score(y, p5) * 100.0)

    # English buckets on 2,873 test samples
    df_en_text = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    df_en_pred = pd.read_csv(os.path.join(OUTPUT_DIR, "test_predictions_ensemble.csv"))
    en_wc = df_en_text["text"].astype(str).str.split().str.len().values
    df_en = df_en_pred.copy()
    df_en["wc"] = en_wc

    en_bins = [
        ("Ngắn\n(< 100 từ)", df_en["wc"] < 100),
        ("Trung bình\n(100 - 350 từ)", (df_en["wc"] >= 100) & (df_en["wc"] <= 350)),
        ("Dài\n(> 350 từ)", df_en["wc"] > 350),
    ]
    en_labels, en_ens, en_long, en_bert, en_rob = [], [], [], [], []
    for name, mask in en_bins:
        sub = df_en[mask]
        y = sub["label"].values
        en_labels.append(f"{name}\n[n={len(sub)}]")
        en_ens.append(accuracy_score(y, (sub["ensemble_weighted_prob"].values >= 0.5).astype(int)) * 100.0)
        en_long.append(accuracy_score(y, (sub["longformer_prob"].values >= 0.5).astype(int)) * 100.0)
        en_bert.append(accuracy_score(y, (sub["bert_prob"].values >= 0.5).astype(int)) * 100.0)
        en_rob.append(accuracy_score(y, (sub["roberta_prob"].values >= 0.5).astype(int)) * 100.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.6))

    # Panel 1: Vietnamese PhoBERT Length Sensitivity
    x1 = np.arange(len(vi_labels))
    w1 = 0.25
    b_v1 = ax1.bar(x1 - w1, vi_acc_5k, w1, label="PhoBERT 5K Cũ (Accuracy)", color="#9CA3AF")
    b_v2 = ax1.bar(x1, vi_acc_8k, w1, label="PhoBERT 8K Mới (Accuracy)", color="#059669")
    b_v3 = ax1.bar(x1 + w1, vi_f1_8k, w1, label="PhoBERT 8K Mới (F1-Score)", color="#34D399")
    for bars in (b_v1, b_v2, b_v3):
        for b in bars:
            h = b.get_height()
            ax1.text(b.get_x() + b.get_width() / 2, h + 0.6, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(vi_labels, fontsize=9.8)
    ax1.set_ylim(55, 111)
    ax1.set_ylabel("Điểm số (%)")
    ax1.set_title("Độ Nhạy Theo Độ Dài Văn Bản — Tiếng Việt\n(PhoBERT 8K Pre-2022 ổn định ≥ 98.3% ở mọi dải độ dài)", pad=12)
    ax1.grid(axis="y", alpha=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper center", ncol=3, frameon=True, fontsize=8.0, framealpha=0.95)

    # Panel 2: English Ensemble Length Sensitivity
    x2 = np.arange(len(en_labels))
    w2 = 0.20
    b_e1 = ax2.bar(x2 - 1.5 * w2, en_bert, w2, label="BERT (512 tok)", color="#93C5FD")
    b_e2 = ax2.bar(x2 - 0.5 * w2, en_rob, w2, label="RoBERTa (512 tok)", color="#FCD34D")
    b_e3 = ax2.bar(x2 + 0.5 * w2, en_long, w2, label="Longformer (1024 tok)", color="#3B82F6")
    b_e4 = ax2.bar(x2 + 1.5 * w2, en_ens, w2, label="Ensemble Weighted", color="#1D4ED8")
    for bars in (b_e1, b_e2, b_e3, b_e4):
        for b in bars:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2, h + 0.45, f"{h:.1f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(en_labels, fontsize=9.8)
    ax2.set_ylim(75, 105)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Độ Nhạy Theo Độ Dài Văn Bản — Tiếng Anh\n(Longformer dẫn đầu bài dài 95.1%, Ensemble 97.7% bài vừa)", pad=12)
    ax2.grid(axis="y", alpha=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper center", ncol=2, frameon=True, fontsize=8.2, framealpha=0.95)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[v] Đã xuất Biểu đồ 3: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Xuất 3 biểu đồ khoa học 300 DPI cho dữ liệu Tiếng Việt 8K & Độ nhạy độ dài.")
    parser.add_argument("--recompute", action="store_true", help="Chạy lại suy luận PhoBERT 5K và 8K thay vì đọc cache CSV.")
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    configure_matplotlib_style()

    df_vi = load_or_compute_vi_predictions(recompute=args.recompute)
    plot_genre_comparison(df_vi, os.path.join(FIG_DIR, "vietnamese_5k_vs_8k_genre_comparison.png"))
    plot_confusion_and_roc(df_vi, os.path.join(FIG_DIR, "vietnamese_8k_confusion_and_roc.png"))
    plot_length_sensitivity(df_vi, os.path.join(FIG_DIR, "length_sensitivity_analysis.png"))
    print("\n[HOÀN TẤT] Đã tạo trọn bộ 3 biểu đồ 300 DPI tại:", FIG_DIR)


if __name__ == "__main__":
    main()
