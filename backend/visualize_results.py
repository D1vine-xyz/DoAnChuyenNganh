"""
Script trực quan hóa kết quả thực nghiệm Đồ án (Ngày 12)
Tạo các biểu đồ khoa học chất lượng cao (300 DPI) cho báo cáo và slide bảo vệ:
1. roc_curves_comparison.png: So sánh đường cong ROC của cả 6 mô hình
2. confusion_matrix_weighted_ensemble.png: Ma trận nhầm lẫn của mô hình Weighted Soft Voting
3. confusion_matrices_all_models.png: Ma trận nhầm lẫn của cả 6 mô hình cạnh nhau
4. metrics_comparison_bar.png: Biểu đồ cột đối sánh Accuracy, Precision, Recall, F1
5. ensemble_weights_distribution.png: Biểu đồ trọng số tối ưu học được của các mô hình
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Thiết lập phong cách hiển thị chuẩn ấn phẩm khoa học
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#eeeeee"
plt.rcParams["grid.linestyle"] = "--"

FIGURES_DIR = os.path.join("reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

PRED_FILE = os.path.join("output", "test_predictions_ensemble.csv")
TABLE_FILE = os.path.join("output", "ensemble_comparison_table.csv")
WEIGHTS_FILE = os.path.join("output", "ensemble_weights.json")

# Danh mục ánh xạ tên mô hình và màu sắc đại diện
MODEL_CONFIGS = [
    {
        "key": "bert_prob",
        "name": "BERT (bert-base-uncased)",
        "short_name": "BERT",
        "color": "#1f77b4",
        "ls": "-.",
        "lw": 1.8,
    },
    {
        "key": "roberta_prob",
        "name": "RoBERTa (roberta-base)",
        "short_name": "RoBERTa",
        "color": "#ff7f0e",
        "ls": "--",
        "lw": 1.8,
    },
    {
        "key": "longformer_prob",
        "name": "Longformer (longformer-base-4096)",
        "short_name": "Longformer",
        "color": "#2ca02c",
        "ls": ":",
        "lw": 2.0,
    },
    {
        "key": "ensemble_simple_prob",
        "name": "Ensemble (Simple Soft Voting)",
        "short_name": "Simple Voting",
        "color": "#9467bd",
        "ls": "-",
        "lw": 2.0,
    },
    {
        "key": "ensemble_weighted_prob",
        "name": "Ensemble (Weighted Soft Voting)",
        "short_name": "Weighted Voting (Best)",
        "color": "#d62728",
        "ls": "-",
        "lw": 3.0,
    },
    {
        "key": "ensemble_stacking_prob",
        "name": "Ensemble (Stacking Meta-Classifier)",
        "short_name": "Stacking (LR)",
        "color": "#8c564b",
        "ls": "-.",
        "lw": 2.0,
    },
]


def plot_roc_curves(df: pd.DataFrame):
    """Vẽ đường cong ROC so sánh cả 6 mô hình trên tập Test."""
    print("[*] Đang vẽ biểu đồ ROC Curves...")
    plt.figure(figsize=(9, 7), dpi=300)
    y_true = df["label"].values

    for cfg in MODEL_CONFIGS:
        prob = df[cfg["key"]].values
        fpr, tpr, _ = roc_curve(y_true, prob)
        roc_auc = auc(fpr, tpr)
        label_text = f"{cfg['short_name']} (AUC = {roc_auc:.4f})"
        if "Best" in cfg["short_name"]:
            label_text = f"★ {label_text} [CHAMPION]"
        plt.plot(
            fpr, tpr,
            label=label_text,
            color=cfg["color"],
            linestyle=cfg["ls"],
            linewidth=cfg["lw"],
            alpha=0.95,
        )

    # Đường tham chiếu ngẫu nhiên
    plt.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Random Guess (AUC = 0.5000)")

    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold", labelpad=8)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, fontweight="bold", labelpad=8)
    plt.title("ROC Curves Comparison - LLM-Generated Text Detection\nIn-Domain Test Set (GossipCop++, N = 2,873)",
              fontsize=14, fontweight="bold", pad=14)
    plt.legend(loc="lower right", fontsize=10, frameon=True, facecolor="white", edgecolor="#cccccc", shadow=True)
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "roc_curves_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [v] Đã lưu: {out_path}")


def plot_confusion_matrix_best(df: pd.DataFrame):
    """Vẽ ma trận nhầm lẫn chi tiết cho mô hình Weighted Soft Voting."""
    print("[*] Đang vẽ ma trận nhầm lẫn cho mô hình Weighted Soft Voting...")
    y_true = df["label"].values
    y_pred = (df["ensemble_weighted_prob"].values >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = len(y_true)

    labels = [
        [f"True Negative (TN)\n{tn:,}\n({tn/total*100:.2f}%)", f"False Positive (FP)\n{fp:,}\n({fp/total*100:.2f}%)"],
        [f"False Negative (FN)\n{fn:,}\n({fn/total*100:.2f}%)", f"True Positive (TP)\n{tp:,}\n({tp/total*100:.2f}%)"],
    ]

    plt.figure(figsize=(7, 6), dpi=300)
    ax = sns.heatmap(
        cm,
        annot=np.array(labels),
        fmt="",
        cmap="Blues",
        cbar=True,
        annot_kws={"size": 11, "fontweight": "bold"},
        linewidths=1.5,
        linecolor="white",
        xticklabels=["Dự đoán: Người viết (0)", "Dự đoán: AI sinh (1)"],
        yticklabels=["Thực tế: Người viết (0)", "Thực tế: AI sinh (1)"],
    )

    acc = (tp + tn) / total * 100
    prec = tp / (tp + fp) * 100
    rec = tp / (tp + fn) * 100
    f1 = 2 * prec * rec / (prec + rec)

    plt.title(f"Confusion Matrix - Weighted Soft Voting Ensemble\nAccuracy: {acc:.2f}% | Precision: {prec:.2f}% | Recall: {rec:.2f}% | F1: {f1:.2f}%\n(N = 2,873 test samples)",
              fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "confusion_matrix_weighted_ensemble.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [v] Đã lưu: {out_path}")


def plot_all_confusion_matrices(df: pd.DataFrame):
    """Vẽ lưới 2x3 ma trận nhầm lẫn của toàn bộ 6 mô hình để so sánh đối kháng."""
    print("[*] Đang vẽ lưới 2x3 ma trận nhầm lẫn của cả 6 mô hình...")
    y_true = df["label"].values

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
    axes = axes.flatten()

    for idx, cfg in enumerate(MODEL_CONFIGS):
        ax = axes[idx]
        prob = df[cfg["key"]].values
        y_pred = (prob >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        total = len(y_true)
        acc = (tp + tn) / total * 100
        prec = tp / (tp + fp) * 100
        rec = tp / (tp + fn) * 100

        labels = [
            [f"TN: {tn:,}\n({tn/total*100:.1f}%)", f"FP: {fp:,}\n({fp/total*100:.1f}%)"],
            [f"FN: {fn:,}\n({fn/total*100:.1f}%)", f"TP: {tp:,}\n({tp/total*100:.1f}%)"],
        ]

        cmap = "Reds" if "Weighted" in cfg["name"] else "Blues"
        sns.heatmap(
            cm,
            annot=np.array(labels),
            fmt="",
            cmap=cmap,
            cbar=False,
            ax=ax,
            annot_kws={"size": 10, "fontweight": "bold"},
            linewidths=1.0,
            linecolor="white",
            xticklabels=["Người (0)", "AI (1)"],
            yticklabels=["Người (0)", "AI (1)"],
        )
        title_suffix = " ★ [BEST]" if "Weighted" in cfg["name"] else ""
        ax.set_title(f"{cfg['short_name']}{title_suffix}\nAcc: {acc:.2f}% | Prec: {prec:.2f}% | Rec: {rec:.2f}%",
                     fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted Label", fontsize=9, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=9, fontweight="bold")

    plt.suptitle("Confusion Matrices Comparison Across All 6 Evaluated Models (N = 2,873)",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "confusion_matrices_all_models.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [v] Đã lưu: {out_path}")


def plot_metrics_comparison_bar(table_df: pd.DataFrame):
    """Vẽ biểu đồ cột nhóm (Grouped Bar Chart) so sánh 4 thang đo."""
    print("[*] Đang vẽ biểu đồ cột đối sánh 4 thang đo...")
    models = table_df["model"].values
    short_names = [
        "BERT",
        "RoBERTa",
        "Longformer",
        "Simple Voting",
        "Weighted Voting\n(Champion)",
        "Stacking (LR)",
    ]

    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    x = np.arange(len(short_names))
    width = 0.18

    plt.figure(figsize=(13, 7), dpi=300)

    for m_idx, (m_col, m_label, col) in enumerate(zip(metrics, metric_labels, colors)):
        values = table_df[m_col].values * 100
        offset = (m_idx - 1.5) * width
        rects = plt.bar(x + offset, values, width, label=m_label, color=col, alpha=0.9, edgecolor="black", linewidth=0.6)
        
        # Ghi giá trị trên đầu cột
        for rect in rects:
            height = rect.get_height()
            plt.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", rotation=45
            )

    plt.xlabel("Mô hình & Chiến lược Kết hợp", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("Tỷ lệ phần trăm (%)", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylim([88, 102])
    plt.title("Performance Benchmarking Across Base Models and Ensemble Strategies\nIn-Domain Test Dataset (GossipCop++, N = 2,873)",
              fontsize=14, fontweight="bold", pad=15)
    plt.xticks(x, short_names, fontsize=10, fontweight="bold")
    plt.legend(loc="upper left", fontsize=11, frameon=True, facecolor="white", edgecolor="#cccccc", shadow=True)
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "metrics_comparison_bar.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [v] Đã lưu: {out_path}")


def plot_ensemble_weights(weights_file: str):
    """Vẽ biểu đồ tròn và thanh ngang phân bổ trọng số Ensemble tối ưu."""
    print("[*] Đang vẽ biểu đồ trọng số tối ưu Ensemble...")
    with open(weights_file, "r", encoding="utf-8") as f:
        weights_data = json.load(f)["weights"]

    names = ["Longformer\n(1024 tokens)", "BERT\n(512 tokens)", "RoBERTa\n(512 tokens)"]
    vals = [weights_data["longformer"] * 100, weights_data["bert"] * 100, weights_data["roberta"] * 100]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]
    explode = (0.05, 0.02, 0.02)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # 1. Pie Chart
    wedges, texts, autotexts = ax1.pie(
        vals,
        labels=names,
        autopct="%1.2f%%",
        startangle=140,
        colors=colors,
        explode=explode,
        shadow=True,
        textprops={"fontsize": 11, "fontweight": "bold"},
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax1.set_title("Optimal Ensemble Weight Distribution\n(SLSQP Optimization on Validation Set, N = 2,872)",
                  fontsize=12, fontweight="bold", pad=10)

    # 2. Horizontal Bar Chart
    y_pos = np.arange(len(names))
    bars = ax2.barh(y_pos, vals, color=colors, edgecolor="black", linewidth=0.8, alpha=0.9)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=10, fontweight="bold")
    ax2.set_xlabel("Đóng góp trọng số (%)", fontsize=11, fontweight="bold")
    ax2.set_xlim([0, 60])
    ax2.invert_yaxis()

    for bar in bars:
        width = bar.get_width()
        ax2.annotate(
            f"{width:.2f}%",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left", va="center",
            fontsize=11, fontweight="bold", color="#333333"
        )
    ax2.set_title("Learned Soft Voting Weights ($w_i$)", fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "ensemble_weights_distribution.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [v] Đã lưu: {out_path}")


def main():
    print("=" * 80)
    print("  TRỰC QUAN HÓA KẾT QUẢ THỰC NGHIỆM ĐỒ ÁN CHUYÊN NGÀNH (NGÀY 12)")
    print("=" * 80)

    if not os.path.exists(PRED_FILE):
        print(f"[!] Không tìm thấy tệp dự đoán: {PRED_FILE}")
        return

    pred_df = pd.read_csv(PRED_FILE)
    table_df = pd.read_csv(TABLE_FILE)

    plot_roc_curves(pred_df)
    plot_confusion_matrix_best(pred_df)
    plot_all_confusion_matrices(pred_df)
    plot_metrics_comparison_bar(table_df)
    plot_ensemble_weights(WEIGHTS_FILE)

    print("\n" + "=" * 80)
    print(f"  [v] TOÀN BỘ 5 BIỂU ĐỒ 300 DPI ĐÃ ĐƯỢC XUẤT THÀNH CÔNG VÀO: {FIGURES_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()

