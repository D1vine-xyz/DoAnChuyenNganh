"""
Module Kiến trúc Mô hình Kết hợp (Ensemble Learning: BERT + RoBERTa + Longformer)
Phục vụ Đồ án Chuyên ngành (Chủ đề 6): Phát hiện văn bản do LLM sinh sử dụng mô hình kết hợp (Ensemble PLMs).

Triển khai 3 chiến lược kết hợp cốt lõi:
1. Simple Soft Voting (Bình chọn mềm đồng đều xác suất):
   P_ensemble = (P_bert + P_roberta + P_longformer) / 3

2. Weighted Soft Voting (Bình chọn mềm tối ưu trọng số):
   P_ensemble = w1 * P_bert + w2 * P_roberta + w3 * P_longformer
   (Trọng số w1, w2, w3 được tối ưu trên tập Validation bằng thuật toán tối ưu hóa)

3. Stacking Meta-Classifier (Mô hình siêu học cấp 2):
   Đầu vào là véc-tơ xác suất/logits từ 3 PLMs -> Meta-Learner (Logistic Regression)
   học ranh giới phân tách nhị phân tối ưu.

Đánh giá toàn diện trên tập Test nội miền (2,873 mẫu) và xuất báo cáo so sánh đa tiêu chí.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, Tuple, Optional, List, Union

# Đảm bảo UTF-8 trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
import joblib

# Import từ các module trong cùng dự án
from models import build_model


class TextDataset(Dataset):
    """Dataset đơn giản phục vụ trích xuất dự đoán theo batch."""
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Tính toán toàn diện các chỉ số phân loại nhị phân."""
    y_pred = (y_prob >= threshold).astype(int)
    y_true_int = y_true.astype(int)

    acc = accuracy_score(y_true_int, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_int, y_pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true_int, y_prob)
    except Exception:
        auc = 0.5

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
        "threshold": float(threshold),
    }


class EnsembleDetector:
    """
    Lớp điều phối mô hình kết hợp Ensemble (BERT, RoBERTa, Longformer).
    Hỗ trợ dự đoán đơn lẻ, theo lô (batch), và suy luận văn bản thô.
    """

    def __init__(
        self,
        bert_path: Optional[str] = None,
        roberta_path: Optional[str] = None,
        longformer_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        use_fp16: bool = True,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and (self.device.type == "cuda")

        print(f"[*] Khởi tạo EnsembleDetector trên thiết bị: {self.device}")
        if self.device.type == "cuda":
            print(f"    - GPU: {torch.cuda.get_device_name(0)}")
            print(f"    - Mixed Precision FP16: {self.use_fp16}")

        # Khởi tạo Tokenizers
        print("[*] Đang nạp Tokenizers...")
        self.tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer_roberta = AutoTokenizer.from_pretrained("roberta-base")
        self.tokenizer_longformer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

        # Nạp mô hình BERT
        self.model_bert = None
        if bert_path and os.path.exists(bert_path):
            print(f"[*] Đang nạp trọng số BERT từ: {bert_path}")
            self.model_bert = build_model("bert").to(self.device)
            ckpt = torch.load(bert_path, map_location=self.device, weights_only=False)
            self.model_bert.load_state_dict(ckpt["model_state_dict"])
            self.model_bert.eval()
            print("  [v] BERT sẵn sàng.")

        # Nạp mô hình RoBERTa
        self.model_roberta = None
        if roberta_path and os.path.exists(roberta_path):
            print(f"[*] Đang nạp trọng số RoBERTa từ: {roberta_path}")
            self.model_roberta = build_model("roberta").to(self.device)
            ckpt = torch.load(roberta_path, map_location=self.device, weights_only=False)
            self.model_roberta.load_state_dict(ckpt["model_state_dict"])
            self.model_roberta.eval()
            print("  [v] RoBERTa sẵn sàng.")

        # Nạp mô hình Longformer
        self.model_longformer = None
        if longformer_path and os.path.exists(longformer_path):
            print(f"[*] Đang nạp trọng số Longformer từ: {longformer_path}")
            self.model_longformer = build_model("longformer").to(self.device)
            ckpt = torch.load(longformer_path, map_location=self.device, weights_only=False)
            self.model_longformer.load_state_dict(ckpt["model_state_dict"])
            self.model_longformer.eval()
            print("  [v] Longformer sẵn sàng.")

        # Trọng số mặc định cho Weighted Voting (1/3 cho mỗi mô hình)
        self.weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

        # Meta-learner cho Stacking
        self.meta_classifier = None

    def _predict_bert_batch(self, texts: List[str], max_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """Dự đoán theo batch với BERT."""
        encoded = self.tokenizer_bert(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.use_fp16):
                logits = self.model_bert(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    token_type_ids=encoded.get("token_type_ids"),
                )
                probs = torch.sigmoid(logits)

        return logits.cpu().numpy(), probs.cpu().numpy()

    def _predict_roberta_batch(self, texts: List[str], max_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """Dự đoán theo batch với RoBERTa."""
        encoded = self.tokenizer_roberta(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.use_fp16):
                logits = self.model_roberta(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )
                probs = torch.sigmoid(logits)

        return logits.cpu().numpy(), probs.cpu().numpy()

    def _predict_longformer_batch(self, texts: List[str], max_length: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        """Dự đoán theo batch với Longformer (kèm global attention cho token đầu)."""
        encoded = self.tokenizer_longformer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        global_attention_mask = torch.zeros_like(encoded["input_ids"])
        global_attention_mask[:, 0] = 1

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.use_fp16):
                logits = self.model_longformer(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    global_attention_mask=global_attention_mask,
                )
                probs = torch.sigmoid(logits)

        return logits.cpu().numpy(), probs.cpu().numpy()

    def extract_predictions(
        self,
        texts: List[str],
        batch_size: int = 16,
        show_progress: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Trích xuất song song logits và probabilities từ cả 3 mô hình cho toàn bộ danh sách văn bản.
        Trả về dict gồm:
        - bert_probs, bert_logits
        - roberta_probs, roberta_logits
        - longformer_probs, longformer_logits
        """
        num_samples = len(texts)
        num_batches = (num_samples + batch_size - 1) // batch_size

        bert_probs_list, bert_logits_list = [], []
        roberta_probs_list, roberta_logits_list = [], []
        longformer_probs_list, longformer_logits_list = [], []

        start_time = time.time()
        print(f"[*] Bắt đầu trích xuất dự đoán trên {num_samples:,} mẫu ({num_batches} batches)...")

        for b_idx in range(num_batches):
            b_start = b_idx * batch_size
            b_end = min(b_start + batch_size, num_samples)
            batch_texts = texts[b_start:b_end]

            # 1. BERT
            if self.model_bert:
                l, p = self._predict_bert_batch(batch_texts, max_length=512)
                bert_logits_list.append(l)
                bert_probs_list.append(p)

            # 2. RoBERTa
            if self.model_roberta:
                l, p = self._predict_roberta_batch(batch_texts, max_length=512)
                roberta_logits_list.append(l)
                roberta_probs_list.append(p)

            # 3. Longformer (batch size nhỏ hơn nếu cần để tránh OOM)
            if self.model_longformer:
                l, p = self._predict_longformer_batch(batch_texts, max_length=1024)
                longformer_logits_list.append(l)
                longformer_probs_list.append(p)

            if show_progress and ((b_idx + 1) % 25 == 0 or (b_idx + 1) == num_batches):
                elapsed = time.time() - start_time
                print(f"  Batch [{b_idx+1:>4}/{num_batches}] ({b_end:>5}/{num_samples}) | {elapsed:.1f}s")

        results = {}
        if self.model_bert:
            results["bert_logits"] = np.concatenate(bert_logits_list)
            results["bert_probs"] = np.concatenate(bert_probs_list)
        if self.model_roberta:
            results["roberta_logits"] = np.concatenate(roberta_logits_list)
            results["roberta_probs"] = np.concatenate(roberta_probs_list)
        if self.model_longformer:
            results["longformer_logits"] = np.concatenate(longformer_logits_list)
            results["longformer_probs"] = np.concatenate(longformer_probs_list)

        return results

    def fit_weights(self, val_probs: np.ndarray, val_labels: np.ndarray) -> np.ndarray:
        """
        Tìm trọng số [w_bert, w_roberta, w_longformer] tối ưu trên tập Validation
        nhằm tối thiểu hóa hàm mất mát Log Loss (Cross-Entropy).
        """
        print("[*] Đang tối ưu hóa trọng số Weighted Soft Voting trên tập Validation...")

        def objective(weights):
            w = weights / np.sum(weights)
            p_blend = np.dot(val_probs, w)
            p_blend = np.clip(p_blend, 1e-15, 1 - 1e-15)
            # Binary Cross Entropy Loss
            loss = -np.mean(val_labels * np.log(p_blend) + (1 - val_labels) * np.log(1 - p_blend))
            return loss

        # Khởi tạo đều
        init_weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

        opt_res = minimize(
            objective,
            init_weights,
            bounds=bounds,
            method="SLSQP",
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        )

        best_weights = opt_res.x / np.sum(opt_res.x)
        self.weights = best_weights
        print(f"  [v] Trọng số tối ưu tìm được:")
        print(f"      - BERT       : {best_weights[0]:.4f}")
        print(f"      - RoBERTa    : {best_weights[1]:.4f}")
        print(f"      - Longformer : {best_weights[2]:.4f}")
        return best_weights

    def fit_stacking(self, val_features: np.ndarray, val_labels: np.ndarray):
        """Huấn luyện Meta-Classifier (Logistic Regression) trên tập Validation."""
        print("[*] Đang huấn luyện Meta-Classifier Stacking (Logistic Regression)...")
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(val_features, val_labels)
        self.meta_classifier = clf
        print(f"  [v] Hệ số Meta-Learner Coefs: {clf.coef_[0]} | Intercept: {clf.intercept_[0]:.4f}")
        return clf

    def predict_ensemble(
        self,
        bert_probs: np.ndarray,
        roberta_probs: np.ndarray,
        longformer_probs: np.ndarray,
        method: str = "weighted",
    ) -> np.ndarray:
        """
        Tính toán xác suất gộp theo phương pháp được chỉ định.
        :param method: 'simple', 'weighted', hoặc 'stacking'
        """
        prob_matrix = np.column_stack([bert_probs, roberta_probs, longformer_probs])

        if method == "simple":
            return np.mean(prob_matrix, axis=1)
        elif method == "weighted":
            return np.dot(prob_matrix, self.weights)
        elif method == "stacking":
            if self.meta_classifier is None:
                raise ValueError("Meta-classifier chưa được huấn luyện! Hãy gọi fit_stacking() trước.")
            return self.meta_classifier.predict_proba(prob_matrix)[:, 1]
        else:
            raise ValueError(f"Không hỗ trợ phương pháp: {method}. Chọn ['simple', 'weighted', 'stacking']")


def run_ensemble_evaluation(args):
    """Quy trình đánh giá thực nghiệm toàn diện Ensemble Pipeline."""
    print("=" * 80)
    print("  QUY TRÌNH THỰC NGHIỆM MÔ HÌNH HỌC KẾT HỢP (ENSEMBLE PLMs PIPELINE)")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Nạp dữ liệu Validation và Test
    print("\n[*] Đang nạp dữ liệu từ thư mục:", args.data_dir)
    val_df = pd.read_csv(os.path.join(args.data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))

    val_texts, val_labels = val_df["text"].tolist(), val_df["label"].values
    test_texts, test_labels = test_df["text"].tolist(), test_df["label"].values

    print(f"  [v] Tập Validation : {len(val_texts):,} mẫu")
    print(f"  [v] Tập Test       : {len(test_texts):,} mẫu")

    # 2. Khởi tạo Ensemble Detector
    detector = EnsembleDetector(
        bert_path=args.bert_path,
        roberta_path=args.roberta_path,
        longformer_path=args.longformer_path,
        use_fp16=args.fp16,
    )

    # 3. Trích xuất dự đoán trên tập Validation
    print("\n" + "-" * 60)
    print("  GIAI ĐOẠN 1: TRÍCH XUẤT VÀ TỐI ƯU TRÊN TẬP VALIDATION")
    print("-" * 60)
    val_preds = detector.extract_predictions(val_texts, batch_size=args.batch_size)

    val_prob_matrix = np.column_stack([
        val_preds["bert_probs"],
        val_preds["roberta_probs"],
        val_preds["longformer_probs"],
    ])

    # Tối ưu trọng số Weighted Soft Voting
    detector.fit_weights(val_prob_matrix, val_labels)

    # Huấn luyện Stacking Meta-Classifier
    detector.fit_stacking(val_prob_matrix, val_labels)

    # 4. Trích xuất dự đoán trên tập Test độc lập
    print("\n" + "-" * 60)
    print("  GIAI ĐOẠN 2: ĐÁNH GIÁ THỰC NGHIỆM TRÊN TẬP TEST NỘI MIỀN")
    print("-" * 60)
    test_preds = detector.extract_predictions(test_texts, batch_size=args.batch_size)

    test_prob_matrix = np.column_stack([
        test_preds["bert_probs"],
        test_preds["roberta_probs"],
        test_preds["longformer_probs"],
    ])

    # 5. Tính toán kết quả cho từng mô hình và từng chiến lược Ensemble
    p_bert = test_preds["bert_probs"]
    p_roberta = test_preds["roberta_probs"]
    p_longformer = test_preds["longformer_probs"]

    p_ens_simple = detector.predict_ensemble(p_bert, p_roberta, p_longformer, method="simple")
    p_ens_weighted = detector.predict_ensemble(p_bert, p_roberta, p_longformer, method="weighted")
    p_ens_stacking = detector.predict_ensemble(p_bert, p_roberta, p_longformer, method="stacking")

    results_table = []
    models_to_eval = [
        ("BERT (bert-base-uncased)", p_bert),
        ("RoBERTa (roberta-base)", p_roberta),
        ("Longformer (longformer-base-4096)", p_longformer),
        ("Ensemble (Simple Soft Voting)", p_ens_simple),
        ("Ensemble (Weighted Soft Voting)", p_ens_weighted),
        ("Ensemble (Stacking Meta-Classifier)", p_ens_stacking),
    ]

    all_metrics = {}
    print("\n" + "=" * 95)
    print(f"{'MÔ HÌNH / PHƯƠNG PHÁP':<38} | {'ACCURACY':<10} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10} | {'AUC-ROC':<10}")
    print("=" * 95)

    for name, probs in models_to_eval:
        m = compute_metrics(test_labels, probs)
        all_metrics[name] = m
        results_table.append({
            "model": name,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "auc": m["auc"],
        })
        print(f"{name:<38} | {m['accuracy']*100:>9.2f}% | {m['precision']*100:>9.2f}% | {m['recall']*100:>9.2f}% | {m['f1']*100:>9.2f}% | {m['auc']*100:>9.2f}%")

    print("=" * 95)

    # 6. Lưu trữ kết quả thực nghiệm
    df_results = pd.DataFrame(results_table)
    csv_path = os.path.join(args.output_dir, "ensemble_comparison_table.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n[*] Đã lưu bảng so sánh chi tiết: {csv_path}")

    json_path = os.path.join(args.output_dir, "ensemble_test_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"[*] Đã lưu file kết quả JSON: {json_path}")

    weights_path = os.path.join(args.output_dir, "ensemble_weights.json")
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump({
            "weights": {
                "bert": float(detector.weights[0]),
                "roberta": float(detector.weights[1]),
                "longformer": float(detector.weights[2]),
            }
        }, f, indent=2)
    print(f"[*] Đã lưu cấu hình trọng số Ensemble: {weights_path}")

    if detector.meta_classifier:
        meta_path = os.path.join(args.output_dir, "stacking_meta_learner.joblib")
        joblib.dump(detector.meta_classifier, meta_path)
        print(f"[*] Đã lưu mô hình Stacking Meta-Learner: {meta_path}")

    # Lưu file dự đoán xác suất để phục vụ vẽ biểu đồ ROC / Confusion Matrix
    preds_df = pd.DataFrame({
        "label": test_labels,
        "bert_prob": p_bert,
        "roberta_prob": p_roberta,
        "longformer_prob": p_longformer,
        "ensemble_simple_prob": p_ens_simple,
        "ensemble_weighted_prob": p_ens_weighted,
        "ensemble_stacking_prob": p_ens_stacking,
    })
    preds_csv_path = os.path.join(args.output_dir, "test_predictions_ensemble.csv")
    preds_df.to_csv(preds_csv_path, index=False)
    print(f"[*] Đã lưu xác suất dự đoán chi tiết: {preds_csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình Ensemble PLMs")
    parser.add_argument("--data_dir", type=str, default="data", help="Thư mục chứa val.csv, test.csv")
    parser.add_argument("--output_dir", type=str, default="output", help="Thư mục lưu kết quả")
    parser.add_argument("--bert_path", type=str, default="output/bert_best.pt", help="Đường dẫn bert_best.pt")
    parser.add_argument("--roberta_path", type=str, default="output/roberta_best.pt", help="Đường dẫn roberta_best.pt")
    parser.add_argument("--longformer_path", type=str, default="output/longformer_best.pt", help="Đường dẫn longformer_best.pt")
    parser.add_argument("--batch_size", type=int, default=16, help="Kích thước batch khi trích xuất dự đoán")
    parser.add_argument("--fp16", action="store_true", default=True, help="Sử dụng FP16 tăng tốc độ")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ensemble_evaluation(args)
