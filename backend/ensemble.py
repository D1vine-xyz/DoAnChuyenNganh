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
from scipy.optimize import minimize, minimize_scalar
import joblib

# Import từ các module trong cùng dự án
from models import build_model


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Tính Expected Calibration Error (ECE) đo độ trung thực xác suất dự đoán."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs <= hi if i == n_bins - 1 else probs < hi)
        if np.any(mask):
            acc_bin = float(np.mean((probs[mask] >= 0.5) == labels[mask]))
            conf_bin = float(np.mean(np.where(probs[mask] >= 0.5, probs[mask], 1.0 - probs[mask])))
            ece += (np.sum(mask) / n) * abs(acc_bin - conf_bin)
    return float(ece)


def fit_temperature_for_logits(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Hiệu chuẩn Temperature Scaling T* tối thiểu hóa Validation NLL."""
    logits_f = np.asarray(logits, dtype=np.float64)
    labels_f = np.asarray(labels, dtype=np.float64)

    def _nll(temp: float) -> float:
        scaled = np.clip(logits_f / temp, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-scaled))
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        return float(-np.mean(labels_f * np.log(p) + (1.0 - labels_f) * np.log(1.0 - p)))

    nll_before = _nll(1.0)
    p_before = 1.0 / (1.0 + np.exp(-np.clip(logits_f, -30.0, 30.0)))
    ece_before = compute_ece(p_before, labels_f)

    opt = minimize_scalar(_nll, bounds=(1.0, 4.5), method="bounded")
    best_t = float(opt.x)
    nll_after = _nll(best_t)
    p_after = 1.0 / (1.0 + np.exp(-np.clip(logits_f / best_t, -30.0, 30.0)))
    ece_after = compute_ece(p_after, labels_f)

    return {
        "temperature": round(best_t, 4),
        "val_nll_before": round(nll_before, 5),
        "val_nll_after": round(nll_after, 5),
        "val_ece_before": round(ece_before, 5),
        "val_ece_after": round(ece_after, 5),
    }


def apply_temperature(logits: np.ndarray, temp: float) -> np.ndarray:
    scaled = np.clip(np.asarray(logits, dtype=np.float64) / max(float(temp), 0.2), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-scaled))


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

        if self.device.type == "cpu":
            num_threads = os.cpu_count() or 4
            torch.set_num_threads(num_threads)
            print(f"[*] Tối ưu hóa CPU Multi-threading: {num_threads} threads (Intel MKL/OpenMP).")

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
            with torch.amp.autocast("cuda", enabled=self.use_fp16):
                logits = self.model_bert(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    token_type_ids=encoded.get("token_type_ids"),
                )
                probs = torch.sigmoid(logits)

        return logits.float().cpu().numpy(), probs.float().cpu().numpy()

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
            with torch.amp.autocast("cuda", enabled=self.use_fp16):
                logits = self.model_roberta(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )
                probs = torch.sigmoid(logits)

        return logits.float().cpu().numpy(), probs.float().cpu().numpy()

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
            with torch.amp.autocast("cuda", enabled=self.use_fp16):
                logits = self.model_longformer(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    global_attention_mask=global_attention_mask,
                )
                probs = torch.sigmoid(logits)

        return logits.float().cpu().numpy(), probs.float().cpu().numpy()

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
    def _resolve_csv(custom: Optional[str], fallback_name: str) -> str:
        if not custom:
            return os.path.join(args.data_dir, fallback_name)
        return custom if os.path.isabs(custom) or os.path.exists(custom) else os.path.join(args.data_dir, custom)

    val_csv = _resolve_csv(args.val_file, "val.csv")
    test_csv = _resolve_csv(args.test_file, "test.csv")
    print(f"\n[*] Đang nạp dữ liệu:\n    - Val : {val_csv}\n    - Test: {test_csv}")
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    val_texts, val_labels = val_df["text"].astype(str).tolist(), val_df["label"].values
    test_texts, test_labels = test_df["text"].astype(str).tolist(), test_df["label"].values

    if args.max_samples is not None and args.max_samples > 0:
        print(f"[*] Chế độ giới hạn mẫu (max_samples): chỉ lấy {args.max_samples} mẫu đầu tiên để kiểm thử nhanh.")
        val_texts, val_labels = val_texts[:args.max_samples], val_labels[:args.max_samples]
        test_texts, test_labels = test_texts[:args.max_samples], test_labels[:args.max_samples]

    print(f"  [v] Tập Validation : {len(val_texts):,} mẫu")
    print(f"  [v] Tập Test       : {len(test_texts):,} mẫu")

    # 2. Khởi tạo Ensemble Detector
    detector = EnsembleDetector(
        bert_path=args.bert_path,
        roberta_path=args.roberta_path,
        longformer_path=args.longformer_path,
        use_fp16=args.fp16,
    )

    # 3. Trích xuất dự đoán trên tập Validation + Hiệu chuẩn Temperature Scaling
    print("\n" + "-" * 60)
    print("  GIAI ĐOẠN 1: TRÍCH XUẤT, TEMPERATURE SCALING & TỐI ƯU TRÊN VAL")
    print("-" * 60)
    val_preds = detector.extract_predictions(val_texts, batch_size=args.batch_size)

    calib_info = {
        "bert": fit_temperature_for_logits(val_preds["bert_logits"], val_labels),
        "roberta": fit_temperature_for_logits(val_preds["roberta_logits"], val_labels),
        "longformer": fit_temperature_for_logits(val_preds["longformer_logits"], val_labels),
    }
    temps = {k: calib_info[k]["temperature"] for k in ("bert", "roberta", "longformer")}
    print("[*] Kết quả Hiệu chuẩn Temperature Scaling (EN Validation):")
    for k in ("bert", "roberta", "longformer"):
        c = calib_info[k]
        print(
            f"    - {k:<10}: T*={c['temperature']:.4f} | "
            f"NLL {c['val_nll_before']:.4f}->{c['val_nll_after']:.4f} | "
            f"ECE {c['val_ece_before']:.4f}->{c['val_ece_after']:.4f}"
        )

    temp_path = os.path.join(args.output_dir, "en_temperature.json")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump({"temperatures": temps, "calibration_metrics": calib_info}, f, indent=2, ensure_ascii=False)
    print(f"  [v] Đã lưu cấu hình Temperature Scaling EN: {temp_path}")

    val_prob_matrix = np.column_stack([
        apply_temperature(val_preds["bert_logits"], temps["bert"]),
        apply_temperature(val_preds["roberta_logits"], temps["roberta"]),
        apply_temperature(val_preds["longformer_logits"], temps["longformer"]),
    ])

    # Tối ưu trọng số Weighted Soft Voting trên xác suất đã hiệu chuẩn
    detector.fit_weights(val_prob_matrix, val_labels)

    # Huấn luyện Stacking Meta-Classifier trên xác suất đã hiệu chuẩn
    detector.fit_stacking(val_prob_matrix, val_labels)

    def _eval_split(split_title: str, texts_list: List[str], labels_arr: np.ndarray):
        print("\n" + "-" * 60)
        print(f"  ĐÁNH GIÁ TRÊN: {split_title} ({len(texts_list):,} mẫu)")
        print("-" * 60)
        sp_preds = detector.extract_predictions(texts_list, batch_size=args.batch_size)
        pb = apply_temperature(sp_preds["bert_logits"], temps["bert"])
        pr = apply_temperature(sp_preds["roberta_logits"], temps["roberta"])
        pl = apply_temperature(sp_preds["longformer_logits"], temps["longformer"])
        ps = detector.predict_ensemble(pb, pr, pl, method="simple")
        pw = detector.predict_ensemble(pb, pr, pl, method="weighted")
        pk = detector.predict_ensemble(pb, pr, pl, method="stacking")
        rows_tbl, metrics_map = [], {}
        models_eval = [
            ("BERT (bert-base-uncased)", pb),
            ("RoBERTa (roberta-base)", pr),
            ("Longformer (longformer-base-4096)", pl),
            ("Ensemble (Simple Soft Voting)", ps),
            ("Ensemble (Weighted Soft Voting)", pw),
            ("Ensemble (Stacking Meta-Classifier)", pk),
        ]
        print("=" * 95)
        print(f"{'MÔ HÌNH / PHƯƠNG PHÁP':<38} | {'ACCURACY':<10} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10} | {'AUC-ROC':<10}")
        print("=" * 95)
        for name, probs in models_eval:
            m = compute_metrics(labels_arr, probs)
            metrics_map[name] = m
            rows_tbl.append({
                "model": name,
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "auc": m["auc"],
            })
            print(f"{name:<38} | {m['accuracy']*100:>9.2f}% | {m['precision']*100:>9.2f}% | {m['recall']*100:>9.2f}% | {m['f1']*100:>9.2f}% | {m['auc']*100:>9.2f}%")
        print("=" * 95)
        return rows_tbl, metrics_map, (pb, pr, pl, ps, pw, pk)

    # 4. Đánh giá trên tập Test chính (test.csv = In-Domain 2.873 mẫu)
    results_table, all_metrics, (p_bert, p_roberta, p_longformer, p_ens_simple, p_ens_weighted, p_ens_stacking) = _eval_split(
        f"TẬP TEST CHÍNH ({os.path.basename(test_csv)})", test_texts, test_labels
    )

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

    # 7. Đánh giá bổ sung trên Multi-Domain Test (english_v2_test.csv) nếu tồn tại & khác test_csv
    v2_test_csv = os.path.join(args.data_dir, "english_v2_test.csv")
    if os.path.exists(v2_test_csv) and os.path.abspath(v2_test_csv) != os.path.abspath(test_csv):
        df_v2t = pd.read_csv(v2_test_csv)
        v2_rows, v2_metrics, _ = _eval_split(
            "TẬP TEST ĐA MIỀN (english_v2_test.csv)",
            df_v2t["text"].astype(str).tolist(),
            df_v2t["label"].values,
        )
        with open(os.path.join(args.output_dir, "english_v2_test_results.json"), "w", encoding="utf-8") as f:
            json.dump(v2_metrics, f, indent=2, ensure_ascii=False)

    # 8. Đánh giá ngoại miền độc lập trên TOEFL (252 bài luận chưa từng thấy)
    toefl_csv = _resolve_csv(args.toefl_file, "toefl_dataset.csv")
    if os.path.exists(toefl_csv):
        df_toefl = pd.read_csv(toefl_csv)
        t_rows, t_metrics, (tb, tr, tl, ts, tw, tk) = _eval_split(
            "TẬP NGOẠI MIỀN ĐỘC LẬP TOEFL (252 ESSAYS — ZERO-SHOT)",
            df_toefl["text"].astype(str).tolist(),
            df_toefl["label"].values,
        )
        pd.DataFrame(t_rows).to_csv(os.path.join(args.output_dir, "toefl_comparison_table.csv"), index=False)
        pd.DataFrame({
            "essay_id": df_toefl["essay_id"] if "essay_id" in df_toefl.columns else range(len(df_toefl)),
            "label": df_toefl["label"].values,
            "bert_prob": tb,
            "roberta_prob": tr,
            "longformer_prob": tl,
            "ensemble_simple_prob": ts,
            "ensemble_weighted_prob": tw,
            "ensemble_stacking_prob": tk,
        }).to_csv(os.path.join(args.output_dir, "toefl_predictions.csv"), index=False)

        shift = {}
        for m_name, tm in t_metrics.items():
            im = all_metrics.get(m_name)
            if im:
                shift[m_name] = {
                    "in_domain_accuracy": float(im["accuracy"]),
                    "toefl_accuracy": float(tm["accuracy"]),
                    "delta_accuracy": float(tm["accuracy"] - im["accuracy"]),
                    "in_domain_f1": float(im["f1"]),
                    "toefl_f1": float(tm["f1"]),
                    "delta_f1": float(tm["f1"] - im["f1"]),
                }
        with open(os.path.join(args.output_dir, "toefl_domain_shift_analysis.json"), "w", encoding="utf-8") as f:
            json.dump(shift, f, indent=2, ensure_ascii=False)
        print(f"[*] Đã lưu kết quả ngoại miền TOEFL & phân tích Domain Shift.")


def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình Ensemble PLMs + Temperature Scaling")
    parser.add_argument("--data_dir", type=str, default="data", help="Thư mục chứa val.csv, test.csv")
    parser.add_argument("--val_file", type=str, default=None, help="File CSV validation (vd: english_v2_val.csv)")
    parser.add_argument("--test_file", type=str, default=None, help="File CSV test chính (mặc định: test.csv)")
    parser.add_argument("--toefl_file", type=str, default=None, help="File CSV TOEFL ngoại miền")
    parser.add_argument("--output_dir", type=str, default="output", help="Thư mục lưu kết quả")
    parser.add_argument("--bert_path", type=str, default="output/bert_best.pt", help="Đường dẫn bert_best.pt")
    parser.add_argument("--roberta_path", type=str, default="output/roberta_best.pt", help="Đường dẫn roberta_best.pt")
    parser.add_argument("--longformer_path", type=str, default="output/longformer_best.pt", help="Đường dẫn longformer_best.pt")
    parser.add_argument("--batch_size", type=int, default=16, help="Kích thước batch khi trích xuất dự đoán")
    parser.add_argument("--max_samples", type=int, default=None, help="Giới hạn số mẫu kiểm thử nhanh")
    parser.add_argument("--fp16", action="store_true", default=True, help="Sử dụng FP16 tăng tốc độ")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ensemble_evaluation(args)

