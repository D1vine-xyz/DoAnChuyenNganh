"""
Pipeline Huấn Luyện Mô Hình Phân Loại AI Tiếng Việt: PhoBERT (vinai/phobert-base-v2)
Phục vụ Đồ án Chuyên ngành (Chủ đề 6): Phát hiện văn bản do LLM sinh (VeriText AI PRO).
- Tách từ trước (pre-tokenize) bằng pyvi và lưu cache RAM -> tăng tốc 3x trên GPU.
- Hỗ trợ Mixed Precision FP16 (CUDA AMP) & Label Smoothing (mặc định 0.05) chống overconfidence.
- Đánh giá đa chiều: Accuracy, Precision, Recall, F1-Score, ROC-AUC + Breakdown chi tiết từng thể loại (genre).
- Tự động lưu checkpoint tối ưu (phobert_best.pt) và báo cáo đánh giá trên tập Test độc lập.
"""

import os
import sys
import time
import json
import random
import argparse
from typing import Dict, Any, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from pyvi import ViTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from models import build_model


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class VietnameseTextDataset(Dataset):
    """Dataset tiếng Việt tách từ sẵn bằng PyVi & mã hóa 1 lần để tối ưu tốc độ huấn luyện."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        raw_texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(float).tolist()
        self.genres = df["genre"].astype(str).tolist() if "genre" in df.columns else ["unknown"] * len(raw_texts)
        self.encoded_items: List[Dict[str, Any]] = []

        t0 = time.time()
        for idx, raw_text in enumerate(raw_texts):
            seg_text = ViTokenizer.tokenize(raw_text)
            encoding = tokenizer(
                seg_text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )
            self.encoded_items.append({
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "label": self.labels[idx],
            })
        print(f"    -> Đã tiền xử lý PyVi + PhoBERT Tokenizer cho {len(raw_texts):,} mẫu trong {time.time() - t0:.1f}s")

    def __len__(self):
        return len(self.encoded_items)

    def __getitem__(self, idx):
        return self.encoded_items[idx]


def collate_fn(batch, pad_token_id=1):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        labels.append(item["label"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.float),
    }


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Tính Expected Calibration Error (ECE) đo độ trung thực của xác suất dự đoán."""
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


def fit_temperature_scaling(model, dataloader, device, use_fp16: bool = False) -> Dict[str, float]:
    """Hiệu chuẩn xác suất hậu kiểm (Post-hoc Temperature Scaling) tối ưu NLL trên tập Validation."""
    from scipy.optimize import minimize_scalar

    model.eval()
    all_logits, all_labels = [], []
    with torch.inference_mode():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.amp.autocast("cuda", enabled=use_fp16):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(logits.float().cpu().numpy())
            all_labels.append(batch["labels"].numpy())

    logits_arr = np.concatenate(all_logits)
    labels_arr = np.concatenate(all_labels).astype(np.float64)

    def _nll(temp: float) -> float:
        scaled = np.clip(logits_arr / temp, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-scaled))
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        return float(-np.mean(labels_arr * np.log(p) + (1.0 - labels_arr) * np.log(1.0 - p)))

    nll_before = _nll(1.0)
    probs_before = 1.0 / (1.0 + np.exp(-np.clip(logits_arr, -30.0, 30.0)))
    ece_before = compute_ece(probs_before, labels_arr)

    opt = minimize_scalar(_nll, bounds=(1.0, 4.5), method="bounded")
    best_temp = float(opt.x)
    nll_after = _nll(best_temp)
    probs_after = 1.0 / (1.0 + np.exp(-np.clip(logits_arr / best_temp, -30.0, 30.0)))
    ece_after = compute_ece(probs_after, labels_arr)

    return {
        "temperature": round(best_temp, 4),
        "val_nll_before": round(nll_before, 5),
        "val_nll_after": round(nll_after, 5),
        "val_ece_before": round(ece_before, 5),
        "val_ece_after": round(ece_after, 5),
    }


def evaluate(model, dataloader, criterion, device, genres: List[str] = None, use_fp16: bool = False, temperature: float = 1.0):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.inference_mode():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=use_fp16):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                scaled_logits = logits / max(float(temperature), 0.1)
                loss = criterion(scaled_logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(scaled_logits).float().cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().astype(int).tolist())

    avg_loss = total_loss / max(len(dataloader), 1)
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5

    result: Dict[str, Any] = {
        "loss": float(avg_loss),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
    }

    if genres and len(genres) == len(all_labels):
        per_genre = {}
        df_eval = pd.DataFrame({
            "genre": genres,
            "label": all_labels,
            "pred": all_preds,
            "prob": all_probs,
        })
        for g, grp in df_eval.groupby("genre"):
            g_acc = float((grp["label"] == grp["pred"]).mean())
            per_genre[g] = {
                "count": int(len(grp)),
                "true_label": int(grp["label"].iloc[0]),
                "accuracy": round(g_acc, 4),
                "mean_ai_prob": round(float(grp["prob"].mean()), 4),
            }
        result["per_genre_metrics"] = per_genre

    return result


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình PhoBERT Tiếng Việt (VeriText AI PRO)")
    parser.add_argument("--train_file", type=str, default="data/vietnamese_train.csv", help="Đường dẫn file train CSV")
    parser.add_argument("--val_file", type=str, default="data/vietnamese_val.csv", help="Đường dẫn file val CSV")
    parser.add_argument("--test_file", type=str, default="data/vietnamese_test.csv", help="Đường dẫn file test CSV")
    parser.add_argument("--epochs", type=int, default=4, help="Số epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=16, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=256, help="Độ dài tối đa token")
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="Label smoothing (0.0 - 0.1) chống overconfidence")
    args = parser.parse_args()

    set_seed(42)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    train_file = args.train_file if os.path.isabs(args.train_file) else os.path.join(base_dir, args.train_file)
    val_file = args.val_file if os.path.isabs(args.val_file) else os.path.join(base_dir, args.val_file)
    test_file = args.test_file if os.path.isabs(args.test_file) else os.path.join(base_dir, args.test_file)

    print("=" * 75)
    print("  HUẤN LUYỆN MÔ HÌNH PHOBERT TIẾNG VIỆT (vinai/phobert-base-v2)")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda"
    print(f"[*] Thiết bị thực thi        : {device} (FP16 AMP: {use_fp16})")
    print(f"[*] Siêu tham số             : Epochs={args.epochs}, Batch={args.batch_size}, LR={args.lr}, MaxLen={args.max_length}, LabelSmooth={args.label_smoothing}")

    # 1. Nạp dữ liệu
    print(f"[*] Đang đọc dữ liệu từ:\n    - Train: {train_file}\n    - Val  : {val_file}\n    - Test : {test_file}")
    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)
    df_test = pd.read_csv(test_file)

    print(f"  - Tập Train : {len(df_train):,} mẫu | Tập Val: {len(df_val):,} mẫu | Tập Test: {len(df_test):,} mẫu")

    # 2. Khởi tạo Tokenizer & Pre-tokenize
    model_name = "vinai/phobert-base-v2"
    print(f"[*] Đang tải Tokenizer PhoBERT: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = VietnameseTextDataset(df_train, tokenizer, max_length=args.max_length)
    val_ds = VietnameseTextDataset(df_val, tokenizer, max_length=args.max_length)
    test_ds = VietnameseTextDataset(df_test, tokenizer, max_length=args.max_length)

    pad_id = tokenizer.pad_token_id or 1
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_id))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_id))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_id))

    # 3. Xây dựng mô hình
    print("[*] Đang nạp mô hình PhoBERT backbone...")
    model = build_model(model_type="phobert", pretrained_name=model_name, dropout_rate=0.3)
    model.to(device)

    # 4. Optimizer, Cosine Scheduler & AMP Scaler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    best_val_f1 = -1.0
    best_checkpoint_path = os.path.join(output_dir, "phobert_best.pt")
    history = []

    print("\n" + "=" * 75)
    print("  BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN (FINE-TUNING + LABEL SMOOTHING)")
    print("=" * 75)

    eps = max(0.0, min(0.15, float(args.label_smoothing)))

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t_epoch_start = time.time()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            raw_labels = batch["labels"].to(device)
            smooth_labels = raw_labels * (1.0 - eps) + 0.5 * eps

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_fp16):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, smooth_labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()

            if step % 50 == 0 or step == len(train_loader):
                print(f"  [Epoch {epoch}/{args.epochs}] Step {step}/{len(train_loader)} | Batch Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        epoch_time = time.time() - t_epoch_start

        # Đánh giá trên Validation (trên nhãn chuẩn 0/1)
        val_metrics = evaluate(model, val_loader, criterion, device, use_fp16=use_fp16)
        print(f"\n>> [EPOCH {epoch} KẾT QUẢ] ({epoch_time:.1f}s)")
        print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"   Accuracy  : {val_metrics['accuracy']*100:.2f}% | Precision: {val_metrics['precision']*100:.2f}%")
        print(f"   Recall    : {val_metrics['recall']*100:.2f}% | F1-Score : {val_metrics['f1']*100:.2f}% | AUC: {val_metrics['auc']:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 5),
            "val_loss": round(val_metrics["loss"], 5),
            "val_accuracy": round(val_metrics["accuracy"], 5),
            "val_precision": round(val_metrics["precision"], 5),
            "val_recall": round(val_metrics["recall"], 5),
            "val_f1": round(val_metrics["f1"], 5),
            "val_auc": round(val_metrics["auc"], 5),
            "duration_sec": round(epoch_time, 2),
        })

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"   🌟 Checkpoint mới tốt nhất! Đã lưu: {best_checkpoint_path}")

        print("-" * 75)

    # 5. Lưu log & tokenizer
    log_file = os.path.join(output_dir, "training_log_phobert.csv")
    pd.DataFrame(history).to_csv(log_file, index=False)

    tokenizer_dir = os.path.join(output_dir, "tokenizer_phobert")
    tokenizer.save_pretrained(tokenizer_dir)
    print(f"[v] Đã lưu Tokenizer PhoBERT vào: {tokenizer_dir}")

    # 6. HIỆU CHUẨN XÁC SUẤT (TEMPERATURE SCALING) & ĐÁNH GIÁ TẬP TEST ĐỘC LẬP
    print("\n" + "=" * 75)
    print(f"  HIỆU CHUẨN TEMPERATURE SCALING & ĐÁNH GIÁ TEST ({os.path.basename(test_file)})")
    print("=" * 75)

    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device, weights_only=True))
    calib = fit_temperature_scaling(model, val_loader, device, use_fp16=use_fp16)
    best_temp = calib["temperature"]
    print(f"  [*] Optimal Temperature T* = {best_temp:.4f}")
    print(f"      - Val NLL : {calib['val_nll_before']:.5f} -> {calib['val_nll_after']:.5f}")
    print(f"      - Val ECE : {calib['val_ece_before']:.5f} -> {calib['val_ece_after']:.5f}")

    temp_path = os.path.join(output_dir, "phobert_temperature.json")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)
    print(f"  [v] Đã lưu thông số hiệu chuẩn vào: {temp_path}")

    test_metrics = evaluate(model, test_loader, criterion, device, genres=test_ds.genres, use_fp16=use_fp16, temperature=best_temp)
    test_metrics["calibration"] = calib

    print(f"  - Accuracy  : {test_metrics['accuracy']*100:.2f}%")
    print(f"  - Precision : {test_metrics['precision']*100:.2f}%")
    print(f"  - Recall    : {test_metrics['recall']*100:.2f}%")
    print(f"  - F1-Score  : {test_metrics['f1']*100:.2f}%")
    print(f"  - AUC-ROC   : {test_metrics['auc']:.4f}")

    if "per_genre_metrics" in test_metrics:
        print("\n  --- ĐỘ CHÍNH XÁC CHI TIẾT TỪNG THỂ LOẠI (PER-GENRE BREAKDOWN) ---")
        for g, st in sorted(test_metrics["per_genre_metrics"].items()):
            lbl_str = "Human" if st["true_label"] == 0 else "AI   "
            print(f"    • [{lbl_str}] {g:<24} (n={st['count']:>3}): Acc={st['accuracy']*100:>6.2f}% | Mean AI Prob={st['mean_ai_prob']*100:>5.1f}%")

    test_result_path = os.path.join(output_dir, "phobert_test_results.json")
    with open(test_result_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    print("=" * 75)
    print(f"🎉 HOÀN TẤT HUẤN LUYỆN PHOBERT! Checkpoint: {best_checkpoint_path}\n")


if __name__ == "__main__":
    main()
