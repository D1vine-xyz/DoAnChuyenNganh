"""
Pipeline Huấn Luyện Mô Hình Phân Loại AI Tiếng Việt: PhoBERT (vinai/phobert-base-v2)
Phục vụ Đồ án Chuyên ngành (Chủ đề 6): Phát hiện văn bản do LLM sinh (Ensemble PLMs).
- Tách từ bằng pyvi trước khi đưa vào PhoBERT Tokenizer (BPE 64.000 tokens).
- Tinh chỉnh phân loại nhị phân (0: Người viết, 1: AI sinh).
- Đánh giá đa chiều: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
- Tự động lưu checkpoint tối ưu (phobert_best.pt) và đánh giá trên tập Test độc lập.
"""

import os
import sys
import time
import json
import random
import argparse
from typing import Dict, Any, Tuple, Optional, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
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
    """Dataset cho tiếng Việt có tích hợp tự động phân đoạn từ (pyvi)."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.texts = df["text"].tolist()
        self.labels = df["label"].astype(float).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        raw_text = str(self.texts[idx])
        # Tách từ tiếng Việt thành các cụm có dấu gạch dưới (vd: trí_tuệ nhân_tạo)
        seg_text = ViTokenizer.tokenize(raw_text)
        label = self.labels[idx]

        encoding = self.tokenizer(
            seg_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # Sẽ đệm động trong collate_fn
            return_tensors=None,
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "label": label,
        }


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


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().astype(int))

    avg_loss = total_loss / max(len(dataloader), 1)
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
    }


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình PhoBERT Tiếng Việt")
    parser.add_argument("--epochs", type=int, default=3, help="Số epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=16, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=256, help="Độ dài tối đa token")
    args = parser.parse_args()

    set_seed(42)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    train_file = os.path.join(data_dir, "vietnamese_train.csv")
    val_file = os.path.join(data_dir, "vietnamese_val.csv")
    test_file = os.path.join(data_dir, "vietnamese_test.csv")

    print("=" * 70)
    print("  GIAI ĐOẠN 2: HUẤN LUYỆN MÔ HÌNH PHOBERT TIẾNG VIỆT (vinai/phobert-base-v2)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị thực thi        : {device}")
    print(f"[*] Cấu hình siêu tham số    : Epochs={args.epochs}, Batch={args.batch_size}, LR={args.lr}, MaxLen={args.max_length}")

    # 1. Nạp dữ liệu
    print("[*] Đang đọc dữ liệu tiếng Việt...")
    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)
    df_test = pd.read_csv(test_file)

    print(f"  - Tập Train : {len(df_train)} mẫu")
    print(f"  - Tập Val   : {len(df_val)} mẫu")
    print(f"  - Tập Test  : {len(df_test)} mẫu")

    # 2. Khởi tạo Tokenizer
    model_name = "vinai/phobert-base-v2"
    print(f"[*] Đang tải Tokenizer PhoBERT: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = VietnameseTextDataset(df_train, tokenizer, max_length=args.max_length)
    val_ds = VietnameseTextDataset(df_val, tokenizer, max_length=args.max_length)
    test_ds = VietnameseTextDataset(df_test, tokenizer, max_length=args.max_length)

    pad_id = tokenizer.pad_token_id or 1
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_id))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_id))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_id))

    # 3. Xây dựng mô hình
    print(f"[*] Đang nạp mô hình PhoBERT backbone...")
    model = build_model(model_type="phobert", pretrained_name=model_name, dropout_rate=0.3)
    model.to(device)

    # 4. Optimizer & Scheduler
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
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = -1.0
    best_checkpoint_path = os.path.join(output_dir, "phobert_best.pt")
    history = []

    print("\n" + "=" * 70)
    print("  BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN (FINE-TUNING)")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t_epoch_start = time.time()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if step % 15 == 0 or step == len(train_loader):
                print(f"  [Epoch {epoch}/{args.epochs}] Step {step}/{len(train_loader)} | Batch Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - t_epoch_start

        # Đánh giá trên Validation
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"\n>> [EPOCH {epoch} KẾT QUẢ] ({epoch_time:.1f}s)")
        print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"   Accuracy  : {val_metrics['accuracy']*100:.2f}% | Precision: {val_metrics['precision']*100:.2f}%")
        print(f"   Recall    : {val_metrics['recall']*100:.2f}% | F1-Score : {val_metrics['f1']*100:.2f}% | AUC: {val_metrics['auc']:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
            "duration_sec": epoch_time,
        })

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"   🌟 Checkpoint mới tốt nhất! Đã lưu: {best_checkpoint_path}")

        print("-" * 70)

    # 5. Lưu log & tokenizer
    log_file = os.path.join(output_dir, "training_log_phobert.csv")
    pd.DataFrame(history).to_csv(log_file, index=False)

    tokenizer_dir = os.path.join(output_dir, "tokenizer_phobert")
    tokenizer.save_pretrained(tokenizer_dir)
    print(f"[v] Đã lưu Tokenizer PhoBERT vào: {tokenizer_dir}")

    # 6. ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP
    print("\n" + "=" * 70)
    print("  ĐÁNH GIÁ TRÊN TẬP TEST TIẾNG VIỆT ĐỘC LẬP (vietnamese_test.csv)")
    print("=" * 70)

    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device, weights_only=True))
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"  - Accuracy  : {test_metrics['accuracy']*100:.2f}%")
    print(f"  - Precision : {test_metrics['precision']*100:.2f}%")
    print(f"  - Recall    : {test_metrics['recall']*100:.2f}%")
    print(f"  - F1-Score  : {test_metrics['f1']*100:.2f}%")
    print(f"  - AUC-ROC   : {test_metrics['auc']:.4f}")

    test_result_path = os.path.join(output_dir, "phobert_test_results.json")
    with open(test_result_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print("=" * 70)
    print(f"🎉 GIAI ĐOẠN 2 HOÀN TẤT! Checkpoint đã sẵn sàng tại: {best_checkpoint_path}\n")


if __name__ == "__main__":
    main()
