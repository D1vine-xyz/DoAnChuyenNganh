"""
Pipeline Huấn luyện Mô hình Phân loại Longformer (allenai/longformer-base-4096)
Phục vụ Đồ án Chuyên ngành (Chủ đề 6): Phát hiện văn bản do LLM sinh sử dụng mô hình kết hợp (Ensemble PLMs).

Mô hình nền tảng thứ ba:
- Backbone: allenai/longformer-base-4096 (12 layers, 768 hidden, 12 attention heads, 149M parameters)
- Cơ chế chú ý: Sliding Window Attention (cục bộ) kết hợp Global Attention (toàn cục tại token <s>).
- Ưu thế cốt lõi: Xử lý ngữ cảnh dài tới 1024 - 4096 tokens, khắc phục triệt để hạn chế cắt ngắn
                 (truncation) của BERT và RoBERTa trên các bài báo dài.
- Tối ưu hóa: Mixed Precision FP16, AdamW (lr=3e-5, weight_decay=0.01 no-decay bias/LayerNorm),
              Gradient Accumulation (tích lũy gradient giữ effective batch size = 16),
              Linear Warmup Scheduler, Gradient Clipping (1.0).
- Checkpoint: longformer_best.pt lưu khi Validation F1-Score cao nhất.
- Đánh giá độc lập trên tập Test nội miền: longformer_test_results.json.
"""

import os
import sys
import time
import json
import math
import random
import argparse
from typing import Dict, Any, Tuple, Optional, List

# Đảm bảo UTF-8 trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# Import từ các module trong cùng dự án
from dataset import get_dataloaders
from models import build_model


def set_seed(seed: int = 42):
    """Cố định seed để đảm bảo tính tái lập (Reproducibility) 100%."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_fp16: bool = False,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Hàm đánh giá mô hình trên tập dữ liệu (Validation hoặc Test).
    Tính toán: Loss, Accuracy, Precision, Recall, F1-Score, AUC-ROC.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for step, batch in enumerate(dataloader, 1):
            if max_batches is not None and step > max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            global_attention_mask = batch.get("global_attention_mask")
            if global_attention_mask is not None:
                global_attention_mask = global_attention_mask.to(device)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                )
                loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            targets = labels.cpu().numpy().astype(int)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())

    num_samples = len(all_targets)
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0

    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="binary", zero_division=0
    )

    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.5

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


def train_longformer(args):
    """Hàm chính điều phối quá trình huấn luyện Longformer."""
    print("=" * 75)
    print("  QUY TRÌNH HUẤN LUYỆN MÔ HÌNH NỀN TẢNG: LONGFORMER (LONGFORMER-BASE-4096)")
    print("=" * 75)

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Xác định thiết bị tính toán
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = args.fp16 and (device.type == "cuda")

    print(f"[*] Thiết bị sử dụng   : {device}")
    if device.type == "cuda":
        print(f"    - Tên GPU          : {torch.cuda.get_device_name(0)}")
        print(f"    - Bộ nhớ GPU       : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"    - Mixed Precision  : {use_fp16} (FP16 Accelerated)")
    else:
        print("    - Chế độ           : CPU (Khuyến nghị dùng GPU trên Kaggle để tối ưu tốc độ)")

    # 2. Khởi tạo DataLoaders cho Longformer
    print("\n[*] Đang nạp dữ liệu từ thư mục:", args.data_dir)
    print(f"[*] Độ dài token tối đa (max_length): {args.max_length}")
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        model_type="longformer",
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(f"  [v] Train Loader : {len(train_loader)} batches ({len(train_loader.dataset):,} mẫu)")
    print(f"  [v] Val Loader   : {len(val_loader)} batches ({len(val_loader.dataset):,} mẫu)")
    print(f"  [v] Test Loader  : {len(test_loader)} batches ({len(test_loader.dataset):,} mẫu)")

    # 3. Khởi tạo Mô hình Longformer
    print("\n[*] Đang khởi tạo kiến trúc LongformerTextClassifier...")
    model = build_model(model_type="longformer", dropout_rate=args.dropout_rate)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [v] Tổng tham số : {total_params:,} (Trainable: {trainable_params:,})")

    # 4. Cấu hình Optimizer & Layer-wise Weight Decay
    # Loại bỏ Weight Decay cho bias và LayerNorm (chuẩn transformer training)
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)

    # 5. Cấu hình Gradient Accumulation & Scheduler & Loss
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    num_training_steps = steps_per_epoch * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    criterion = nn.BCEWithLogitsLoss()

    # Khởi tạo GradScaler cho Mixed Precision FP16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    print(f"\n[*] Thông số Huấn luyện Longformer:")
    print(f"  - Epochs                     : {args.epochs}")
    print(f"  - Micro Batch Size           : {args.batch_size}")
    print(f"  - Grad Accumulation Steps    : {args.gradient_accumulation_steps}")
    print(f"  - Effective Batch Size       : {effective_batch_size} (Đồng bộ với BERT và RoBERTa)")
    print(f"  - Learning Rate              : {args.lr}")
    print(f"  - Weight Decay               : {args.weight_decay}")
    print(f"  - Total Optimizer Steps      : {num_training_steps} (Warmup: {num_warmup_steps} steps - {args.warmup_ratio*100:.0f}%)")
    print(f"  - Max Grad Norm              : {args.max_grad_norm}")
    print(f"  - Thư mục lưu kết quả        : {args.output_dir}")

    # 6. Vòng lặp Huấn luyện chính (Training Loop)
    best_val_f1 = 0.0
    history = []
    best_checkpoint_path = os.path.join(args.output_dir, "longformer_best.pt")

    start_total_time = time.time()
    print("\n" + "=" * 75)
    print("  BẮT ĐẦU HUẤN LUYỆN LONGFORMER (EPOCHS LOOP)")
    print("=" * 75)

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        train_samples = 0

        print(f"\n>>> Epoch {epoch}/{args.epochs}")
        print("-" * 55)

        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            if args.dry_run and step > 4:
                print("  [*] Chế độ Dry Run: Dừng sớm sau 4 batches kiểm thử.")
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            global_attention_mask = batch.get("global_attention_mask")
            if global_attention_mask is not None:
                global_attention_mask = global_attention_mask.to(device)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                )
                loss = criterion(logits, labels)
                # Chia loss theo số bước tích lũy gradient
                loss_scaled = loss / args.gradient_accumulation_steps

            scaler.scale(loss_scaled).backward()

            train_loss += loss.item() * len(labels)
            train_samples += len(labels)

            # Thực hiện cập nhật trọng số sau mỗi gradient_accumulation_steps hoặc batch cuối
            if (step % args.gradient_accumulation_steps == 0) or (step == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            # In tiến độ định kỳ
            if step % args.log_interval == 0 or step == len(train_loader):
                current_lr = scheduler.get_last_lr()[0]
                batch_loss = loss.item()
                elapsed = time.time() - epoch_start_time
                print(
                    f"  Batch [{step:>4}/{len(train_loader)}] | "
                    f"Loss: {batch_loss:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Thời gian: {elapsed:.1f}s"
                )

        avg_train_loss = train_loss / train_samples if train_samples > 0 else 0.0
        epoch_duration = time.time() - epoch_start_time

        # 7. Đánh giá trên tập Validation
        print(f"\n[*] Đang đánh giá Validation Epoch {epoch}...")
        max_eval_batches = 4 if args.dry_run else None
        val_metrics = evaluate(model, val_loader, criterion, device, use_fp16=use_fp16, max_batches=max_eval_batches)

        print(f"  + Train Loss    : {avg_train_loss:.4f}")
        print(f"  + Val Loss      : {val_metrics['loss']:.4f}")
        print(f"  + Val Accuracy  : {val_metrics['accuracy']*100:.2f}%")
        print(f"  + Val Precision : {val_metrics['precision']*100:.2f}%")
        print(f"  + Val Recall    : {val_metrics['recall']*100:.2f}%")
        print(f"  + Val F1-Score  : {val_metrics['f1']*100:.2f}%")
        print(f"  + Val AUC-ROC   : {val_metrics['auc']*100:.2f}%")
        print(f"  + Thời gian     : {epoch_duration:.1f} giây")

        # Ghi nhận lịch sử
        log_entry = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch_duration_sec": epoch_duration,
        }
        history.append(log_entry)

        # 8. Lưu Checkpoint tốt nhất dựa trên Val F1
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            print(f"  🎉 F1-Score cao nhất mới ({best_val_f1*100:.2f}%)! Đang lưu checkpoint...")
            torch.save(
                {
                    "epoch": int(epoch),
                    "model_state_dict": model.state_dict(),
                    "val_f1": float(best_val_f1),
                    "val_auc": float(val_metrics["auc"]),
                    "val_accuracy": float(val_metrics["accuracy"]),
                },
                best_checkpoint_path,
            )
            print(f"  [v] Đã lưu model tại: {best_checkpoint_path}")

    total_training_time = time.time() - start_total_time
    print("\n" + "=" * 75)
    print(f"  HOÀN TẤT HUẤN LUYỆN LONGFORMER TRONG {total_training_time/60:.2f} PHÚT!")
    print(f"  F1-Score tốt nhất trên Validation: {best_val_f1*100:.2f}%")
    print("=" * 75)

    # 9. Lưu file Log Lịch sử huấn luyện (CSV)
    df_history = pd.DataFrame(history)
    csv_log_path = os.path.join(args.output_dir, "training_log_longformer.csv")
    df_history.to_csv(csv_log_path, index=False)
    print(f"[*] Đã lưu lịch sử huấn luyện: {csv_log_path}")

    # Lưu tokenizer để suy luận (inference) khép kín
    tokenizer_dir = os.path.join(args.output_dir, "tokenizer_longformer")
    tokenizer.save_pretrained(tokenizer_dir)
    print(f"[*] Đã lưu Tokenizer tại: {tokenizer_dir}")

    # 10. Đánh giá độc lập trên tập Test (In-domain)
    if os.path.exists(best_checkpoint_path):
        print("\n[*] Đang nạp Checkpoint tốt nhất để đánh giá trên tập Test (In-domain)...")
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate(model, test_loader, criterion, device, use_fp16=use_fp16, max_batches=max_eval_batches)

        print("\n" + "=" * 55)
        print("  KẾT QUẢ ĐÁNH GIÁ LONGFORMER TRÊN TẬP TEST NỘI MIỀN (IN-DOMAIN)")
        print("=" * 55)
        print(f"  - Test Loss      : {test_metrics['loss']:.4f}")
        print(f"  - Test Accuracy  : {test_metrics['accuracy']*100:.2f}%")
        print(f"  - Test Precision : {test_metrics['precision']*100:.2f}%")
        print(f"  - Test Recall    : {test_metrics['recall']*100:.2f}%")
        print(f"  - Test F1-Score  : {test_metrics['f1']*100:.2f}%")
        print(f"  - Test AUC-ROC   : {test_metrics['auc']*100:.2f}%")
        print("=" * 55)

        test_result_path = os.path.join(args.output_dir, "longformer_test_results.json")
        with open(test_result_path, "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2, ensure_ascii=False)
        print(f"[*] Đã lưu kết quả kiểm thử: {test_result_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Longformer phân loại văn bản AI")
    parser.add_argument("--data_dir", type=str, default="data", help="Thư mục chứa train.csv, val.csv, test.csv")
    parser.add_argument("--output_dir", type=str, default="output", help="Thư mục lưu checkpoint và log")
    parser.add_argument("--epochs", type=int, default=3, help="Số epochs huấn luyện")
    parser.add_argument("--batch_size", type=int, default=4, help="Kích thước micro-batch (Khuyến nghị 4 cho Longformer 1024)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Số bước tích lũy gradient (4 x 4 = effective batch 16)")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate (chuẩn Longformer: 3e-5)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay cho AdamW")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Tỷ lệ Warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--max_length", type=int, default=1024, help="Độ dài chuỗi tối đa (1024 bao phủ 92.2% mẫu tin dài)")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Tỷ lệ Dropout")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên")
    parser.add_argument("--log_interval", type=int, default=200, help="Số batch in log tiến độ một lần")
    parser.add_argument("--fp16", action="store_true", default=True, help="Sử dụng Mixed Precision FP16")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Chế độ kiểm thử nhanh 4 batches")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_longformer(args)
