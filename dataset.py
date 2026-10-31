import os
import sys
import argparse
from typing import Dict, List, Optional, Tuple, Union

# Đảm bảo in tiếng Việt / emoji trên console Windows không bị lỗi
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Tắt cảnh báo không cần thiết từ Hugging Face
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Định danh cấu hình cho 3 backbone PLMs trong Ensemble
MODEL_CONFIGS = {
    "bert": {
        "pretrained_name": "bert-base-uncased",
        "default_max_length": 512,
        "needs_token_type_ids": True,
        "needs_global_attention": False,
    },
    "roberta": {
        "pretrained_name": "roberta-base",
        "default_max_length": 512,
        "needs_token_type_ids": False,
        "needs_global_attention": False,
    },
    "longformer": {
        "pretrained_name": "allenai/longformer-base-4096",
        "default_max_length": 1024,  # Ngưỡng tối ưu bao phủ 92.2% dữ liệu tin dài
        "needs_token_type_ids": False,
        "needs_global_attention": True,
    },
}

class TextDetectionDataset(Dataset):
    """
    Lớp Dataset tùy chỉnh cho bài toán phát hiện văn bản do AI sinh (Binary Classification).
    Tương thích chuẩn hóa với cả 3 mô hình ngôn ngữ tiền huấn luyện (BERT, RoBERTa, Longformer).
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[Union[int, float]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        model_type: str = "bert",
    ):
        """
        Khởi tạo Dataset.
        :param texts: Danh sách các văn bản đầu vào.
        :param labels: Danh sách nhãn (0: Người viết, 1: AI sinh).
        :param tokenizer: Tokenizer tương ứng của mô hình.
        :param max_length: Chiều dài tối đa của chuỗi token.
        :param model_type: 'bert', 'roberta' hoặc 'longformer'.
        """
        assert len(texts) == len(labels), "Số lượng văn bản và nhãn phải bằng nhau!"
        self.texts = [str(t) if pd.notna(t) else "" for t in texts]
        self.labels = [float(lbl) for lbl in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_type = model_type.lower()
        self.config = MODEL_CONFIGS.get(self.model_type, MODEL_CONFIGS["bert"])

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize không đệm (padding=False) để áp dụng Dynamic Padding ở Collator
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
            return_token_type_ids=self.config["needs_token_type_ids"],
            add_special_tokens=True,
        )

        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoding["attention_mask"], dtype=torch.long)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.float32),
        }

        # Bổ sung token_type_ids cho BERT nếu cần
        if self.config["needs_token_type_ids"] and "token_type_ids" in encoding:
            item["token_type_ids"] = torch.tensor(encoding["token_type_ids"], dtype=torch.long)

        # Bổ sung global_attention_mask đặc thù cho Longformer
        # Token đầu tiên (<s> / [CLS]) được gán giá trị 1 để chú ý toàn cục
        if self.config["needs_global_attention"]:
            global_attention_mask = torch.zeros_like(input_ids)
            if len(global_attention_mask) > 0:
                global_attention_mask[0] = 1
            item["global_attention_mask"] = global_attention_mask

        return item

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        model_type: str = "bert",
        text_column: str = "text",
        label_column: str = "label",
    ) -> "TextDetectionDataset":
        """Hàm tiện ích nạp Dataset trực tiếp từ file CSV."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {csv_path}")
        df = pd.read_csv(csv_path)
        return cls(
            texts=df[text_column].tolist(),
            labels=df[label_column].tolist(),
            tokenizer=tokenizer,
            max_length=max_length,
            model_type=model_type,
        )


class DynamicPaddingCollator:
    """
    Collate function thực hiện Dynamic Padding theo kích thước batch thực tế.
    Giúp tiết kiệm 40% - 60% VRAM/RAM so với fixed max padding tĩnh.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, pad_to_multiple_of: Optional[int] = 8):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Xác định độ dài lớn nhất trong batch
        max_len = max(len(item["input_ids"]) for item in batch)
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 0:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        batch_input_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []
        batch_global_attention = []
        batch_labels = []

        has_token_types = "token_type_ids" in batch[0]
        has_global_attn = "global_attention_mask" in batch[0]

        for item in batch:
            input_id = item["input_ids"]
            attn_mask = item["attention_mask"]
            curr_len = len(input_id)
            pad_len = max_len - curr_len

            # Đệm input_ids và attention_mask
            if pad_len > 0:
                padded_input_id = torch.cat([input_id, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
                padded_attn_mask = torch.cat([attn_mask, torch.zeros((pad_len,), dtype=torch.long)])
            else:
                padded_input_id = input_id[:max_len]
                padded_attn_mask = attn_mask[:max_len]

            batch_input_ids.append(padded_input_id)
            batch_attention_mask.append(padded_attn_mask)
            batch_labels.append(item["labels"])

            # Đệm token_type_ids nếu có
            if has_token_types:
                token_type = item["token_type_ids"]
                if pad_len > 0:
                    padded_type = torch.cat([token_type, torch.zeros((pad_len,), dtype=torch.long)])
                else:
                    padded_type = token_type[:max_len]
                batch_token_type_ids.append(padded_type)

            # Đệm global_attention_mask nếu có (Longformer)
            if has_global_attn:
                global_attn = item["global_attention_mask"]
                if pad_len > 0:
                    padded_global = torch.cat([global_attn, torch.zeros((pad_len,), dtype=torch.long)])
                else:
                    padded_global = global_attn[:max_len]
                batch_global_attention.append(padded_global)

        result = {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
        }

        if has_token_types:
            result["token_type_ids"] = torch.stack(batch_token_type_ids)
        if has_global_attn:
            result["global_attention_mask"] = torch.stack(batch_global_attention)

        return result


def get_dataloaders(
    model_type: str = "bert",
    data_dir: str = "data",
    batch_size: int = 16,
    max_length: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, PreTrainedTokenizerBase]:
    """
    Factory function chuẩn hóa tạo bộ 3 DataLoader (Train, Val, Test In-domain).
    """
    model_type = model_type.lower()
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Không hỗ trợ mô hình: '{model_type}'. Chọn trong: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_type]
    actual_max_length = max_length or config["default_max_length"]

    # Khởi tạo Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["pretrained_name"])

    # Đường dẫn các tập dữ liệu
    train_csv = os.path.join(data_dir, "train.csv")
    val_csv = os.path.join(data_dir, "val.csv")
    test_csv = os.path.join(data_dir, "test.csv")

    for path, name in [(train_csv, "Train"), (val_csv, "Val"), (test_csv, "Test")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy file tập {name}: {path}. Vui lòng chạy split_data.py trước!")

    # Khởi tạo Datasets
    train_dataset = TextDetectionDataset.from_csv(
        train_csv, tokenizer=tokenizer, max_length=actual_max_length, model_type=model_type
    )
    val_dataset = TextDetectionDataset.from_csv(
        val_csv, tokenizer=tokenizer, max_length=actual_max_length, model_type=model_type
    )
    test_dataset = TextDetectionDataset.from_csv(
        test_csv, tokenizer=tokenizer, max_length=actual_max_length, model_type=model_type
    )

    # Dynamic Padding Collator
    collator = DynamicPaddingCollator(tokenizer=tokenizer, pad_to_multiple_of=8)

    # Cấu hình phần cứng
    use_pin_memory = pin_memory if pin_memory is not None else torch.cuda.is_available()

    # Khởi tạo DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader, test_loader, tokenizer


def get_unseen_test_loader(
    model_type: str = "bert",
    data_path: str = "data/toefl_dataset.csv",
    batch_size: int = 16,
    max_length: Optional[int] = None,
) -> Tuple[DataLoader, PreTrainedTokenizerBase]:
    """Hàm tạo DataLoader cho tập kiểm thử miền mới Unseen Domain (TOEFL)."""
    model_type = model_type.lower()
    config = MODEL_CONFIGS[model_type]
    actual_max_length = max_length or config["default_max_length"]
    tokenizer = AutoTokenizer.from_pretrained(config["pretrained_name"])

    dataset = TextDetectionDataset.from_csv(
        data_path, tokenizer=tokenizer, max_length=actual_max_length, model_type=model_type
    )
    collator = DynamicPaddingCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    return loader, tokenizer


def verify_pipeline(model_name: str, batch_size: int = 4):
    """Hàm kiểm thử tích hợp (Smoke Test & Tensor Verification) cho từng mô hình."""
    print("\n" + "=" * 75)
    print(f"[*] KIỂM THỬ XÁC THỰC PIPELINE CHO BACKBONE: {model_name.upper()}")
    print("=" * 75)

    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        model_type=model_name,
        batch_size=batch_size,
    )

    print(f"  [v] Khởi tạo Tokenizer: {tokenizer.__class__.__name__}")
    print(f"  [v] Số lượng batch: Train={len(train_loader):,}, Val={len(val_loader):,}, Test={len(test_loader):,}")

    # Rút trích 1 batch mẫu từ tập Train
    batch = next(iter(train_loader))

    print("\n  🔍 CẤU TRÚC VÀ KÍCH THƯỚC TENSOR BATCH ĐẦU TIÊN:")
    for k, v in batch.items():
        print(f"     - {k:<22}: shape = {str(list(v.shape)):<18} | dtype = {v.dtype}")

    # Kiểm tra tính toàn vẹn
    assert not torch.isnan(batch["labels"]).any(), "Phát hiện giá trị NaN trong nhãn!"
    assert batch["input_ids"].shape[0] == batch_size, "Batch size không khớp cấu hình!"
    assert (
        batch["attention_mask"].shape == batch["input_ids"].shape
    ), "Attention mask và input_ids không cùng kích thước!"

    # Kiểm tra giải mã ngược token đầu câu
    decoded = tokenizer.decode(batch["input_ids"][0][:25], skip_special_tokens=False)
    print(f"\n  📝 Trích xuất 25 token đầu của mẫu đầu tiên:")
    print(f"     \"{decoded}...\"")

    # Kiểm tra tương thích với hàm mất mát BCEWithLogitsLoss
    dummy_logits = torch.randn(batch_size)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_val = loss_fn(dummy_logits, batch["labels"])
    print(f"\n  🎯 Kiểm tra tương thích BCEWithLogitsLoss: loss = {loss_val.item():.4f} (OK [✓])")

    # Kiểm tra nhanh tập kiểm thử TOEFL
    toefl_loader, _ = get_unseen_test_loader(model_type=model_name, batch_size=batch_size)
    toefl_batch = next(iter(toefl_loader))
    print(f"  🎯 Kiểm tra TOEFL Unseen Loader: {len(toefl_loader)} batches | Shape = {list(toefl_batch['input_ids'].shape)} (OK [✓])")

    print(f"\n  ✅ Đạt chuẩn 100%! Mô hình {model_name.upper()} sẵn sàng nạp huấn luyện ở Ngày 8.")


def main():
    parser = argparse.ArgumentParser(description="PyTorch Dataset & DataLoader Pipeline cho Ensemble PLMs (Ngày 7)")
    parser.add_argument("--model", type=str, default="all", choices=["all", "bert", "roberta", "longformer"], help="Mô hình cần kiểm thử (mặc định: all)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size kiểm thử (mặc định: 8)")
    args = parser.parse_args()

    print("=" * 80)
    print("  NGÀY 7: THIẾT LẬP PYTORCH DATASET & DATALOADER CHO ENSEMBLE PLMS")
    print("=" * 80)

    models_to_test = ["bert", "roberta", "longformer"] if args.model == "all" else [args.model]

    for m in models_to_test:
        verify_pipeline(m, batch_size=args.batch_size)

    print("\n" + "=" * 80)
    print("🎉 HOÀN TẤT CÔNG VIỆC NGÀY 7 & CHỐT TOÀN DIỆN TIẾN ĐỘ TUẦN 1 (GIAI ĐOẠN 1)!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
