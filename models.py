"""
Module định nghĩa kiến trúc các mô hình phân loại văn bản dựa trên Pre-trained Language Models (PLMs).
Phục vụ Đồ án Chuyên ngành (Chủ đề 6): Phát hiện văn bản do LLM sinh sử dụng mô hình kết hợp (Ensemble PLMs).

Bao gồm:
1. BERTTextClassifier: Mô hình dựa trên bert-base-uncased (Tối ưu cho văn bản ngắn).
2. RoBERTATextClassifier: Mô hình dựa trên roberta-base (Tối ưu cho ngữ pháp & văn phong).
3. LongformerTextClassifier: Mô hình dựa trên allenai/longformer-base-4096 (Tối ưu cho bài báo dài).
4. build_model(): Factory function khởi tạo mô hình theo định danh chuẩn hóa.
"""

import os
import sys
from typing import Optional, Dict, Any

# Đảm bảo UTF-8 trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class BERTTextClassifier(nn.Module):
    """
    Mô hình phát hiện văn bản do AI sinh dựa trên BERT (bert-base-uncased).
    Kiến trúc: BERT Backbone -> Dropout -> Linear Classification Head (Logit nhị phân).
    """

    def __init__(
        self,
        pretrained_name: str = "bert-base-uncased",
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.pretrained_name = pretrained_name
        self.config = AutoConfig.from_pretrained(pretrained_name)
        self.bert = AutoModel.from_pretrained(pretrained_name, config=self.config)

        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, 1)

        # Khởi tạo trọng số cho tầng phân loại theo chuẩn Xavier
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass của BERT.
        :param input_ids: Tensor tensor (batch_size, seq_len)
        :param attention_mask: Tensor (batch_size, seq_len)
        :param token_type_ids: Tensor (batch_size, seq_len) đặc thù của BERT
        :return: logits dạng raw tensor shape (batch_size,) cho BCEWithLogitsLoss
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Trích xuất biểu diễn của token đầu tiên [CLS]
        cls_output = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
        pooled = self.dropout(cls_output)
        logits = self.classifier(pooled).squeeze(-1)      # Shape: (batch_size,)
        return logits


class RoBERTATextClassifier(nn.Module):
    """
    Mô hình phát hiện văn bản do AI sinh dựa trên RoBERTa (roberta-base).
    Lưu ý: RoBERTa không sử dụng token_type_ids.
    """

    def __init__(
        self,
        pretrained_name: str = "roberta-base",
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.pretrained_name = pretrained_name
        self.config = AutoConfig.from_pretrained(pretrained_name)
        self.roberta = AutoModel.from_pretrained(pretrained_name, config=self.config)

        if freeze_backbone:
            for param in self.roberta.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, 1)

        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_output = outputs.last_hidden_state[:, 0, :]  # Token <s>
        pooled = self.dropout(cls_output)
        logits = self.classifier(pooled).squeeze(-1)
        return logits


class LongformerTextClassifier(nn.Module):
    """
    Mô hình phát hiện văn bản do AI sinh dựa trên Longformer (allenai/longformer-base-4096).
    Đặc thù: Hỗ trợ văn bản dài tới 4096 tokens với cơ chế Sliding Window Attention.
    """

    def __init__(
        self,
        pretrained_name: str = "allenai/longformer-base-4096",
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.pretrained_name = pretrained_name
        self.config = AutoConfig.from_pretrained(pretrained_name)
        self.longformer = AutoModel.from_pretrained(pretrained_name, config=self.config)

        if freeze_backbone:
            for param in self.longformer.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, 1)

        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(cls_output)
        logits = self.classifier(pooled).squeeze(-1)
        return logits


def build_model(
    model_type: str = "bert",
    pretrained_name: Optional[str] = None,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Factory function khởi tạo mô hình chuẩn hóa cho Ensemble Pipeline.
    :param model_type: 'bert', 'roberta', hoặc 'longformer'
    :param pretrained_name: Tên checkpoint Hugging Face (nếu None sẽ lấy mặc định)
    :param dropout_rate: Tỷ lệ Dropout cho head phân loại
    :param freeze_backbone: Có đóng băng tầng backbone hay không
    :return: Thực thể nn.Module tương ứng
    """
    model_type = model_type.lower().strip()
    if model_type == "bert":
        name = pretrained_name or "bert-base-uncased"
        return BERTTextClassifier(pretrained_name=name, dropout_rate=dropout_rate, freeze_backbone=freeze_backbone)
    elif model_type == "roberta":
        name = pretrained_name or "roberta-base"
        return RoBERTATextClassifier(pretrained_name=name, dropout_rate=dropout_rate, freeze_backbone=freeze_backbone)
    elif model_type == "longformer":
        name = pretrained_name or "allenai/longformer-base-4096"
        return LongformerTextClassifier(pretrained_name=name, dropout_rate=dropout_rate, freeze_backbone=freeze_backbone)
    else:
        raise ValueError(f"Loại mô hình '{model_type}' không được hỗ trợ. Chọn trong: ['bert', 'roberta', 'longformer']")


if __name__ == "__main__":
    print("=" * 70)
    print("  KIỂM THỬ KHỞI TẠO KIẾN TRÚC MÔ HÌNH (MODELS.PY)")
    print("=" * 70)

    # 1. Khởi tạo mô hình BERT
    print("\n[*] Đang khởi tạo BERTTextClassifier...")
    bert_model = build_model(model_type="bert", dropout_rate=0.3)
    num_params = sum(p.numel() for p in bert_model.parameters())
    num_trainable = sum(p.numel() for p in bert_model.parameters() if p.requires_grad)
    print(f"  [v] Khởi tạo thành công!")
    print(f"  - Tổng số tham số        : {num_params:,}")
    print(f"  - Tham số có thể train   : {num_trainable:,}")

    # 2. Dummy forward test
    batch_size, seq_len = 2, 16
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    dummy_attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    dummy_token_types = torch.zeros((batch_size, seq_len), dtype=torch.long)

    with torch.no_grad():
        out = bert_model(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            token_type_ids=dummy_token_types,
        )

    print(f"\n[*] Kiểm tra Dummy Forward Pass:")
    print(f"  - Input Shape : {dummy_input_ids.shape}")
    print(f"  - Output Shape: {out.shape} (Kỳ vọng: ({batch_size},))")
    print(f"  - Output Value: {out.tolist()}")
    assert out.shape == (batch_size,), f"Lỗi shape đầu ra: {out.shape} khác ({batch_size},)"
    print("  [v] Forward pass chính xác 100%!")
    print("=" * 70)

