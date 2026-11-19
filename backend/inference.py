"""
Module Suy Luận & Phân Tích Diễn Giải Văn Bản AI (Inference Engine) - Hỗ trợ Song ngữ Anh - Việt
Đồ án Chuyên ngành: Phát hiện văn bản do LLM sinh sử dụng Mô hình kết hợp (Ensemble PLMs & PhoBERT)

Tính năng chính:
1. Dynamic Language Routing:
   - Tự động nhận diện Tiếng Việt ('vi') bằng tập ký tự dấu đặc trưng kết hợp fallback langdetect (< 0.05ms).
   - Tuyến Tiếng Việt ('vi'): Sử dụng mô hình tiền huấn luyện chuyên biệt PhoBERT (vinai/phobert-base-v2) kết hợp tách từ pyvi.
   - Tuyến Tiếng Anh ('en'): Sử dụng bộ ba Ensemble PLMs (BERT, RoBERTa, Longformer 1024 tokens) với Weighted Soft Voting và Stacking Meta-Learner.
2. Phân tích diễn giải cấp câu (Sentence-Level Explainability / AI Hotspot Detection):
   - Bóc tách văn bản thành từng câu độc lập.
   - Tính toán xác suất AI cho từng câu, gán nhãn màu cảnh báo (🟢 Green / 🟡 Yellow / 🔴 Red).
3. Đánh giá mức độ tin cậy (Confidence Level) & Phân tích chi tiết từng mô hình.
4. Giao diện dòng lệnh đa năng (CLI): Hỗ trợ nhập trực tiếp, đọc từ file, chế độ tương tác (interactive) và đo đạc độ trễ (latency benchmark).
"""

import os
import sys
import re
import json
import time
import argparse
from typing import Dict, Any, List, Optional, Tuple, Union

import torch
import numpy as np
import joblib
from transformers import AutoTokenizer

from models import build_model


# Tập ký tự dấu đặc trưng Tiếng Việt (chữ thường và chữ hoa)
VIETNAMESE_DIACRITICS = set(
    "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ"
)


def detect_language(text: str) -> str:
    """
    Tự động nhận diện ngôn ngữ của văn bản đầu vào:
    1. Quét ký tự dấu đặc trưng Tiếng Việt (siêu tốc < 0.05ms, độ chính xác cao).
    2. Fallback sử dụng thư viện `langdetect` nếu văn bản không dấu.
    Trả về 'vi' (Tiếng Việt) hoặc 'en' (Tiếng Anh/mặc định).
    """
    clean = text.strip()
    if not clean:
        return "en"

    # Bước 1: Quét ký tự dấu Tiếng Việt
    if any(ch in VIETNAMESE_DIACRITICS for ch in clean):
        return "vi"

    # Bước 2: Quét fallback bằng langdetect cho văn bản không dấu hoặc ngắn
    try:
        import langdetect
        sample = clean[:300]
        detected = langdetect.detect(sample)
        if detected == "vi":
            return "vi"
    except Exception:
        pass

    return "en"


class AITextDetector:
    """
    Hệ thống phát hiện văn bản do AI sinh hỗ trợ Đa ngôn ngữ (Anh - Việt):
    - Tiếng Anh: Ensemble PLMs (BERT + RoBERTa + Longformer).
    - Tiếng Việt: PhoBERT base v2 + PyVi Tokenizer.
    """

    def __init__(
        self,
        models_dir: str = "output",
        device: Optional[Union[str, torch.device]] = None,
        use_fp16: bool = True,
        load_bert: bool = True,
        load_roberta: bool = True,
        load_longformer: bool = True,
        load_phobert: bool = True,
    ):
        # Tự động phân giải đường dẫn models_dir linh hoạt
        if not os.path.isabs(models_dir):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            if os.path.exists(models_dir):
                self.models_dir = os.path.abspath(models_dir)
            elif os.path.exists(os.path.join(base_dir, models_dir)):
                self.models_dir = os.path.join(base_dir, models_dir)
            elif os.path.exists(os.path.join(base_dir, "..", models_dir)):
                self.models_dir = os.path.abspath(os.path.join(base_dir, "..", models_dir))
            else:
                self.models_dir = models_dir
        else:
            self.models_dir = models_dir

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.use_fp16 = use_fp16 and (self.device.type == "cuda")

        if self.device.type == "cpu":
            num_threads = min(os.cpu_count() or 4, 8)
            torch.set_num_threads(num_threads)

        print("=" * 70)
        print("  KHỞI TẠO BỘ SUY LUẬN AI TEXT DETECTOR (SONG NGỮ EN / VI)")
        print("=" * 70)
        print(f"[*] Thiết bị thực thi : {self.device}")
        if self.device.type == "cuda":
            print(f"    - Tên GPU          : {torch.cuda.get_device_name(0)}")
            print(f"    - Mixed Precision  : {self.use_fp16}")

        # 1. Nạp Tokenizers Tiếng Anh
        print("[*] Đang nạp Tokenizers Tiếng Anh...")
        self.tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer_roberta = AutoTokenizer.from_pretrained("roberta-base")
        self.tokenizer_longformer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

        # 2. Nạp Tokenizer PhoBERT (Tiếng Việt)
        print("[*] Đang nạp Tokenizer PhoBERT Tiếng Việt...")
        self.tokenizer_phobert = None
        phobert_tok_dir = os.path.join(self.models_dir, "tokenizer_phobert")
        if os.path.exists(phobert_tok_dir):
            try:
                self.tokenizer_phobert = AutoTokenizer.from_pretrained(phobert_tok_dir)
            except Exception:
                self.tokenizer_phobert = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        else:
            try:
                self.tokenizer_phobert = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
            except Exception as e:
                print(f"[!] Không thể nạp Tokenizer PhoBERT: {e}")

        # 3. Nạp mô hình BERT (Tiếng Anh)
        self.model_bert = None
        bert_path = os.path.join(self.models_dir, "bert_best.pt")
        if load_bert and os.path.exists(bert_path):
            print(f"[*] Nạp checkpoint BERT từ: {bert_path}")
            self.model_bert = build_model("bert").to(self.device)
            ckpt = torch.load(bert_path, map_location=self.device, weights_only=False)
            state_dict = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt
            self.model_bert.load_state_dict(state_dict)
            self.model_bert.eval()
            print("  [v] BERT sẵn sàng.")

        # 4. Nạp mô hình RoBERTa (Tiếng Anh)
        self.model_roberta = None
        roberta_path = os.path.join(self.models_dir, "roberta_best.pt")
        if load_roberta and os.path.exists(roberta_path):
            print(f"[*] Nạp checkpoint RoBERTa từ: {roberta_path}")
            self.model_roberta = build_model("roberta").to(self.device)
            ckpt = torch.load(roberta_path, map_location=self.device, weights_only=False)
            state_dict = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt
            self.model_roberta.load_state_dict(state_dict)
            self.model_roberta.eval()
            print("  [v] RoBERTa sẵn sàng.")

        # 5. Nạp mô hình Longformer (Tiếng Anh 1024 tokens)
        self.model_longformer = None
        longformer_path = os.path.join(self.models_dir, "longformer_best.pt")
        if load_longformer and os.path.exists(longformer_path):
            print(f"[*] Nạp checkpoint Longformer từ: {longformer_path}")
            self.model_longformer = build_model("longformer").to(self.device)
            ckpt = torch.load(longformer_path, map_location=self.device, weights_only=False)
            state_dict = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt
            self.model_longformer.load_state_dict(state_dict)
            self.model_longformer.eval()
            print("  [v] Longformer sẵn sàng.")

        # 6. Nạp mô hình PhoBERT (Tiếng Việt)
        self.model_phobert = None
        phobert_path = os.path.join(self.models_dir, "phobert_best.pt")
        if load_phobert and os.path.exists(phobert_path):
            print(f"[*] Nạp checkpoint PhoBERT từ: {phobert_path}")
            try:
                self.model_phobert = build_model("phobert").to(self.device)
                ckpt = torch.load(phobert_path, map_location=self.device, weights_only=False)
                state_dict = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt
                self.model_phobert.load_state_dict(state_dict)
                self.model_phobert.eval()
                print("  [v] PhoBERT sẵn sàng (Chuyên trách phát hiện văn bản Tiếng Việt).")
            except Exception as e:
                print(f"[!] Lỗi nạp mô hình PhoBERT: {e}")
                self.model_phobert = None

        # 7. Nạp cấu hình trọng số Ensemble tối ưu Tiếng Anh
        weights_file = os.path.join(self.models_dir, "ensemble_weights.json")
        if os.path.exists(weights_file):
            try:
                with open(weights_file, "r", encoding="utf-8") as f:
                    w_dict = json.load(f)["weights"]
                self.weights = np.array([
                    float(w_dict.get("bert", 1.0 / 3.0)),
                    float(w_dict.get("roberta", 1.0 / 3.0)),
                    float(w_dict.get("longformer", 1.0 / 3.0)),
                ])
                print(f"[*] Trọng số Ensemble tối ưu (EN): BERT={self.weights[0]:.4f}, RoBERTa={self.weights[1]:.4f}, Longformer={self.weights[2]:.4f}")
            except Exception as e:
                print(f"[!] Lỗi đọc {weights_file}: {e}, dùng trọng số đều (1/3).")
                self.weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        else:
            self.weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

        # 8. Nạp mô hình Stacking Meta-Learner Tiếng Anh
        self.meta_classifier = None
        stacking_file = os.path.join(self.models_dir, "stacking_meta_learner.joblib")
        if os.path.exists(stacking_file):
            try:
                self.meta_classifier = joblib.load(stacking_file)
                print("  [v] Stacking Meta-Classifier (Logistic Regression) sẵn sàng.")
            except Exception as e:
                print(f"[!] Không nạp được mô hình Stacking: {e}")

        print("=" * 70 + "\n")

    def _predict_bert(self, texts: List[str]) -> np.ndarray:
        """Dự đoán xác suất theo batch cho BERT (max 512 tokens)."""
        encoded = self.tokenizer_bert(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model_bert(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                token_type_ids=encoded.get("token_type_ids"),
            )
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs

    def _predict_roberta(self, texts: List[str]) -> np.ndarray:
        """Dự đoán xác suất theo batch cho RoBERTa (max 512 tokens)."""
        encoded = self.tokenizer_roberta(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model_roberta(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs

    def _predict_longformer(self, texts: List[str]) -> np.ndarray:
        """Dự đoán xác suất theo batch cho Longformer (max 1024 tokens + global attention)."""
        encoded = self.tokenizer_longformer(
            texts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(self.device)

        global_attention_mask = torch.zeros_like(encoded["input_ids"])
        global_attention_mask[:, 0] = 1

        with torch.no_grad():
            logits = self.model_longformer(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                global_attention_mask=global_attention_mask,
            )
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs

    def _predict_phobert(self, texts: List[str]) -> np.ndarray:
        """Dự đoán xác suất theo batch cho PhoBERT Tiếng Việt (max 256 tokens)."""
        if self.model_phobert is None or self.tokenizer_phobert is None:
            return np.full(len(texts), 0.5)

        from pyvi import ViTokenizer
        segmented_texts = [ViTokenizer.tokenize(t) for t in texts]

        encoded = self.tokenizer_phobert(
            segmented_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model_phobert(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs

    def predict(
        self,
        text: str,
        threshold: float = 0.5,
        detailed: bool = True,
        force_language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Dự đoán một đoạn văn bản thô (Tự động định tuyến ngôn ngữ):
        - Tiếng Việt -> PhoBERT
        - Tiếng Anh -> Ensemble PLMs (Weighted Soft Voting)
        """
        text = text.strip()
        if not text:
            return {"error": "Văn bản rỗng (Empty text)."}

        words = text.split()
        word_count = len(words)
        char_count = len(text)

        # 1. Định tuyến ngôn ngữ
        lang = force_language if force_language else detect_language(text)

        t0 = time.time()
        batch_text = [text]

        # ======================================================================
        # NHÁNH TIẾNG VIỆT (PHOBERT)
        # ======================================================================
        if lang == "vi" and self.model_phobert is not None:
            from pyvi import ViTokenizer
            seg_text = ViTokenizer.tokenize(text)
            token_count = len(self.tokenizer_phobert.encode(seg_text, add_special_tokens=False, truncation=False))

            p_phobert = float(self._predict_phobert(batch_text)[0])
            final_ai_prob = p_phobert
            is_ai = final_ai_prob >= threshold

            label = "AI-generated" if is_ai else "Human-written"
            label_vi = "Do AI sinh (ChatGPT/LLM)" if is_ai else "Do người viết (Human)"

            # Độ tin cậy
            diff = abs(final_ai_prob - threshold)
            if diff >= 0.35:
                confidence = "Rất cao (Very High)"
                confidence_level = "VERY_HIGH"
            elif diff >= 0.20:
                confidence = "Cao (High)"
                confidence_level = "HIGH"
            elif diff >= 0.10:
                confidence = "Trung bình (Moderate)"
                confidence_level = "MODERATE"
            else:
                confidence = "Không chắc chắn / Ngưỡng ranh giới (Uncertain)"
                confidence_level = "LOW"

            inference_time_ms = (time.time() - t0) * 1000

            result = {
                "label": label,
                "label_vi": label_vi,
                "ai_probability": round(final_ai_prob, 4),
                "human_probability": round(1.0 - final_ai_prob, 4),
                "ai_percentage": f"{final_ai_prob * 100:.2f}%",
                "confidence": confidence,
                "confidence_level": confidence_level,
                "threshold_used": threshold,
                "language": "vi",
                "language_label": "Tiếng Việt 🇻🇳",
                "detected_model": "PhoBERT (vinai/phobert-base-v2)",
                "text_stats": {
                    "word_count": word_count,
                    "char_count": char_count,
                    "token_count": token_count,
                    "context_coverage": "Full Context (256)" if token_count <= 256 else "Truncated (First 256 tokens)",
                },
                "inference_time_ms": round(inference_time_ms, 2),
            }

            if detailed:
                result["model_breakdown"] = {
                    "ensemble_weighted": {
                        "probability": round(p_phobert, 4),
                        "is_champion": True,
                    },
                    "ensemble_simple": {
                        "probability": round(p_phobert, 4),
                    },
                    "ensemble_stacking": {
                        "probability": round(p_phobert, 4),
                    },
                    "individual_models": {
                        "phobert": {
                            "probability": round(p_phobert, 4),
                            "weight": 1.0,
                            "weight_pct": "100%",
                            "model_name": "PhoBERT base v2",
                        },
                        # Giữ các key bert, roberta, longformer tương thích CSDL và UI cũ
                        "bert": {
                            "probability": round(p_phobert, 4),
                            "weight": 0.0,
                            "weight_pct": "N/A (Tiếng Việt)",
                        },
                        "roberta": {
                            "probability": round(p_phobert, 4),
                            "weight": 0.0,
                            "weight_pct": "N/A (Tiếng Việt)",
                        },
                        "longformer": {
                            "probability": round(p_phobert, 4),
                            "weight": 0.0,
                            "weight_pct": "N/A (Tiếng Việt)",
                        },
                    },
                }

            return result

        # ======================================================================
        # NHÁNH TIẾNG ANH (ENSEMBLE PLMs)
        # ======================================================================
        token_count = len(self.tokenizer_roberta.encode(text, add_special_tokens=False, truncation=False))

        # 1. Suy luận từ các mô hình thành phần
        p_bert = float(self._predict_bert(batch_text)[0]) if self.model_bert else 0.5
        p_roberta = float(self._predict_roberta(batch_text)[0]) if self.model_roberta else 0.5
        p_longformer = float(self._predict_longformer(batch_text)[0]) if self.model_longformer else 0.5

        # 2. Tính toán điểm số Ensemble
        p_weighted = float(
            self.weights[0] * p_bert +
            self.weights[1] * p_roberta +
            self.weights[2] * p_longformer
        )
        p_simple = float((p_bert + p_roberta + p_longformer) / 3.0)

        if self.meta_classifier:
            X = np.array([[p_bert, p_roberta, p_longformer]])
            p_stacking = float(self.meta_classifier.predict_proba(X)[0, 1])
        else:
            p_stacking = p_weighted

        inference_time_ms = (time.time() - t0) * 1000

        # 3. Quyết định phân loại nhị phân
        final_ai_prob = p_weighted
        is_ai = final_ai_prob >= threshold

        label = "AI-generated" if is_ai else "Human-written"
        label_vi = "Do AI sinh (ChatGPT/LLM)" if is_ai else "Do người viết (Human)"

        # 4. Xác định mức độ tin cậy
        diff = abs(final_ai_prob - threshold)
        if diff >= 0.40:
            confidence = "Rất cao (Very High)"
            confidence_level = "VERY_HIGH"
        elif diff >= 0.25:
            confidence = "Cao (High)"
            confidence_level = "HIGH"
        elif diff >= 0.10:
            confidence = "Trung bình (Moderate)"
            confidence_level = "MODERATE"
        else:
            confidence = "Không chắc chắn / Ngưỡng ranh giới (Uncertain)"
            confidence_level = "LOW"

        result = {
            "label": label,
            "label_vi": label_vi,
            "ai_probability": round(final_ai_prob, 4),
            "human_probability": round(1.0 - final_ai_prob, 4),
            "ai_percentage": f"{final_ai_prob * 100:.2f}%",
            "confidence": confidence,
            "confidence_level": confidence_level,
            "threshold_used": threshold,
            "language": "en",
            "language_label": "Tiếng Anh 🇬🇧",
            "detected_model": "Ensemble PLMs (Longformer + BERT + RoBERTa)",
            "text_stats": {
                "word_count": word_count,
                "char_count": char_count,
                "token_count": token_count,
                "context_coverage": "Full Context (1024)" if token_count <= 1024 else "Truncated (First 1024 tokens)",
            },
            "inference_time_ms": round(inference_time_ms, 2),
        }

        if detailed:
            result["model_breakdown"] = {
                "ensemble_weighted": {
                    "probability": round(p_weighted, 4),
                    "is_champion": True,
                },
                "ensemble_simple": {
                    "probability": round(p_simple, 4),
                },
                "ensemble_stacking": {
                    "probability": round(p_stacking, 4),
                },
                "individual_models": {
                    "bert": {
                        "probability": round(p_bert, 4),
                        "weight": round(float(self.weights[0]), 4),
                        "weight_pct": f"{self.weights[0]*100:.1f}%",
                    },
                    "roberta": {
                        "probability": round(p_roberta, 4),
                        "weight": round(float(self.weights[1]), 4),
                        "weight_pct": f"{self.weights[1]*100:.1f}%",
                    },
                    "longformer": {
                        "probability": round(p_longformer, 4),
                        "weight": round(float(self.weights[2]), 4),
                        "weight_pct": f"{self.weights[2]*100:.1f}%",
                    },
                },
            }

        return result

    def analyze_sentences(
        self,
        text: str,
        threshold: float = 0.5,
        max_sentences: int = 40,
        force_language: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Bóc tách văn bản thành từng câu độc lập và tính toán xác suất AI theo Batch nhanh chóng.
        Phục vụ tính năng Highlight màu (AI Hotspots) cho Chrome Extension và Web UI.
        Hỗ trợ cả Tiếng Việt (PhoBERT) và Tiếng Anh (Ensemble PLMs).
        """
        raw_sentences = re.split(r'(?<=[.!?…\n])\s+', text.strip())
        sentences = [s.strip() for s in raw_sentences if len(s.strip().split()) >= 2]

        if not sentences:
            sentences = [text.strip()]

        eval_sentences = sentences[:max_sentences]
        lang = force_language if force_language else detect_language(text)

        # 1. Nếu là Tiếng Việt và PhoBERT đã nạp
        if lang == "vi" and self.model_phobert is not None:
            probs = self._predict_phobert(eval_sentences)
            sentence_results = []
            for idx, sentence in enumerate(eval_sentences):
                prob = float(probs[idx])
                if prob >= 0.80:
                    color = "red"
                    level = "Chắc chắn AI"
                elif prob >= 0.50:
                    color = "yellow"
                    level = "Nghi vấn AI"
                else:
                    color = "green"
                    level = "Văn phong tự nhiên (Người)"

                sentence_results.append({
                    "sentence_index": idx + 1,
                    "sentence": sentence,
                    "word_count": len(sentence.split()),
                    "ai_probability": round(prob, 4),
                    "ai_percentage": f"{prob * 100:.1f}%",
                    "highlight_color": color,
                    "assessment": level,
                })
            return sentence_results

        # 2. Nếu là Tiếng Anh: Tối ưu hóa Batch Inference
        p_bert = self._predict_bert(eval_sentences) if self.model_bert else np.full(len(eval_sentences), 0.5)
        p_roberta = self._predict_roberta(eval_sentences) if self.model_roberta else np.full(len(eval_sentences), 0.5)

        if self.device.type == "cuda" and self.model_longformer:
            p_longformer = self._predict_longformer(eval_sentences)
            p_weighted = (
                self.weights[0] * p_bert +
                self.weights[1] * p_roberta +
                self.weights[2] * p_longformer
            )
        else:
            w_sum = self.weights[0] + self.weights[1]
            w_bert = self.weights[0] / w_sum
            w_roberta = self.weights[1] / w_sum
            p_weighted = w_bert * p_bert + w_roberta * p_roberta

        sentence_results = []
        for idx, sentence in enumerate(eval_sentences):
            prob = float(p_weighted[idx])
            if prob >= 0.80:
                color = "red"
                level = "Chắc chắn AI"
            elif prob >= 0.50:
                color = "yellow"
                level = "Nghi vấn AI"
            else:
                color = "green"
                level = "Văn phong tự nhiên (Người)"

            sentence_results.append({
                "sentence_index": idx + 1,
                "sentence": sentence,
                "word_count": len(sentence.split()),
                "ai_probability": round(prob, 4),
                "ai_percentage": f"{prob * 100:.1f}%",
                "highlight_color": color,
                "assessment": level,
            })

        return sentence_results

    def benchmark_latency(self, num_runs: int = 5) -> Dict[str, Any]:
        """Đo đạc độ trễ suy luận chuẩn khoa học trên cả tiếng Anh và tiếng Việt."""
        samples = {
            "short_text": "This is a quick breaking news update regarding the latest entertainment awards ceremony. The actors gathered on the red carpet in Hollywood to celebrate their recent achievements.",
            "medium_text": (
                "Artificial intelligence has transformed numerous fields in recent years, particularly in education and software development. "
                "Students and professionals now utilize large language models to assist with drafting essays, writing computer code, and brainstorming creative ideas. "
                "While these advanced technologies offer immense productivity gains, they also raise critical concerns regarding academic integrity, intellectual property, and original thought. "
                "Consequently, educators and researchers have focused extensively on building dependable detection mechanisms to distinguish between synthetic machine outputs and authentic human authorship."
            ),
            "vietnamese_text": (
                "Trí tuệ nhân tạo đang tạo ra những bước chuyển biến to lớn trong nhiều lĩnh vực của đời sống xã hội. "
                "Từ việc hỗ trợ tra cứu thông tin, soạn thảo văn bản cho tới phân tích dữ liệu chuyên sâu, các mô hình ngôn ngữ lớn đã chứng minh được tính ưu việt của mình. "
                "Tuy nhiên, việc lạm dụng công nghệ cũng đặt ra nhiều thách thức mới đối với tính trung thực và bản quyền tác giả trong môi trường giáo dục."
            ),
        }

        benchmarks = {}
        print("\n[*] Đang thực hiện Benchmark đo độ trễ suy luận (Inference Latency)...")

        for sample_type, sample_text in samples.items():
            words = len(sample_text.split())
            lang = detect_language(sample_text)
            tok = self.tokenizer_phobert if (lang == "vi" and self.tokenizer_phobert) else self.tokenizer_roberta
            tokens = len(tok.tokenize(sample_text))

            latencies = []
            # Warm-up 1 lần
            self.predict(sample_text, detailed=False)

            for _ in range(num_runs):
                t0 = time.time()
                self.predict(sample_text, detailed=False)
                latencies.append((time.time() - t0) * 1000)

            mean_lat = float(np.mean(latencies))
            std_lat = float(np.std(latencies))

            benchmarks[sample_type] = {
                "language": lang,
                "word_count": words,
                "token_count": tokens,
                "mean_latency_ms": round(mean_lat, 2),
                "std_latency_ms": round(std_lat, 2),
                "throughput_tokens_per_sec": round((tokens / (mean_lat / 1000)), 1),
            }
            print(f"  - {sample_type:<15} [{lang.upper()}] ({words:>3} words, {tokens:>3} tokens) : {mean_lat:>6.2f} ms ± {std_lat:.2f} ms ({benchmarks[sample_type]['throughput_tokens_per_sec']} tokens/s)")

        return benchmarks


def print_prediction_report(res: Dict[str, Any], sentences: Optional[List[Dict[str, Any]]] = None):
    """In báo cáo kết quả dự đoán trực quan và đẹp mắt ra Terminal."""
    print("\n" + "=" * 70)
    print("  KẾT QUẢ PHÂN TÍCH PHÁT HIỆN VĂN BẢN AI (AI TEXT DETECTION REPORT)")
    print("=" * 70)

    color_mark = "[AI DETECTED]" if "AI" in res["label"] else "[HUMAN WRITTEN]"
    print(f"[*] NGÔN NGỮ ĐỊNH TUYẾN: {res.get('language_label', 'Tiếng Anh')} ({res.get('detected_model', 'Ensemble')})")
    print(f"[*] KẾT LUẬN CHÍNH     : {res['label_vi']} {color_mark}")
    print(f"[*] XÁC SUẤT AI SINH   : {res['ai_percentage']} (Xác suất Người: {res['human_probability']*100:.2f}%)")
    print(f"[*] ĐỘ TIN CẬY         : {res['confidence']}")
    print(f"[*] ĐỘ TRỄ SUY LUẬN    : {res['inference_time_ms']} ms")
    print(f"[*] THỐNG KÊ VĂN BẢN   : {res['text_stats']['word_count']} từ | {res['text_stats']['token_count']} tokens ({res['text_stats']['context_coverage']})")

    if "model_breakdown" in res:
        print("\n" + "-" * 70)
        print("  ĐIỂM SỐ ĐÓNG GÓP TỪNG MÔ HÌNH THÀNH PHẦN (BREAKDOWN)")
        print("-" * 70)
        models_data = res["model_breakdown"]["individual_models"]
        if "phobert" in models_data:
            print(f"  • PhoBERT (Base v2)        : {models_data['phobert']['probability']*100:>6.2f}% AI | Trọng số: {models_data['phobert']['weight_pct']} [Chuyên trách VN]")
        else:
            print(f"  • BERT (512 tokens)        : {models_data['bert']['probability']*100:>6.2f}% AI | Trọng số: {models_data['bert']['weight_pct']}")
            print(f"  • RoBERTa (512 tokens)     : {models_data['roberta']['probability']*100:>6.2f}% AI | Trọng số: {models_data['roberta']['weight_pct']}")
            print(f"  • Longformer (1024 tokens) : {models_data['longformer']['probability']*100:>6.2f}% AI | Trọng số: {models_data['longformer']['weight_pct']} [Trụ cột]")
            print(f"  -------------------------------------------------------------")
            print(f"  • Ensemble (Weighted Soft) : {res['model_breakdown']['ensemble_weighted']['probability']*100:>6.2f}% AI [Mô hình chính]")
            print(f"  • Ensemble (Simple Soft)   : {res['model_breakdown']['ensemble_simple']['probability']*100:>6.2f}% AI")
            print(f"  • Ensemble (Stacking LR)   : {res['model_breakdown']['ensemble_stacking']['probability']*100:>6.2f}% AI")

    if sentences:
        print("\n" + "-" * 70)
        print("  PHÂN TÍCH DIỄN GIẢI TỪNG CÂU (SENTENCE-LEVEL AI HOTSPOTS)")
        print("-" * 70)
        for s in sentences:
            mark = "[🔴]" if s["highlight_color"] == "red" else "[🟡]" if s["highlight_color"] == "yellow" else "[🟢]"
            print(f"  {mark} Câu {s['sentence_index']:>2} ({s['ai_percentage']:>6} AI - {s['assessment']}):")
            print(f"     \"{s['sentence']}\"")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="AI Text Detection Inference Engine (Ensemble PLMs & PhoBERT)")
    parser.add_argument("--text", type=str, default=None, help="Đoạn văn bản cần kiểm tra")
    parser.add_argument("--file", type=str, default=None, help="Đường dẫn file .txt chứa văn bản")
    parser.add_argument("--models_dir", type=str, default="output", help="Thư mục chứa checkpoints .pt")
    parser.add_argument("--threshold", type=float, default=0.5, help="Ngưỡng phân loại nhị phân")
    parser.add_argument("--sentences", action="store_true", help="Bật phân tích chi tiết từng câu")
    parser.add_argument("--interactive", action="store_true", help="Chế độ CLI tương tác nhập trực tiếp")
    parser.add_argument("--benchmark", action="store_true", help="Chạy benchmark đo độ trễ suy luận")
    args = parser.parse_args()

    detector = AITextDetector(models_dir=args.models_dir)

    if args.benchmark:
        benchmarks = detector.benchmark_latency(num_runs=5)
        bench_file = os.path.join(args.models_dir, "inference_latency_benchmark.json")
        with open(bench_file, "w", encoding="utf-8") as f:
            json.dump(benchmarks, f, indent=2, ensure_ascii=False)
        print(f"\n[v] Đã lưu kết quả benchmark vào: {bench_file}")
        return

    if args.interactive:
        print("\n[*] ĐÃ BẬT CHẾ ĐỘ TƯƠNG TÁC (INTERACTIVE MODE). Gõ 'exit' hoặc 'quit' để thoát.")
        while True:
            try:
                user_input = input("\n[Nhập văn bản cần kiểm tra]: ").strip()
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("[*] Tạm biệt!")
                    break
                if not user_input:
                    continue
                res = detector.predict(user_input, threshold=args.threshold)
                sentences = detector.analyze_sentences(user_input, threshold=args.threshold) if args.sentences else None
                print_prediction_report(res, sentences)
            except KeyboardInterrupt:
                print("\n[*] Đã thoát.")
                break
        return

    target_text = None
    if args.text:
        target_text = args.text
    elif args.file:
        if os.path.exists(args.file):
            with open(args.file, "r", encoding="utf-8") as f:
                target_text = f.read()
        else:
            print(f"[!] Không tìm thấy file: {args.file}")
            sys.exit(1)

    if target_text:
        res = detector.predict(target_text, threshold=args.threshold)
        sentences = detector.analyze_sentences(target_text, threshold=args.threshold) if args.sentences else None
        print_prediction_report(res, sentences)
    else:
        print("\n[*] Không có văn bản đầu vào, chạy thử nghiệm mẫu với 1 bài văn Tiếng Việt:")
        demo_text = (
            "Trí tuệ nhân tạo ngày nay đang đóng vai trò then chốt trong sự phát triển của công nghệ toàn cầu. "
            "Các mô hình xử lý ngôn ngữ tự nhiên hiện đại có khả năng tiếp nhận và xử lý lượng thông tin khổng lồ một cách ấn tượng. "
            "Tuy nhiên, sự xuất hiện ngày càng nhiều của các nội dung do máy sinh đòi hỏi các giải pháp kiểm định nghiêm ngặt và đáng tin cậy."
        )
        res = detector.predict(demo_text, threshold=args.threshold)
        sentences = detector.analyze_sentences(demo_text, threshold=args.threshold)
        print_prediction_report(res, sentences)


if __name__ == "__main__":
    main()
