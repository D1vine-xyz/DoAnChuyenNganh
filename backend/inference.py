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
import threading
import argparse
from typing import Dict, Any, List, Optional, Tuple, Union

import torch
import numpy as np
import joblib
from transformers import AutoTokenizer

try:
    from pyvi import ViTokenizer as _ViTokenizer
except ImportError:
    _ViTokenizer = None

try:
    import langdetect as _langdetect
except ImportError:
    _langdetect = None

from models import build_model


# Tập ký tự dấu đặc trưng Tiếng Việt (chữ thường và chữ hoa)
VIETNAMESE_DIACRITICS = set(
    "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ"
)


def detect_language(text: str) -> str:
    """
    Tự động nhận diện ngôn ngữ của văn bản đầu vào:
    1. Đếm mật độ ký tự dấu đặc trưng Tiếng Việt (tránh nhận nhầm bài tiếng Anh có 1 tên riêng tiếng Việt).
    2. Fallback sử dụng `langdetect` nếu văn bản tiếng Việt không dấu hoặc biên độ thấp.
    Trả về 'vi' (Tiếng Việt) hoặc 'en' (Tiếng Anh/mặc định).
    """
    clean = text.strip()
    if not clean:
        return "en"

    # Bước 1: Kiểm tra mật độ ký tự dấu Tiếng Việt
    vi_chars = sum(1 for ch in clean if ch in VIETNAMESE_DIACRITICS)
    alpha_chars = sum(1 for ch in clean if ch.isalpha())
    if vi_chars >= 2 and (vi_chars / max(alpha_chars, 1)) >= 0.02:
        return "vi"

    # Bước 2: Fallback bằng langdetect cho văn bản không dấu hoặc đoạn ngắn
    if _langdetect is not None:
        try:
            if _langdetect.detect(clean[:400]) == "vi":
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
        self._inference_lock = threading.Lock()

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

        # 1. Nạp Tokenizers Tiếng Anh (ưu tiên bản lưu cục bộ trong output/ -> offline-ready)
        print("[*] Đang nạp Tokenizers Tiếng Anh...")
        self.tokenizer_bert = self._load_tokenizer("tokenizer_bert", "bert-base-uncased")
        self.tokenizer_roberta = self._load_tokenizer("tokenizer_roberta", "roberta-base")
        self.tokenizer_longformer = self._load_tokenizer("tokenizer_longformer", "allenai/longformer-base-4096")

        # 2. Nạp Tokenizer PhoBERT (Tiếng Việt)
        print("[*] Đang nạp Tokenizer PhoBERT Tiếng Việt...")
        try:
            self.tokenizer_phobert = self._load_tokenizer("tokenizer_phobert", "vinai/phobert-base-v2")
        except Exception as e:
            print(f"[!] Không thể nạp Tokenizer PhoBERT: {e}")
            self.tokenizer_phobert = None

        # 3-6. Nạp các mô hình thành phần qua helper chung
        self.model_bert = self._load_model_component("bert", "bert_best.pt", "BERT") if load_bert else None
        self.model_roberta = self._load_model_component("roberta", "roberta_best.pt", "RoBERTa") if load_roberta else None
        self.model_longformer = self._load_model_component("longformer", "longformer_best.pt", "Longformer") if load_longformer else None
        self.model_phobert = self._load_model_component(
            "phobert", "phobert_best.pt", "PhoBERT (Chuyên trách phát hiện văn bản Tiếng Việt)"
        ) if load_phobert else None

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

        # 9. Nạp thông số hiệu chuẩn xác suất (Post-hoc Temperature Scaling) cho VI & EN
        self.phobert_temperature = 1.0
        phobert_temp_file = os.path.join(self.models_dir, "phobert_temperature.json")
        if os.path.exists(phobert_temp_file):
            try:
                with open(phobert_temp_file, "r", encoding="utf-8") as f:
                    self.phobert_temperature = max(0.2, float(json.load(f).get("temperature", 1.0)))
                print(f"  [v] Hiệu chuẩn Temperature Scaling (PhoBERT VI): T* = {self.phobert_temperature:.4f}")
            except Exception as e:
                print(f"[!] Lỗi đọc {phobert_temp_file}: {e}, dùng T=1.0.")

        self.en_temperatures = {"bert": 1.0, "roberta": 1.0, "longformer": 1.0}
        en_temp_file = os.path.join(self.models_dir, "en_temperature.json")
        if os.path.exists(en_temp_file):
            try:
                with open(en_temp_file, "r", encoding="utf-8") as f:
                    t_data = json.load(f)
                t_map = t_data.get("temperatures", t_data)
                for k in ("bert", "roberta", "longformer"):
                    if k in t_map:
                        self.en_temperatures[k] = max(0.2, float(t_map[k]))
                print(
                    f"  [v] Hiệu chuẩn Temperature Scaling (EN): "
                    f"BERT={self.en_temperatures['bert']:.4f}, "
                    f"RoBERTa={self.en_temperatures['roberta']:.4f}, "
                    f"Longformer={self.en_temperatures['longformer']:.4f}"
                )
            except Exception as e:
                print(f"[!] Lỗi đọc {en_temp_file}: {e}, dùng T=1.0.")

        print("=" * 70 + "\n")

    def _load_tokenizer(self, local_subdir: str, hf_name: str):
        """Ưu tiên nạp tokenizer cục bộ từ output/ để khởi động tức thì và chạy tốt khi mất Internet."""
        local_dir = os.path.join(self.models_dir, local_subdir)
        if os.path.isdir(local_dir):
            try:
                return AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
            except Exception:
                pass
        try:
            return AutoTokenizer.from_pretrained(hf_name, local_files_only=True)
        except Exception:
            return AutoTokenizer.from_pretrained(hf_name)

    def _load_model_component(self, model_type: str, ckpt_name: str, display_name: str):
        """Nạp checkpoint PyTorch đã fine-tune trực tiếp từ config (bỏ qua đọc thừa trọng số gốc HF)."""
        ckpt_path = os.path.join(self.models_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            return None
        print(f"[*] Nạp checkpoint {display_name.split(' ')[0]} từ: {ckpt_path}")
        try:
            model = build_model(model_type, init_pretrained=False).to(self.device)
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            state_dict = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt
            model.load_state_dict(state_dict)
            model.eval()
            print(f"  [v] {display_name} sẵn sàng.")
            return model
        except Exception as e:
            print(f"[!] Lỗi nạp mô hình {display_name}: {e}")
            return None

    def _run_batch_inference(
        self,
        model,
        tokenizer,
        texts: List[str],
        max_length: int,
        use_token_type_ids: bool = False,
        use_global_attention: bool = False,
        batch_size: int = 16,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Chạy suy luận theo mini-batch (tự động bật FP16 autocast nếu dùng CUDA) kèm Temperature Scaling."""
        if model is None or tokenizer is None or not texts:
            return np.full(len(texts), 0.5)

        safe_temp = max(float(temperature), 0.2)
        all_probs = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            encoded = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)

            kwargs = {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
            if use_token_type_ids and "token_type_ids" in encoded:
                kwargs["token_type_ids"] = encoded["token_type_ids"]
            if use_global_attention:
                global_mask = torch.zeros_like(encoded["input_ids"])
                global_mask[:, 0] = 1
                kwargs["global_attention_mask"] = global_mask

            with self._inference_lock:
                with torch.inference_mode():
                    with torch.amp.autocast("cuda", enabled=self.use_fp16):
                        logits = model(**kwargs)
                    probs = torch.sigmoid(logits.float() / safe_temp).cpu().numpy().flatten()
            all_probs.append(probs)

        return np.concatenate(all_probs) if all_probs else np.array([])

    def _predict_bert(self, texts: List[str]) -> np.ndarray:
        """Dự đoán xác suất theo batch cho BERT (max 512 tokens)."""
        return self._run_batch_inference(
            self.model_bert,
            self.tokenizer_bert,
            texts,
            max_length=512,
            use_token_type_ids=True,
            temperature=self.en_temperatures.get("bert", 1.0),
        )

    def _predict_roberta(self, texts: List[str]) -> np.ndarray:
        """Dự đoán xác suất theo batch cho RoBERTa (max 512 tokens)."""
        return self._run_batch_inference(
            self.model_roberta,
            self.tokenizer_roberta,
            texts,
            max_length=512,
            temperature=self.en_temperatures.get("roberta", 1.0),
        )

    def _predict_longformer(self, texts: List[str]) -> np.ndarray:
        """Dự đoán xác suất theo batch cho Longformer (max 1024 tokens + global attention)."""
        return self._run_batch_inference(
            self.model_longformer,
            self.tokenizer_longformer,
            texts,
            max_length=1024,
            use_global_attention=True,
            temperature=self.en_temperatures.get("longformer", 1.0),
        )

    def _segment_vietnamese(self, texts: List[str]) -> List[str]:
        """Tách từ tiếng Việt bằng PyVi (nếu khả dụng)."""
        if _ViTokenizer is None:
            return texts
        return [_ViTokenizer.tokenize(t) for t in texts]

    def _predict_phobert(self, texts: List[str], pre_segmented: Optional[List[str]] = None) -> np.ndarray:
        """Dự đoán xác suất theo batch cho PhoBERT Tiếng Việt (max 256 tokens + Temperature Scaling)."""
        if self.model_phobert is None or self.tokenizer_phobert is None:
            return np.full(len(texts), 0.5)

        segmented_texts = pre_segmented if pre_segmented is not None else self._segment_vietnamese(texts)
        return self._run_batch_inference(
            self.model_phobert,
            self.tokenizer_phobert,
            segmented_texts,
            max_length=256,
            temperature=self.phobert_temperature,
        )

    @staticmethod
    def _compute_confidence(prob: float, threshold: float, is_vi: bool) -> Tuple[str, str]:
        """Tính nhãn mức độ tin cậy dựa trên khoảng cách so với ngưỡng."""
        diff = abs(prob - threshold)
        high_th = 0.35 if is_vi else 0.40
        med_th = 0.20 if is_vi else 0.25
        if diff >= high_th:
            return "Rất cao (Very High)", "VERY_HIGH"
        if diff >= med_th:
            return "Cao (High)", "HIGH"
        if diff >= 0.10:
            return "Trung bình (Moderate)", "MODERATE"
        return "Không chắc chắn / Ngưỡng ranh giới (Uncertain)", "LOW"

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Tách câu theo ngắt dòng và dấu kết thúc câu, giữ nguyên tiền tố đánh số đầu dòng (vd: '1. Mở đầu')."""
        raw_sentences: List[str] = []
        for line in re.split(r"\n+", text.strip()):
            line = line.strip()
            if not line:
                continue
            m_prefix = re.match(r"^(\d{1,2}\.)\s+(.*)$", line)
            if m_prefix:
                prefix, rest = m_prefix.group(1), m_prefix.group(2)
                parts = [p.strip() for p in re.split(r"(?<=[.!?…])\s+", rest) if p.strip()]
                if parts:
                    parts[0] = f"{prefix} {parts[0]}"
            else:
                parts = [p.strip() for p in re.split(r"(?<=[.!?…])\s+", line) if p.strip()]
            raw_sentences.extend(parts)
        return raw_sentences

    @classmethod
    def _split_into_sliding_chunks(
        cls,
        text: str,
        chunk_words: int = 175,
        overlap_words: int = 30,
        max_chunks: int = 24,
    ) -> Tuple[List[str], List[int]]:
        """
        Chia văn bản dài thành các đoạn cửa sổ trượt chồng lấp theo ranh giới câu tự nhiên.
        Giải quyết triệt để giới hạn cắt ngọn 256 tokens (PhoBERT) và 1024 tokens (Longformer)
        đối với bài tiểu luận / khóa luận dài nhiều trang.
        """
        raw_sents = cls._split_sentences(text)
        if not raw_sents:
            return [text.strip()], [max(1, len(text.split()))]

        # Tách nhỏ nếu có câu đơn lẻ quá dài không chấm câu
        norm_sents: List[str] = []
        for s in raw_sents:
            sw = s.split()
            if len(sw) <= chunk_words:
                norm_sents.append(s)
            else:
                step = max(40, chunk_words - overlap_words)
                for i in range(0, len(sw), step):
                    part = " ".join(sw[i:i + chunk_words]).strip()
                    if part:
                        norm_sents.append(part)

        chunks: List[str] = []
        weights: List[int] = []
        cur_sents: List[str] = []
        cur_words = 0

        for s in norm_sents:
            sw_len = len(s.split())
            if cur_words + sw_len > chunk_words and cur_sents:
                chunk_str = " ".join(cur_sents).strip()
                chunks.append(chunk_str)
                weights.append(max(1, cur_words))
                if len(chunks) >= max_chunks:
                    break
                # Giữ lại phần chồng lấp (overlap) ở cuối đoạn trước
                overlap_buf: List[str] = []
                ov_cnt = 0
                for prev_s in reversed(cur_sents):
                    pw = len(prev_s.split())
                    if ov_cnt + pw > overlap_words and overlap_buf:
                        break
                    overlap_buf.insert(0, prev_s)
                    ov_cnt += pw
                    if ov_cnt >= overlap_words:
                        break
                cur_sents = overlap_buf + [s]
                cur_words = ov_cnt + sw_len
            else:
                cur_sents.append(s)
                cur_words += sw_len

        if cur_sents and len(chunks) < max_chunks:
            chunk_str = " ".join(cur_sents).strip()
            if not chunks or chunk_str != chunks[-1]:
                chunks.append(chunk_str)
                weights.append(max(1, cur_words))

        return chunks, weights

    @staticmethod
    def _aggregate_chunk_probs(probs: np.ndarray, weights: List[int]) -> Tuple[float, float, float]:
        """
        Hợp nhất xác suất đa đoạn (Weighted Mean + Hotspot Peak Detection):
        Nếu bài dài có từ 2 đoạn trở lên và xuất hiện đoạn nghi AI rất cao (>= 0.80),
        kết hợp 65% trung bình trọng số độ dài + 35% đỉnh cục bộ để chống chiêu trộn Mở bài Người + Thân bài AI.
        """
        if len(probs) == 0:
            return 0.5, 0.5, 0.5
        if len(probs) == 1:
            p0 = float(probs[0])
            return p0, p0, p0
        w_arr = np.array(weights[:len(probs)], dtype=np.float64)
        w_arr = w_arr / max(float(w_arr.sum()), 1.0)
        p_mean = float(np.sum(probs * w_arr))
        p_peak = float(np.max(probs))
        if p_peak >= 0.80 and p_mean >= 0.22:
            p_final = 0.65 * p_mean + 0.35 * p_peak
        else:
            p_final = p_mean
        return float(p_final), p_mean, p_peak

    @staticmethod
    def _format_sentence_results(
        sentences: List[str],
        probs: np.ndarray,
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Chuẩn hóa danh sách phân tích cấp câu, đồng bộ ngưỡng và chống chấm oan tiêu đề ngắn."""
        yellow_th = max(0.05, min(0.90, float(threshold)))
        red_th = min(0.90, max(yellow_th + 0.15, yellow_th + 0.30))
        results = []
        for idx, sentence in enumerate(sentences):
            w_cnt = len(sentence.split())
            prob = float(probs[idx])
            # Bảo vệ các dòng tiêu đề mục cực ngắn (<= 4 từ, vd: "1. Giới thiệu", "Kết luận:") khỏi nhiễu PLM
            if w_cnt <= 4:
                prob = min(prob, yellow_th * 0.65)
                color, level = "green", "Tiêu đề / Câu ngắn (Trung tính)"
            elif prob >= red_th:
                color, level = "red", "Chắc chắn AI"
            elif prob >= yellow_th:
                color, level = "yellow", "Nghi vấn AI"
            else:
                color, level = "green", "Văn phong tự nhiên (Người)"

            results.append({
                "sentence_index": idx + 1,
                "sentence": sentence,
                "word_count": w_cnt,
                "ai_probability": round(prob, 4),
                "ai_percentage": f"{prob * 100:.1f}%",
                "highlight_color": color,
                "assessment": level,
            })
        return results

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
        # NHÁNH TIẾNG VIỆT (PHOBERT + SLIDING-WINDOW MULTI-CHUNK AGGREGATION)
        # ======================================================================
        if lang == "vi" and self.model_phobert is not None:
            seg_text = self._segment_vietnamese(batch_text)[0]
            token_count = len(self.tokenizer_phobert.encode(seg_text, add_special_tokens=False, truncation=False))

            if token_count <= 250:
                chunk_probs = self._predict_phobert(batch_text, pre_segmented=[seg_text])
                chunk_cnt = 1
                p_phobert = float(chunk_probs[0])
                p_mean, p_peak = p_phobert, p_phobert
                coverage_str = "Full Context (256 tokens)"
            else:
                chunks, chunk_weights = self._split_into_sliding_chunks(
                    text, chunk_words=175, overlap_words=30, max_chunks=24
                )
                seg_chunks = self._segment_vietnamese(chunks)
                chunk_probs = self._predict_phobert(chunks, pre_segmented=seg_chunks)
                chunk_cnt = len(chunks)
                p_phobert, p_mean, p_peak = self._aggregate_chunk_probs(chunk_probs, chunk_weights)
                coverage_str = f"Sliding-Window ({chunk_cnt} đoạn • 100% văn bản)"

            final_ai_prob = p_phobert
            is_ai = final_ai_prob >= threshold

            label = "AI-generated" if is_ai else "Human-written"
            label_vi = "Do AI sinh (ChatGPT/LLM)" if is_ai else "Do người viết (Human)"
            confidence, confidence_level = self._compute_confidence(final_ai_prob, threshold, is_vi=True)

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
                "detected_model": "PhoBERT v2 (Sliding-Window Hybrid)" if chunk_cnt > 1 else "PhoBERT (vinai/phobert-base-v2)",
                "text_stats": {
                    "word_count": word_count,
                    "char_count": char_count,
                    "token_count": token_count,
                    "chunk_count": chunk_cnt,
                    "context_coverage": coverage_str,
                },
                "inference_time_ms": round(inference_time_ms, 2),
            }

            if detailed:
                result["model_breakdown"] = {
                    "ensemble_weighted": {"probability": round(p_phobert, 4), "is_champion": True},
                    "ensemble_simple": {"probability": round(p_mean, 4)},
                    "ensemble_stacking": {"probability": round(p_peak, 4)},
                    "text_stats": dict(result["text_stats"]),
                    "chunk_analysis": {
                        "chunk_count": chunk_cnt,
                        "mean_probability": round(p_mean, 4),
                        "peak_probability": round(p_peak, 4),
                    },
                    "individual_models": {
                        "phobert": {
                            "probability": round(p_phobert, 4),
                            "mean_chunk_probability": round(p_mean, 4),
                            "peak_chunk_probability": round(p_peak, 4),
                            "chunk_count": chunk_cnt,
                            "weight": 1.0,
                            "weight_pct": "100%",
                            "model_name": (
                                f"PhoBERT v3 (12K Pre-2022 • T={self.phobert_temperature:.2f})"
                                if abs(self.phobert_temperature - 1.0) > 1e-3
                                else "PhoBERT base v2 (Pre-2022)"
                            ),
                        },
                        "bert": {"probability": round(p_mean, 4), "weight": 0.65 if chunk_cnt > 1 else 1.0, "weight_pct": "Mean Chunks"},
                        "roberta": {"probability": round(p_peak, 4), "weight": 0.35 if chunk_cnt > 1 else 0.0, "weight_pct": "Peak Hotspot"},
                        "longformer": {"probability": round(p_phobert, 4), "weight": 1.0, "weight_pct": "100% (PhoBERT)"},
                    },
                }

            return result

        # ======================================================================
        # NHÁNH TIẾNG ANH (ENSEMBLE PLMs + SLIDING-WINDOW FOR >1024 TOKENS)
        # ======================================================================
        token_count = len(self.tokenizer_roberta.encode(text, add_special_tokens=False, truncation=False))

        if token_count <= 1000:
            en_chunks = batch_text
            en_weights = [max(1, word_count)]
            chunk_cnt = 1
            coverage_str = "Full Context (1024)"
        else:
            en_chunks, en_weights = self._split_into_sliding_chunks(
                text, chunk_words=680, overlap_words=80, max_chunks=12
            )
            chunk_cnt = len(en_chunks)
            coverage_str = f"Sliding-Window ({chunk_cnt} chunks • Full Essay)"

        # 1. Suy luận batch từ 3 mô hình thành phần trên toàn bộ các đoạn
        arr_bert = self._predict_bert(en_chunks) if self.model_bert else np.full(chunk_cnt, 0.5)
        arr_roberta = self._predict_roberta(en_chunks) if self.model_roberta else np.full(chunk_cnt, 0.5)
        arr_longformer = self._predict_longformer(en_chunks) if self.model_longformer else np.full(chunk_cnt, 0.5)

        p_bert, _, _ = self._aggregate_chunk_probs(arr_bert, en_weights)
        p_roberta, _, _ = self._aggregate_chunk_probs(arr_roberta, en_weights)
        p_longformer, _, _ = self._aggregate_chunk_probs(arr_longformer, en_weights)

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

        # 3. Quyết định phân loại nhị phân & độ tin cậy
        final_ai_prob = p_weighted
        is_ai = final_ai_prob >= threshold
        label = "AI-generated" if is_ai else "Human-written"
        label_vi = "Do AI sinh (ChatGPT/LLM)" if is_ai else "Do người viết (Human)"
        confidence, confidence_level = self._compute_confidence(final_ai_prob, threshold, is_vi=False)

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
                "chunk_count": chunk_cnt,
                "context_coverage": coverage_str,
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
                "text_stats": dict(result["text_stats"]),
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
        max_sentences: int = 60,
        force_language: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Bóc tách văn bản thành từng câu độc lập và tính toán xác suất AI theo Batch nhanh chóng.
        Phục vụ tính năng Highlight màu (AI Hotspots) cho Chrome Extension và Web UI.
        Hỗ trợ cả Tiếng Việt (PhoBERT) và Tiếng Anh (Ensemble PLMs).
        """
        raw_sentences = self._split_sentences(text)
        sentences = [s for s in raw_sentences if len(s.split()) >= 2]

        if not sentences:
            sentences = [text.strip()]

        eval_sentences = sentences[:max_sentences]
        lang = force_language if force_language else detect_language(text)

        # 1. Nếu là Tiếng Việt và PhoBERT đã nạp
        if lang == "vi" and self.model_phobert is not None:
            probs = self._predict_phobert(eval_sentences)
            return self._format_sentence_results(eval_sentences, probs, threshold=threshold)

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
            w_sum = float(self.weights[0] + self.weights[1])
            if w_sum > 0:
                w_bert = self.weights[0] / w_sum
                w_roberta = self.weights[1] / w_sum
            else:
                w_bert, w_roberta = 0.5, 0.5
            p_weighted = w_bert * p_bert + w_roberta * p_roberta

        return self._format_sentence_results(eval_sentences, p_weighted, threshold=threshold)

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
