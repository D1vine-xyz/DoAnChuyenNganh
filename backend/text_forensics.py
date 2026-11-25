"""
Module Phân Tích Pháp Y Văn Bản, Chống Qua Mặt AI & Giải Thích Ngôn Ngữ Học (Text Forensics & XAI)
Đồ án Chuyên ngành Kỹ thuật Phần mềm (Chủ đề 6): VeriText AI

Gồm 4 thành phần cốt lõi phục vụ Giảng viên & Nhà trường:
1. Anti-Bypass Detector (`detect_bypass_attempts`):
   - Phát hiện ký tự tàng hình (Zero-Width Space, Soft Hyphen, BOM, Word Joiner, Bidi controls).
   - Phát hiện tấn công ký tự đồng hình (Cyrillic / Greek Homoglyph spoofing giả dạng Latin).
   - Phát hiện khoảng trắng Unicode bất thường / chèn ký tự giãn cách phá tokenizer.
   - Tự động trả về `sanitized_text` sạch để lõi AI chấm điểm chính xác không bị đánh lừa.
2. Smart Academic Filter (`apply_academic_filters`):
   - Tùy chọn bỏ qua trích dẫn nguyên văn trong ngoặc kép dài (>= 8 từ).
   - Tùy chọn tự động cắt phần Tài liệu tham khảo / References ở cuối bài tiểu luận.
3. Explainable AI - Linguistic XAI (`analyze_linguistic_xai`):
   - Burstiness Score (0-100): Đo độ biến thiên độ dài & nhịp điệu câu (CV = std / mean).
   - Lexical Diversity (MATTR & TTR, 0-100): Đo độ phong phú từ vựng độc lập với độ dài bài.
   - Perplexity Proxy / Predictability Index (0-100): Kết hợp entropy phân bố từ, lặp đầu câu và mật độ cụm sáo rỗng.
   - AI Cliché Detector (Song ngữ EN + VI): Bắt các cụm từ khuôn mẫu đặc trưng của ChatGPT/Gemini.
4. Multi-Tier Authorship Classification (`classify_authorship_tier`):
   - Phân loại 4 cấp độ thực tế thay vì chỉ 0/1: Tự viết / AI hỗ trợ biên tập / Văn bản lai / AI toàn phần.
"""

import math
import re
import unicodedata
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple


# ==============================================================================
# 1. CẤU HÌNH PRESET NGƯỠNG CHẤM & TỪ ĐIỂN PHÁP Y
# ==============================================================================

STRICTNESS_PRESETS: Dict[str, Tuple[float, str]] = {
    "strict": (0.30, "Khóa luận / Nghiên cứu (Nghiêm ngặt - 30%)"),
    "balanced": (0.50, "Tiểu luận môn học (Cân bằng - 50%)"),
    "lenient": (0.70, "Bài tập tham khảo (Nới lỏng - 70%)"),
}

# Các ký tự tàng hình / điều khiển thường dùng để phá tokenizer
ZERO_WIDTH_CHARS = {
    "\u200b": "Zero-Width Space (ZWSP)",
    "\u200c": "Zero-Width Non-Joiner (ZWNJ)",
    "\u200d": "Zero-Width Joiner (ZWJ)",
    "\u2060": "Word Joiner",
    "\ufeff": "Zero-Width No-Break Space (BOM)",
    "\u00ad": "Soft Hyphen",
    "\u180e": "Mongolian Vowel Separator",
    "\u200e": "Left-to-Right Mark",
    "\u200f": "Right-to-Left Mark",
    "\u202a": "Left-to-Right Embedding",
    "\u202b": "Right-to-Left Embedding",
    "\u202c": "Pop Directional Formatting",
    "\u202d": "Left-to-Right Override",
    "\u202e": "Right-to-Left Override",
    "\u034f": "Combining Grapheme Joiner",
}

# Bảng ánh xạ ký tự đồng hình (Cyrillic & Greek -> Latin tương ứng)
HOMOGLYPH_MAP = {
    # Cyrillic lowercase
    "\u0430": "a",  # а
    "\u0435": "e",  # е
    "\u043e": "o",  # о
    "\u0440": "p",  # р
    "\u0441": "c",  # с
    "\u0443": "y",  # у
    "\u0445": "x",  # х
    "\u0456": "i",  # і
    "\u0458": "j",  # ј
    "\u0455": "s",  # ѕ
    "\u04bb": "h",  # һ
    "\u043a": "k",  # к
    "\u043c": "m",  # м
    "\u043d": "h",  # н
    "\u0442": "t",  # т
    "\u0432": "b",  # в
    # Cyrillic uppercase
    "\u0410": "A",  # А
    "\u0412": "B",  # В
    "\u0415": "E",  # Е
    "\u041a": "K",  # К
    "\u041c": "M",  # М
    "\u041d": "H",  # Н
    "\u041e": "O",  # О
    "\u0420": "P",  # Р
    "\u0421": "C",  # С
    "\u0422": "T",  # Т
    "\u0425": "X",  # Х
    "\u0406": "I",  # І
    "\u0405": "S",  # Ѕ
    "\u0408": "J",  # Ј
    # Greek lookalikes
    "\u03b1": "a",  # α
    "\u03bf": "o",  # ο
    "\u03c1": "p",  # ρ
    "\u03bd": "v",  # ν
    "\u0391": "A",  # Α
    "\u0392": "B",  # Β
    "\u0395": "E",  # Ε
    "\u0397": "H",  # Η
    "\u0399": "I",  # Ι
    "\u039a": "K",  # Κ
    "\u039c": "M",  # Μ
    "\u039d": "N",  # Ν
    "\u039f": "O",  # Ο
    "\u03a1": "P",  # Ρ
    "\u03a4": "T",  # Τ
    "\u03a7": "X",  # Χ
}

# Các khoảng trắng Unicode bất thường dùng để lách bộ tách từ
WEIRD_SPACES = {
    "\u00a0",  # NBSP
    "\u2000", "\u2001", "\u2002", "\u2003", "\u2004",
    "\u2005", "\u2006", "\u2007", "\u2008", "\u2009", "\u200a",
    "\u202f", "\u205f", "\u3000",
}

# Cụm từ sáo rỗng / dấu hiệu khuôn mẫu LLM đặc trưng (Tiếng Việt & Tiếng Anh)
AI_CLICHES_VI = [
    "đóng vai trò then chốt",
    "đóng vai trò quan trọng",
    "không thể phủ nhận",
    "trong bối cảnh hiện nay",
    "trong thời đại công nghệ",
    "bên cạnh đó",
    "hơn nữa",
    "mặt khác",
    "tóm lại",
    "nhìn chung",
    "chính vì vậy",
    "từ đó có thể thấy",
    "có thể thấy rằng",
    "vô cùng cấp thiết",
    "bước chuyển biến to lớn",
    "mang lại nhiều lợi ích",
    "đặt ra nhiều thách thức",
    "một cách ấn tượng",
    "một cách toàn diện",
    "hướng đi vô cùng",
    "nâng cao chất lượng",
    "tối ưu hóa hiệu suất",
    "góp phần không nhỏ",
    "đáng chú ý là",
    "tuy nhiên, sự xuất hiện",
    "cân bằng giữa",
]

AI_CLICHES_EN = [
    "delve into",
    "it is crucial to note",
    "it is important to note",
    "paramount importance",
    "play a pivotal role",
    "plays a crucial role",
    "in conclusion",
    "furthermore",
    "moreover",
    "consequently",
    "in today's digital age",
    "rapidly evolving landscape",
    "multifaceted",
    "testament to",
    "underscores the importance",
    "ever-evolving",
    "foster innovation",
    "holistic approach",
    "harness the power",
    "unprecedented",
    "rich tapestry",
    "navigate the complexities",
    "shed light on",
    "pave the way for",
    "realm of",
    "noteworthy that",
]


def resolve_threshold(threshold: float, strictness_preset: Optional[str] = None) -> Tuple[float, str]:
    """Chuẩn hóa ngưỡng phân loại từ preset môn học (nếu có) hoặc giá trị thanh trượt."""
    if strictness_preset:
        key = strictness_preset.strip().lower()
        if key in STRICTNESS_PRESETS:
            return STRICTNESS_PRESETS[key]
    clamped = max(0.05, min(0.95, float(threshold)))
    return clamped, f"Tùy chỉnh ({int(round(clamped * 100))}%)"


# ==============================================================================
# 2. BỘ PHÁT HIỆN TIỂU XẢO QUA MẶT AI (ANTI-BYPASS DETECTOR)
# ==============================================================================

def detect_bypass_attempts(text: str) -> Dict[str, Any]:
    """
    Quét văn bản đầu vào để phát hiện các kỹ thuật cố tình né máy dò AI:
    - Ký tự tàng hình (Zero-Width / Soft Hyphen / Bidi)
    - Tráo ký tự đồng hình (Cyrillic / Greek Homoglyph)
    - Khoảng trắng Unicode bất thường
    Trả về báo cáo chi tiết kèm `sanitized_text` đã khôi phục về dạng chuẩn.
    """
    if not text:
        return {
            "has_bypass_warning": False,
            "risk_level": "NONE",
            "zero_width_count": 0,
            "homoglyph_count": 0,
            "weird_space_count": 0,
            "total_anomalies": 0,
            "summary": "Không phát hiện dấu hiệu can thiệp kỹ thuật.",
            "flagged_samples": [],
            "sanitized_text": "",
        }

    zero_width_count = 0
    homoglyph_count = 0
    weird_space_count = 0
    flagged_samples: List[str] = []

    # Quét từng từ để gom ví dụ cụ thể cho Giảng viên xem
    raw_tokens = re.findall(r"\S+", text)
    for tok in raw_tokens:
        has_zw = any(ch in ZERO_WIDTH_CHARS for ch in tok)
        has_hg = any(ch in HOMOGLYPH_MAP for ch in tok)
        if (has_zw or has_hg) and len(flagged_samples) < 8:
            clean_tok = "".join(
                HOMOGLYPH_MAP.get(ch, ch) for ch in tok if ch not in ZERO_WIDTH_CHARS
            )
            tag = []
            if has_zw:
                tag.append("ký tự ẩn")
            if has_hg:
                tag.append("Cyrillic/Greek giả Latin")
            flagged_samples.append(f"\"{clean_tok}\" ({' + '.join(tag)})")

    sanitized_chars: List[str] = []
    for ch in text:
        if ch in ZERO_WIDTH_CHARS:
            zero_width_count += 1
            continue
        if ch in HOMOGLYPH_MAP:
            homoglyph_count += 1
            sanitized_chars.append(HOMOGLYPH_MAP[ch])
            continue
        if ch in WEIRD_SPACES:
            weird_space_count += 1
            sanitized_chars.append(" ")
            continue
        sanitized_chars.append(ch)

    sanitized_text = unicodedata.normalize("NFC", "".join(sanitized_chars))
    # Gom khoảng trắng thừa liên tiếp nhưng giữ ngắt đoạn \n
    sanitized_text = re.sub(r"[ \t]{2,}", " ", sanitized_text).strip()

    # Chỉ coi weird_space là bất thường nếu xuất hiện dày (>= 5 lần) hoặc đi kèm zw/homoglyph
    significant_weird_space = weird_space_count if (weird_space_count >= 5 or zero_width_count > 0 or homoglyph_count > 0) else 0
    total_anomalies = zero_width_count + homoglyph_count + significant_weird_space
    has_warning = (zero_width_count >= 1) or (homoglyph_count >= 1) or (weird_space_count >= 8)

    if not has_warning:
        risk_level = "NONE"
        summary = "Văn bản sạch, không phát hiện dấu hiệu chèn ký tự ẩn hoặc giả mạo bảng chữ cái."
    else:
        if total_anomalies >= 10 or (zero_width_count >= 3 and homoglyph_count >= 2):
            risk_level = "HIGH"
        elif total_anomalies >= 3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        parts = []
        if zero_width_count > 0:
            parts.append(f"{zero_width_count} ký tự tàng hình (Zero-Width)")
        if homoglyph_count > 0:
            parts.append(f"{homoglyph_count} ký tự đồng hình Cyrillic/Hy Lạp giả Latin")
        if significant_weird_space > 0:
            parts.append(f"{significant_weird_space} khoảng trắng Unicode bất thường")

        summary = (
            f"CẢNH BÁO LÁCH LUẬT: Phát hiện {', '.join(parts)}. "
            "Hệ thống đã tự động làm sạch và khôi phục văn bản gốc trước khi chấm điểm."
        )

    return {
        "has_bypass_warning": has_warning,
        "risk_level": risk_level,
        "zero_width_count": zero_width_count,
        "homoglyph_count": homoglyph_count,
        "weird_space_count": weird_space_count,
        "total_anomalies": total_anomalies,
        "summary": summary,
        "flagged_samples": flagged_samples,
        "sanitized_text": sanitized_text,
    }


# ==============================================================================
# 3. BỘ LỌC HỌC THUẬT THÔNG MINH (SMART ACADEMIC FILTERS)
# ==============================================================================

REFERENCE_HEADING_PATTERN = re.compile(
    r"(?im)^\s*(?:\d+\.?\s*)?(?:tài\s+liệu\s+tham\s+khảo|danh\s+mục\s+tài\s+liệu\s+tham\s+khảo|references|bibliography|works\s+cited)\s*:?\s*$"
)

LONG_QUOTE_PATTERN = re.compile(r'(["“«])([^"”»]{35,})(["”»])')


def apply_academic_filters(
    text: str,
    filter_quotes: bool = False,
    filter_references: bool = False,
) -> Dict[str, Any]:
    """
    Loại bỏ các thành phần học thuật dễ gây chấm oan (trích dẫn nguyên văn & danh mục tài liệu tham khảo).
    Nếu sau khi lọc văn bản còn quá ngắn (< 5 từ), tự động giữ lại bản gốc để không làm hỏng suy luận.
    """
    original_words = len(text.split())
    working = text
    quotes_removed = 0
    references_stripped = False

    if filter_references and working:
        matches = list(REFERENCE_HEADING_PATTERN.finditer(working))
        if matches:
            # Lấy mục Tài liệu tham khảo cuối cùng nếu nằm sau ít nhất 30% độ dài văn bản
            last_match = matches[-1]
            if last_match.start() >= int(len(working) * 0.30):
                working = working[:last_match.start()].strip()
                references_stripped = True

    if filter_quotes and working:
        def _replace_long_quote(match: re.Match) -> str:
            nonlocal quotes_removed
            inner = match.group(2).strip()
            if len(inner.split()) >= 8:
                quotes_removed += 1
                return " "
            return match.group(0)

        working = LONG_QUOTE_PATTERN.sub(_replace_long_quote, working)
        working = re.sub(r"[ \t]{2,}", " ", working).strip()

    filtered_words = len(working.split())
    fallback_used = False
    if filtered_words < 5 and original_words >= 5:
        working = text
        filtered_words = original_words
        fallback_used = True

    applied_notes = []
    if references_stripped and not fallback_used:
        applied_notes.append("đã bỏ mục Tài liệu tham khảo")
    if quotes_removed > 0 and not fallback_used:
        applied_notes.append(f"đã loại trừ {quotes_removed} đoạn trích dẫn nguyên văn")

    if applied_notes:
        filter_summary = f"Bộ lọc học thuật: {', '.join(applied_notes)} ({original_words} -> {filtered_words} từ)."
    elif fallback_used:
        filter_summary = "Bộ lọc học thuật: Nội dung còn lại quá ngắn nên đã giữ nguyên toàn văn để thẩm định."
    elif filter_quotes or filter_references:
        filter_summary = "Bộ lọc học thuật đang bật: Không tìm thấy trích dẫn dài hoặc mục tài liệu tham khảo cần cắt."
    else:
        filter_summary = ""

    return {
        "filtered_text": working,
        "filter_quotes_enabled": filter_quotes,
        "filter_references_enabled": filter_references,
        "quotes_removed_count": 0 if fallback_used else quotes_removed,
        "references_stripped": False if fallback_used else references_stripped,
        "original_word_count": original_words,
        "filtered_word_count": filtered_words,
        "filter_applied": bool(applied_notes),
        "filter_summary": filter_summary,
    }


# ==============================================================================
# 4. ĐỘNG CƠ GIẢI THÍCH NGÔN NGỮ HỌC (EXPLAINABLE AI - XAI)
# ==============================================================================

def _compute_mattr(tokens: List[str], window: int = 50) -> float:
    """Tính Moving-Average Type-Token Ratio (MATTR) không bị lệch bởi độ dài bài viết."""
    n = len(tokens)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(tokens)) / n

    counts = Counter(tokens[:window])
    unique_count = len(counts)
    total_ttr = unique_count / window
    steps = n - window

    for i in range(steps):
        out_tok = tokens[i]
        in_tok = tokens[i + window]
        counts[out_tok] -= 1
        if counts[out_tok] == 0:
            del counts[out_tok]
            unique_count -= 1
        if counts[in_tok] == 0:
            unique_count += 1
        counts[in_tok] += 1
        total_ttr += unique_count / window

    return total_ttr / (steps + 1)


def analyze_linguistic_xai(text: str, language: str = "en") -> Dict[str, Any]:
    """
    Tính toán các chỉ số bằng chứng ngôn ngữ học (XAI) dễ hiểu cho Giảng viên:
    - Burstiness Score (0 - 100): Biến thiên nhịp điệu câu (cao = giống người, thấp = đều như máy).
    - Lexical Diversity / MATTR (0 - 100): Độ đa dạng vốn từ.
    - Perplexity Proxy (0 - 100): Chỉ số độ bất ngờ từ vựng & cấu trúc (thấp = dễ đoán kiểu LLM).
    - AI Cliché Matches: Danh sách cụm từ sáo rỗng đặc trưng AI tìm thấy trong bài.
    """
    clean = text.strip()
    if not clean:
        return {
            "burstiness_score": 0.0,
            "burstiness_label": "Chưa đủ dữ liệu",
            "perplexity_score": 0.0,
            "perplexity_label": "Chưa đủ dữ liệu",
            "lexical_diversity": 0.0,
            "ttr_score": 0.0,
            "sentence_stats": {
                "count": 0, "avg_length": 0.0, "std_length": 0.0,
                "min_length": 0, "max_length": 0, "starter_repetition_rate": 0.0,
            },
            "ai_cliches": [],
            "ai_cliche_count": 0,
            "explanation_bullets": [],
        }

    # 1. Phân tích cấp câu -> Burstiness
    raw_sentences = re.split(r"(?<=[.!?…\n])\s+", clean)
    sentences = [s.strip() for s in raw_sentences if len(s.strip().split()) >= 2]
    if not sentences:
        sentences = [clean]

    sent_lengths = [len(s.split()) for s in sentences]
    avg_len = sum(sent_lengths) / len(sent_lengths)
    if len(sent_lengths) >= 2:
        variance = sum((x - avg_len) ** 2 for x in sent_lengths) / len(sent_lengths)
        std_len = math.sqrt(variance)
        cv = std_len / max(avg_len, 1.0)
        # Quy đổi CV (thường 0.10 - 0.80) về thang 0 - 100
        burstiness_score = round(min(100.0, max(5.0, cv * 135.0)), 1)
    else:
        std_len = 0.0
        burstiness_score = 45.0  # 1 câu đơn chưa đủ đo nhịp điệu

    if burstiness_score < 32:
        burstiness_label = "Thấp (Nhịp câu đều đặn kiểu AI)"
    elif burstiness_score < 58:
        burstiness_label = "Trung bình (Cân bằng)"
    else:
        burstiness_label = "Cao (Nhịp điệu tự nhiên kiểu Người)"

    # Tỷ lệ lặp từ mở đầu câu
    starters = [re.sub(r"^\W+", "", s.split()[0].lower()) for s in sentences if s.split()]
    starter_rep_rate = 0.0
    if len(starters) >= 3:
        starter_rep_rate = round((1.0 - (len(set(starters)) / len(starters))) * 100.0, 1)

    # 2. Phân tích từ vựng -> TTR, MATTR & Shannon Entropy
    word_tokens = re.findall(r"[A-Za-zÀ-ỹ0-9]+", clean.lower())
    n_tokens = len(word_tokens)
    ttr = (len(set(word_tokens)) / n_tokens) if n_tokens > 0 else 0.0
    mattr = _compute_mattr(word_tokens, window=50)
    lexical_diversity = round(mattr * 100.0, 1)

    # Shannon Entropy chuẩn hóa (0 - 1)
    if n_tokens > 1:
        freq = Counter(word_tokens)
        entropy = -sum((c / n_tokens) * math.log2(c / n_tokens) for c in freq.values())
        max_entropy = math.log2(n_tokens)
        norm_entropy = entropy / max(max_entropy, 1e-6)
    else:
        norm_entropy = 0.5

    # 3. Bắt cụm từ khuôn mẫu (AI Clichés)
    lower_text = clean.lower()
    cliche_bank = AI_CLICHES_VI if language == "vi" else AI_CLICHES_EN
    matched_cliches: List[Dict[str, Any]] = []
    total_cliche_hits = 0

    for phrase in cliche_bank:
        occurrences = len(re.findall(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", lower_text))
        if occurrences > 0:
            matched_cliches.append({"phrase": phrase, "count": occurrences})
            total_cliche_hits += occurrences

    matched_cliches.sort(key=lambda item: item["count"], reverse=True)

    # 4. Perplexity Proxy / Độ bất ngờ ngôn ngữ (0 - 100)
    # Thấp = văn bản dễ đoán, lặp khuôn mẫu AI; Cao = bất ngờ, phong phú kiểu con người
    cliche_penalty = min(35.0, (total_cliche_hits / max(len(sentences), 1)) * 28.0)
    rep_penalty = min(15.0, starter_rep_rate * 0.25)
    raw_perplexity = (norm_entropy * 55.0) + (mattr * 35.0) + (burstiness_score * 0.20) - cliche_penalty - rep_penalty
    perplexity_score = round(min(98.0, max(8.0, raw_perplexity)), 1)

    if perplexity_score < 42:
        perplexity_label = "Thấp (Văn phong dễ đoán, khuôn mẫu AI)"
    elif perplexity_score < 65:
        perplexity_label = "Trung bình (Pha trộn)"
    else:
        perplexity_label = "Cao (Cách diễn đạt sáng tạo, khó đoán)"

    # 5. Sinh nhận định giải thích ngắn gọn cho Giảng viên
    bullets: List[str] = []
    if len(sentences) >= 2:
        bullets.append(
            f"Độ dài trung bình {avg_len:.1f} từ/câu (độ lệch chuẩn ±{std_len:.1f} từ) → Burstiness {burstiness_score}/100 ({burstiness_label})."
        )
    else:
        bullets.append(f"Văn bản chỉ có 1 câu ({avg_len:.0f} từ), độ đa dạng từ vựng đạt {lexical_diversity}%.")

    if total_cliche_hits > 0:
        top_phrases = ", ".join(f'"{m["phrase"]}"' for m in matched_cliches[:4])
        bullets.append(
            f"Phát hiện {total_cliche_hits} cụm từ nối/văn mẫu thường gặp ở ChatGPT/Gemini: {top_phrases}."
        )
    else:
        bullets.append("Không phát hiện cụm từ sáo rỗng khuôn mẫu đặc trưng của LLM trong văn bản.")

    if starter_rep_rate >= 30.0:
        bullets.append(f"Tỷ lệ lặp từ mở đầu câu khá cao ({starter_rep_rate}%), dấu hiệu cấu trúc liệt kê máy móc.")

    return {
        "burstiness_score": burstiness_score,
        "burstiness_label": burstiness_label,
        "perplexity_score": perplexity_score,
        "perplexity_label": perplexity_label,
        "lexical_diversity": lexical_diversity,
        "ttr_score": round(ttr * 100.0, 1),
        "sentence_stats": {
            "count": len(sentences),
            "avg_length": round(avg_len, 1),
            "std_length": round(std_len, 1),
            "min_length": min(sent_lengths),
            "max_length": max(sent_lengths),
            "starter_repetition_rate": starter_rep_rate,
        },
        "ai_cliches": matched_cliches[:10],
        "ai_cliche_count": total_cliche_hits,
        "explanation_bullets": bullets,
    }


# ==============================================================================
# 5. PHÂN LOẠI 4 CẤP ĐỘ CAN THIỆP THỰC TẾ (4-TIER AUTHORSHIP CLASSIFICATION)
# ==============================================================================

def classify_authorship_tier(
    overall_ai_prob: float,
    sentences: List[Dict[str, Any]],
    threshold: float = 0.5,
    has_bypass_warning: bool = False,
    xai_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Phân loại mức độ can thiệp AI thực tế cho môi trường giáo dục thành 4 trạng thái:
    - HUMAN_PURE   : Tự viết hoàn toàn
    - AI_ASSISTED  : Có dấu hiệu nhờ AI biên tập / Paraphrase
    - HYBRID_MIXED : Văn bản lai (Trộn lẫn đoạn người tự viết + đoạn AI sinh)
    - AI_PURE      : AI sinh toàn phần
    """
    total_sents = len(sentences)
    red_cnt = sum(1 for s in sentences if s.get("highlight_color") == "red")
    yellow_cnt = sum(1 for s in sentences if s.get("highlight_color") == "yellow")
    green_cnt = sum(1 for s in sentences if s.get("highlight_color") == "green")

    red_ratio = red_cnt / max(total_sents, 1)
    green_ratio = green_cnt / max(total_sents, 1)

    # Phát hiện văn bản lai (có cả phần chắc chắn AI lẫn phần chắc chắn tự nhiên)
    is_hybrid_pattern = (
        total_sents >= 3
        and red_cnt >= 1
        and green_cnt >= 1
        and (0.20 <= red_ratio <= 0.78)
        and (green_ratio >= 0.20)
    )

    if overall_ai_prob >= max(0.80, threshold + 0.20) and not is_hybrid_pattern:
        tier_code = "AI_PURE"
        label_vi = "🤖 AI sinh toàn phần (Pure AI)"
        badge_class = "tier-ai-pure"
        reason = (
            f"Xác suất AI tổng thể rất cao ({overall_ai_prob*100:.1f}%) với {red_cnt}/{total_sents} câu "
            "mang đậm dấu vết ngôn ngữ của LLM."
        )
    elif is_hybrid_pattern:
        tier_code = "HYBRID_MIXED"
        label_vi = "🧩 Văn bản lai: Trộn Người + AI (Hybrid)"
        badge_class = "tier-hybrid"
        reason = (
            f"Bài viết có sự pha trộn rõ rệt: {red_cnt} câu nghi AI cao xen kẽ {green_cnt} câu văn phong tự nhiên "
            "(dấu hiệu tự viết một phần và dùng AI viết phần còn lại)."
        )
    elif overall_ai_prob >= threshold or (yellow_cnt >= 2 and overall_ai_prob >= 0.35) or (has_bypass_warning and overall_ai_prob >= 0.35):
        tier_code = "AI_ASSISTED"
        label_vi = "✍️ Nghi vấn AI biên tập / Paraphrase"
        badge_class = "tier-assisted"
        reason = (
            f"Xác suất AI ở mức cảnh báo ({overall_ai_prob*100:.1f}%, {yellow_cnt} câu vàng / {red_cnt} câu đỏ) — "
            "thường gặp khi sinh viên tự lên ý rồi nhờ AI trau chuốt hoặc viết lại câu."
        )
    else:
        tier_code = "HUMAN_PURE"
        label_vi = "👤 Tự viết hoàn toàn (Human-Written)"
        badge_class = "tier-human"
        reason = (
            f"Xác suất AI thấp ({overall_ai_prob*100:.1f}%), {green_cnt}/{total_sents} câu có cấu trúc và nhịp điệu tự nhiên."
        )

    if has_bypass_warning:
        reason += " Lưu ý: Phát hiện dấu hiệu chèn ký tự lách bộ lọc!"

    return {
        "code": tier_code,
        "label_vi": label_vi,
        "badge_class": badge_class,
        "reason": reason,
        "sentence_distribution": {
            "red": red_cnt,
            "yellow": yellow_cnt,
            "green": green_cnt,
            "total": total_sents,
        },
    }
