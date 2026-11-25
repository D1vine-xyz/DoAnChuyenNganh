"""
Module Xây Dựng Bộ Dữ Liệu Tiếng Việt Đa Miền Chuẩn Mốc Thời Gian Tiền ChatGPT (<= 11/2022)
Đồ án Chuyên ngành Kỹ thuật Phần mềm (Chủ đề 6: VeriText AI PRO)

Thiết kế khoa học (8.000 mẫu cân bằng 50/50 + Phân tầng theo thể loại):
- LỚP 0 — HUMAN THUẦN NGƯỜI TIỀN 2022 (4.000 mẫu, 100% xuất bản <= 11/2022):
  1. Bách khoa & Học thuật (UIT-ViQuAD 2.0 - VLSP 2021)            : 1.000 mẫu
  2. Báo chí & Truyền thông chính thống (VietNews - 2018/2019)     : 1.000 mẫu
  3. Văn bản Pháp luật / Hành chính trang trọng (<= 2022)          :   500 mẫu
  4. Sách Tri thức & Phi hư cấu tiếng Việt (Books Corpus)          :   600 mẫu
  5. Văn học kinh điển Vũ Trọng Phụng (1912-1939)                  :   350 mẫu
  6. Truyện ngắn / Văn xuôi kể chuyện                              :   200 mẫu
  7. Thơ ca cổ điển & tiền chiến Việt Nam (Lê, Nguyễn, Tây Sơn...) :   350 mẫu

- LỚP 1 — AI ĐA MÔ HÌNH & ĐA THỂ LOẠI (4.000 mẫu, khớp phân bố độ dài):
  1. AI Tổng hợp / Tiểu luận (kngann2201)                          : 1.300 mẫu
  2. AI Công nghệ / Tin tức sinh mới (DeckardLong generated)       :   650 mẫu
  3. AI Paraphrase / Viết lại bài báo (DeckardLong rewrited)       :   650 mẫu
  4. AI Giải thích / Học thuật (BKAI ShareGPT Vietnamese)          :   730 mẫu
  5. AI Sáng tác Văn học / Tiểu thuyết (Gemma-3-12B & Sailor2-8B)  :   500 mẫu
  6. AI Paraphrase Gemini (Nguồn gốc giai đoạn 1 dự án)            :   170 mẫu

Lưu ra bộ tệp riêng `vietnamese_pre2022_*.csv` để người dùng duyệt trước khi huấn luyện lại PhoBERT.
"""

import os
import re
import sys
import random
import unicodedata
from typing import List, Dict, Any

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from datasets import load_dataset
from huggingface_hub import HfApi

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

POST_2022_PATTERN = re.compile(
    r"\b(2023|2024|2025|2026|chatgpt|gpt-4|gpt-3\.5|gemini|claude|openai|llm|trí tuệ nhân tạo tạo sinh)\b",
    re.IGNORECASE,
)
LAW_PRE2022_YEAR_PATTERN = re.compile(
    r"\b\d+\s*/\s*(198\d|199\d|200\d|201\d|202[012])\s*/\s*(QH|NĐ|ND|TT|QĐ|QD|NQ|PL|UBTVQH|CP|BTC|BGD|BYT|BTTTT|BLĐTBXH)",
    re.IGNORECASE,
)
VI_DIACRITICS = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")


def has_good_vietnamese(text: str) -> bool:
    """Đảm bảo văn bản thực sự là tiếng Việt có dấu tự nhiên, không bị lỗi font hoặc toàn code."""
    if not text:
        return False
    low = text.lower()
    vi_cnt = sum(1 for ch in low if ch in VI_DIACRITICS)
    alpha_cnt = sum(1 for ch in low if ch.isalpha())
    if alpha_cnt < 40:
        return False
    return (vi_cnt / alpha_cnt) >= 0.12


POST_2022_YEAR_ONLY = re.compile(r"\b(2023|2024|2025|2026)\b")


def clean_prose(text: str, keep_newlines: bool = False) -> str:
    """Làm sạch văn bản Unicode NFC, chuẩn hóa link Markdown, bỏ URL/HTML thừa."""
    if not text or not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    # Chuyển markdown link [Tên văn bản](https://...) -> Tên văn bản TRƯỚC khi xóa URL
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(\s*", r"\1 ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\\_", "_")
    if keep_newlines:
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
        lines = [ln for ln in lines if ln]
        return "\n".join(lines).strip()
    return re.sub(r"\s+", " ", text).strip()


def trim_to_clean_sentence_start(text: str) -> str:
    """Nếu đoạn văn bị cắt giữa chừng ở đầu (chữ thường), bỏ phần dở dang đầu câu."""
    t = text.strip()
    if not t:
        return ""
    if t[0].islower():
        m = re.search(r"[.!?…]\s+([A-ZÀ-Ỹ\"“])", t)
        if m:
            t = t[m.start(1):].strip()
    return t


def detokenize_vietnews(text: str) -> str:
    """Khôi phục văn bản VietNews đã tách từ bằng gạch dưới (_) về tiếng Việt tự nhiên."""
    t = clean_prose(text)
    t = t.replace("_", " ")
    t = re.sub(r"\s+([,.;:?!\)\]\}])", r"\1", t)
    t = re.sub(r"([\(\[\{])\s+", r"\1", t)
    t = re.sub(r'"\s+([^"]+?)\s+"', r'"\1"', t)
    return re.sub(r"\s+", " ", t).strip()


def strip_markdown_lightly(text: str) -> str:
    """Loại bỏ ký hiệu Markdown thừa (**, ##, ```) để mô hình không học vẹt dấu sao Markdown."""
    t = re.sub(r"```[\s\S]*?```", " ", text)
    t = re.sub(r"^#{1,6}\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"\*([^*\n]+)\*", r"\1", t)
    t = re.sub(r"^\s*[-•*]\s+", "", t, flags=re.MULTILINE)
    return clean_prose(t)


def slice_by_sentences(text: str, min_words: int = 60, max_words: int = 340, vary_length: bool = False) -> str:
    """Cắt đoạn văn sạch theo ranh giới câu tự nhiên trong khoảng độ dài chuẩn cho PhoBERT."""
    words = text.split()
    if len(words) < min_words:
        return ""
    target_max = max_words
    if vary_length and len(words) > 150:
        # Tạo sự đa dạng độ dài (ngắn 70-150 từ, vừa 150-240 từ, dài 240-340 từ)
        bucket = random.random()
        if bucket < 0.30:
            target_max = random.randint(80, 150)
        elif bucket < 0.65:
            target_max = random.randint(155, 250)
        else:
            target_max = random.randint(255, max_words)

    if len(words) <= target_max:
        return text

    sents = re.split(r"(?<=[.!?…])\s+", text)
    acc = []
    w_cnt = 0
    for s in sents:
        sw = len(s.split())
        if w_cnt + sw > target_max and w_cnt >= min_words:
            break
        acc.append(s)
        w_cnt += sw
        if w_cnt >= target_max:
            break
    out = " ".join(acc).strip()
    out_words = out.split()
    if len(out_words) < min_words or len(out_words) > target_max:
        out = " ".join(words[:target_max]).strip()
    return out


def build_pre2022_dataset():
    print("=" * 78)
    print("  XÂY DỰNG BỘ DỮ LIỆU TIẾNG VIỆT ĐA MIỀN CHUẨN TIỀN 2022 (8.000 MẪU)")
    print("=" * 78)

    records: List[Dict[str, Any]] = []
    seen_fingerprints = set()

    def add_record(text: str, label: int, source: str, genre: str, era: str) -> bool:
        if not text:
            return False
        fp = re.sub(r"\W+", "", text[:180].lower())
        if len(fp) < 30 or fp in seen_fingerprints:
            return False
        seen_fingerprints.add(fp)
        records.append({
            "text": text,
            "label": label,
            "label_name": "Human" if label == 0 else "AI-generated",
            "source": source,
            "genre": genre,
            "era_verified": era,
            "word_count": len(text.split()),
            "char_count": len(text),
        })
        return True

    # ==========================================================================
    # PHẦN A: 4.000 MẪU HUMAN THUẦN NGƯỜI TIỀN 2022 (<= 11/2022)
    # ==========================================================================

    # 1. Bách khoa & Học thuật — UIT-ViQuAD 2.0 (VLSP 2021) -> 1.000 mẫu
    print("\n[H1] Đang thu thập 1.000 mẫu Bách khoa & Học thuật từ UIT-ViQuAD 2.0 (VLSP 2021)...")
    ds_viquad = load_dataset("taidng/UIT-ViQuAD2.0", split="train", streaming=True)
    cnt = 0
    for row in ds_viquad:
        ctx = clean_prose(row.get("context", ""))
        if not has_good_vietnamese(ctx):
            continue
        seg = slice_by_sentences(ctx, min_words=60, max_words=340, vary_length=True)
        if seg and add_record(seg, 0, "UIT-ViQuAD2.0 (VLSP 2021)", "HocThuat_BachKhoa", "<=2021"):
            cnt += 1
            if cnt >= 1000:
                break
    print(f"    -> Đã lấy: {cnt} mẫu (HocThuat_BachKhoa)")

    # 2. Báo chí & Truyền thông chính thống — VietNews (2018-2019) -> 1.000 mẫu
    print("[H2] Đang thu thập 1.000 mẫu Báo chí chính thống từ VietNews (2019)...")
    ds_vnews = load_dataset("nam194/vietnews", split="train", streaming=True)
    cnt = 0
    for row in ds_vnews:
        art = detokenize_vietnews(f"{row.get('abstract', '')} {row.get('article', '')}")
        if not has_good_vietnamese(art):
            continue
        seg = slice_by_sentences(art, min_words=65, max_words=340, vary_length=True)
        if seg and add_record(seg, 0, "VietNews (2019)", "BaoChi_TruyenThong", "<=2019"):
            cnt += 1
            if cnt >= 1000:
                break
    print(f"    -> Đã lấy: {cnt} mẫu (BaoChi_TruyenThong)")

    # 3. Văn bản Pháp luật / Nghị định / Hành chính trang trọng (lọc cứng năm <= 2022) -> 500 mẫu
    print("[H3] Đang thu thập 500 mẫu Văn bản Pháp luật / Học thuật hành chính (năm <= 2022)...")
    ds_law = load_dataset("kiil-lab/vietnamese-law-corpus", split="train", streaming=True)
    cnt = 0
    scanned_law = 0
    for row in ds_law:
        scanned_law += 1
        md = str(row.get("markdown", ""))
        head = md[:600]
        if not LAW_PRE2022_YEAR_PATTERN.search(head):
            if scanned_law > 12000:
                break
            continue
        # Lấy các đoạn văn điều khoản sạch (bỏ bảng markdown |...| và phần tiêu đề/Căn cứ)
        lines = []
        for ln in md.splitlines():
            s = ln.strip()
            if not s or s.startswith("|") or s.startswith("---") or s.startswith("\\_"):
                continue
            s = strip_markdown_lightly(s)
            if s.lower().startswith("căn cứ "):
                continue
            if len(s.split()) >= 12 and not s.isupper():
                lines.append(s)
        body = " ".join(lines[:10])
        body = clean_prose(body)
        if not has_good_vietnamese(body) or POST_2022_PATTERN.search(body):
            continue
        seg = slice_by_sentences(body, min_words=65, max_words=320, vary_length=True)
        if seg and add_record(seg, 0, "VN-Law-Corpus (<=2022)", "PhapLuat_ChinhThong", "<=2022"):
            cnt += 1
            if cnt >= 500:
                break
    print(f"    -> Đã lấy: {cnt} mẫu (PhapLuat_ChinhThong, quét {scanned_law} văn bản)")

    # 4. Sách Tri thức & Phi hư cấu tiếng Việt — Vietnamese Books -> 600 mẫu
    print("[H4] Đang thu thập 600 mẫu Sách Tri thức & Văn xuôi từ Vietnamese-Books-dedup...")
    ds_books = load_dataset("tmnam20/Vietnamese-Books-dedup", split="train", streaming=True)
    cnt = 0
    scanned_books = 0
    for row in ds_books:
        scanned_books += 1
        raw = clean_prose(row.get("text", ""))
        if len(raw.split()) < 80 or raw.isupper():
            continue
        if POST_2022_PATTERN.search(raw) or not has_good_vietnamese(raw):
            continue
        # Kiểm tra có ít nhất 3 câu tự nhiên, tránh mục lục / danh sách khô
        if raw.count(".") + raw.count("?") + raw.count("!") < 3:
            continue
        seg = slice_by_sentences(raw, min_words=75, max_words=340, vary_length=True)
        if seg and add_record(seg, 0, "Vietnamese-Books-dedup", "Sach_TriThuc", "<=2022"):
            cnt += 1
            if cnt >= 600:
                break
        if scanned_books > 40000:
            break
    print(f"    -> Đã lấy: {cnt} mẫu (Sach_TriThuc)")

    # 5. Văn học kinh điển Vũ Trọng Phụng (1912-1939) -> 350 mẫu
    print("[H5] Đang thu thập 350 mẫu Văn học kinh điển Vũ Trọng Phụng (1912-1939)...")
    ds_vtp = load_dataset("trieunh/Vietnamese_literature_VuTrongPhung", split="train", streaming=True)
    cnt = 0
    for row in ds_vtp:
        content = trim_to_clean_sentence_start(clean_prose(row.get("content", "")))
        if not has_good_vietnamese(content):
            continue
        seg = slice_by_sentences(content, min_words=60, max_words=340, vary_length=True)
        if seg and add_record(seg, 0, "VuTrongPhung (1912-1939)", "VanHoc_KinhDien", "<=1939"):
            cnt += 1
            if cnt >= 350:
                break
    print(f"    -> Đã lấy: {cnt} mẫu (VanHoc_KinhDien)")

    # 6. Truyện ngắn / Văn xuôi kể chuyện -> 200 mẫu
    print("[H6] Đang thu thập 200 mẫu Truyện ngắn từ vietnamese_story...")
    ds_story = load_dataset("truongpdd/vietnamese_story", split="train", streaming=True)
    cnt = 0
    for row in ds_story:
        txt = trim_to_clean_sentence_start(clean_prose(row.get("text", "")))
        if POST_2022_PATTERN.search(txt) or not has_good_vietnamese(txt):
            continue
        seg = slice_by_sentences(txt, min_words=65, max_words=340, vary_length=True)
        if seg and add_record(seg, 0, "Vietnamese-Story", "TruyenNgan", "<=2022"):
            cnt += 1
            if cnt >= 200:
                break
    print(f"    -> Đã lấy: {cnt} mẫu (TruyenNgan)")

    # 7. Thơ ca cổ điển & tiền chiến Việt Nam (LOẠI TRỪ 'Hiện đại' để đảm bảo 100% <= 2022) -> 350 mẫu
    print("[H7] Đang thu thập 350 mẫu Thơ ca truyền thống/cổ điển từ vietnamese-poetry-corpus...")
    ds_poem = load_dataset("phamson02/vietnamese-poetry-corpus", split="train", streaming=True)
    cnt = 0
    for row in ds_poem:
        period = str(row.get("period", "") or "").strip()
        if not period or "hiện đại" in period.lower():
            continue
        raw_poem = str(row.get("content", ""))
        if "< >" not in raw_poem and "\n" not in raw_poem:
            continue
        # Format < > thành xuống dòng câu thơ tự nhiên & viết hoa chữ cái đầu mỗi câu thơ
        poem = raw_poem.replace("< >", "\n").replace("<>", "\n")
        poem = re.sub(r"\s+([,.;:?!])", r"\1", poem)
        poem = clean_prose(poem, keep_newlines=True)
        poem_lines = [ln[0].upper() + ln[1:] if ln else "" for ln in poem.splitlines()]
        poem = "\n".join(poem_lines)
        w_len = len(poem.split())
        if w_len < 35 or w_len > 260 or not has_good_vietnamese(poem):
            continue
        if add_record(poem, 0, f"Poetry-VN ({period})", "VanTho_TruyenThong", "<=1945"):
            cnt += 1
            if cnt >= 350:
                break
    print(f"    -> Đã lấy: {cnt} mẫu (VanTho_TruyenThong)")

    human_total = sum(1 for r in records if r["label"] == 0)
    print(f"\n>>> TỔNG MẪU HUMAN TIỀN 2022 ĐÃ THU THẬP: {human_total:,} mẫu")

    # ==========================================================================
    # PHẦN B: 4.000 MẪU AI ĐA MÔ HÌNH & ĐA THỂ LOẠI (KHỚP TỔNG HUMAN)
    # ==========================================================================

    # 1. Gemini Paraphrase gốc giai đoạn 1 -> 170 mẫu
    print("\n[A1] Đang nạp AI Gemini Paraphrase từ nguồn gốc dự án...")
    cnt = 0
    backup_1k = os.path.join(DATA_DIR, "vietnamese_full_dataset_1k_backup.csv")
    if not os.path.exists(backup_1k):
        backup_1k = os.path.join(DATA_DIR, "vietnamese_full_dataset.csv")
    if os.path.exists(backup_1k):
        df_old = pd.read_csv(backup_1k)
        for _, row in df_old[df_old["label"] == 1].iterrows():
            src_name = str(row.get("source", ""))
            if "gemini" in src_name.lower() or "paraphrase" in src_name.lower():
                t = clean_prose(str(row["text"]))
                seg = slice_by_sentences(t, min_words=55, max_words=340, vary_length=True)
                if seg and add_record(seg, 1, "Gemini-Paraphrase", "AI_Paraphrase", "LLM"):
                    cnt += 1
                    if cnt >= 170:
                        break
    print(f"    -> Đã lấy: {cnt} mẫu (AI_Paraphrase - Gemini)")

    # 2. AI Công nghệ & AI Paraphrase — DeckardLong (chống rò rỉ mốc năm 2023-2026)
    print("[A2] Đang thu thập 1.300 mẫu AI Generated & AI Rewritten từ DeckardLong (lọc chống rò rỉ năm)...")
    ds_deck = load_dataset("DeckardLong/ai-content-detection-dataset", split="train", streaming=True)
    cnt_gen, cnt_rew = 0, 0
    for row in ds_deck:
        if int(row.get("label", 0)) != 1:
            continue
        gtype = str(row.get("generated_type", "")).lower()
        t = strip_markdown_lightly(str(row.get("text", "")))
        if not has_good_vietnamese(t) or POST_2022_YEAR_ONLY.search(t):
            continue
        seg = slice_by_sentences(t, min_words=65, max_words=340, vary_length=True)
        if not seg:
            continue
        if "gen" in gtype and cnt_gen < 650:
            if add_record(seg, 1, "DeckardLong-AI-Generated", "AI_BaoChi_CongNghe", "LLM"):
                cnt_gen += 1
        elif "rewrit" in gtype and cnt_rew < 650:
            if add_record(seg, 1, "DeckardLong-AI-Paraphrase", "AI_Paraphrase", "LLM"):
                cnt_rew += 1
        if cnt_gen >= 650 and cnt_rew >= 650:
            break
    print(f"    -> Đã lấy: {cnt_gen} AI Generated + {cnt_rew} AI Rewritten/Paraphrase")

    # 3. AI Học thuật / Giải thích kiến thức — BKAI ShareGPT Vietnamese -> 730 mẫu
    print("[A3] Đang thu thập 730 mẫu AI Học thuật / Phân tích từ BKAI ShareGPT...")
    ds_bkai = load_dataset("bkai-foundation-models/vi-self-chat-sharegpt-format", split="train", streaming=True)
    cnt = 0
    for row in ds_bkai:
        convs = row.get("conversations") or []
        for turn in convs:
            if turn.get("from") in ("gpt", "assistant"):
                val = str(turn.get("value", ""))
                if "```" in val or "def " in val or "import " in val:
                    continue
                t = strip_markdown_lightly(val)
                if not has_good_vietnamese(t) or len(t.split()) < 75 or POST_2022_YEAR_ONLY.search(t):
                    continue
                seg = slice_by_sentences(t, min_words=70, max_words=340, vary_length=True)
                if seg and add_record(seg, 1, "BKAI-ShareGPT-VI", "AI_HocThuat_GiaiThich", "LLM"):
                    cnt += 1
                    break
        if cnt >= 730:
            break
    print(f"    -> Đã lấy: {cnt} mẫu (AI_HocThuat_GiaiThich)")

    # 4. AI Văn học / Tiểu thuyết (Gemma-3-12B + Sailor2-8B) -> 500 mẫu
    print("[A4] Đang thu thập 500 mẫu AI Sáng tác Văn học (Gemma-3-12B & Sailor2-8B)...")
    cnt_lit_ai = 0
    try:
        api = HfApi()
        repo_id = "trieunh/Vietnamese.Ai.Human.Literary.works.VuTrongPhung"
        parquet_files = [f for f in api.list_repo_files(repo_id, repo_type="dataset") if f.endswith(".parquet")]
        per_file_cap = max(150, 500 // max(len(parquet_files), 1) + 50)
        for pf in parquet_files:
            if cnt_lit_ai >= 500:
                break
            url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{pf}"
            ds_pf = load_dataset("parquet", data_files=url, split="train", streaming=True)
            model_tag = pf.split("-VuTrongPhung")[0]
            taken = 0
            for row in ds_pf:
                gen_col = next((k for k in row.keys() if "generated_text" in k), None)
                if not gen_col:
                    continue
                raw_ai = str(row.get(gen_col, ""))
                # Bỏ dòng tiêu đề markdown kiểu "**Tiểu thuyết: ...**"
                cleaned = strip_markdown_lightly(raw_ai)
                cleaned = re.sub(r"^(Tiểu thuyết|Chương \d+|Truyện ngắn)\s*:?[^.!?]{0,60}", "", cleaned).strip()
                if not has_good_vietnamese(cleaned):
                    continue
                seg = slice_by_sentences(cleaned, min_words=65, max_words=340, vary_length=True)
                if seg and add_record(seg, 1, f"Lit-AI ({model_tag})", "AI_VanHoc_Truyen", "LLM"):
                    cnt_lit_ai += 1
                    taken += 1
                    if taken >= per_file_cap or cnt_lit_ai >= 500:
                        break
    except Exception as e:
        print(f"    [!] Cảnh báo tải AI Văn học: {e}")
    print(f"    -> Đã lấy: {cnt_lit_ai} mẫu (AI_VanHoc_Truyen)")

    # 5. Bù phần còn lại để cân bằng ĐÚNG 50/50 với Human từ kngann2201/vietnamese-ai-detection (label == 1)
    ai_so_far = sum(1 for r in records if r["label"] == 1)
    needed_ai = max(0, human_total - ai_so_far)
    print(f"[A5] Đang thu thập {needed_ai} mẫu AI Tiểu luận / Đời sống từ kngann2201 để cân bằng 50/50...")
    ds_kng = load_dataset("kngann2201/vietnamese-ai-detection", split="train", streaming=True)
    cnt = 0
    for row in ds_kng:
        if int(row.get("label", 0)) != 1:
            continue
        t = clean_prose(str(row.get("text", "")))
        if not has_good_vietnamese(t) or POST_2022_YEAR_ONLY.search(t):
            continue
        seg = slice_by_sentences(t, min_words=65, max_words=340, vary_length=True)
        if seg and add_record(seg, 1, "kngann2201-AI", "AI_TieuLuan_TongHop", "LLM"):
            cnt += 1
            if cnt >= needed_ai:
                break
    print(f"    -> Đã lấy: {cnt} mẫu (AI_TieuLuan_TongHop)")

    # ==========================================================================
    # PHẦN C: ĐÓNG GÓI, PHÂN TẦNG THEO THỂ LOẠI & LƯU FILE PREVIEW
    # ==========================================================================
    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    df.insert(0, "id", [f"VN_PRE22_{i+1:05d}" for i in range(len(df))])

    # Chia tập phân tầng theo (label + genre) -> 70% Train / 15% Val / 15% Test
    strat_key = df["label"].astype(str) + "__" + df["genre"].astype(str)
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_SEED)
    train_val_idx, test_idx = next(sss_test.split(df, strat_key))
    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    strat_tv = df_train_val["label"].astype(str) + "__" + df_train_val["genre"].astype(str)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.17647, random_state=RANDOM_SEED)
    train_idx, val_idx = next(sss_val.split(df_train_val, strat_tv))
    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].reset_index(drop=True)

    out_files = {
        "vietnamese_pre2022_full.csv": df,
        "vietnamese_pre2022_train.csv": df_train,
        "vietnamese_pre2022_val.csv": df_val,
        "vietnamese_pre2022_test.csv": df_test,
    }
    for fname, sub_df in out_files.items():
        sub_df.to_csv(os.path.join(DATA_DIR, fname), index=False, encoding="utf-8")

    print("\n" + "=" * 78)
    print("  BÁO CÁO BỘ DỮ LIỆU TIẾNG VIỆT ĐA MIỀN TIỀN 2022 (SẴN SÀNG DUYỆT)")
    print("=" * 78)
    print(f"Tổng cộng : {len(df):,} mẫu | Human (<=2022): {(df['label']==0).sum():,} | AI (Multi-LLM): {(df['label']==1).sum():,}")
    print(f"Split     : Train={len(df_train):,} | Val={len(df_val):,} | Test={len(df_test):,}")
    print("\n--- PHÂN BỐ ĐỘ DÀI TỪ THEO NHÃN (KIỂM TRA CHỐNG LỆCH ĐỘ DÀI) ---")
    print(df.groupby("label_name")["word_count"].describe()[["count", "mean", "std", "min", "50%", "max"]].round(1))
    print("\n--- CHI TIẾT TỪNG THỂ LOẠI & NGUỒN ---")
    summary = df.groupby(["label_name", "genre", "era_verified"]).agg(
        so_mau=("id", "count"),
        tu_tb=("word_count", "mean"),
    ).reset_index()
    summary["tu_tb"] = summary["tu_tb"].round(1)
    print(summary.to_string(index=False))

    print("\n--- MẪU THỰC TẾ ĐẠI DIỆN TỪ MỖI THỂ LOẠI ---")
    for g, grp in df.groupby("genre"):
        row = grp.iloc[0]
        preview = str(row["text"]).replace("\n", " / ")[:190]
        print(f"  [{row['label_name']:<12} | {g:<22} | {row['era_verified']:<6} | {row['word_count']:>3} từ] {preview}...")
    print("=" * 78)


if __name__ == "__main__":
    build_pre2022_dataset()
