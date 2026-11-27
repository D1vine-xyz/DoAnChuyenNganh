"""
Module Xây Dựng Bộ Dữ Liệu Tiếng Việt v3 (12.000 mẫu — 50/50) Chuẩn Tiền ChatGPT (<= 11/2022)
Đồ án Chuyên ngành Kỹ thuật Phần mềm (Chủ đề 6: VeriText AI)

Giải quyết trực diện 3 điểm yếu thực chiến phát hiện qua Stress-Test:
1. Lệch chủ đề Học thuật / CNTT (Topic Confounding):
   + Bổ sung 1.400 mẫu Human Pre-2022 (2010-2021) chuyên mục Công nghệ, Khoa học, Giáo dục từ BinhVQ21
   + Bổ sung 350 mẫu Bách khoa Khoa học/Kỹ thuật (UIT-ViQuAD 2.0 - VLSP 2021)
   + Bổ sung 250 mẫu Đời sống & Đoạn văn ngắn (25-85 từ) Pre-2022
2. Bị lọt AI viết giọng tự nhiên / sinh viên / không văn mẫu (Humanized AI):
   + Bổ sung 650 mẫu AI đa dạng (vi-alpaca) + 600 mẫu AI hội thoại tự nhiên (vi-self-chat)
   + Bổ sung 550 mẫu AI đời mới 2024-2025 (ICCIES-2025-DetectAI)
   + Bổ sung 200 mẫu AI Humanized giọng sinh viên đời thường sinh bằng Gemini 2.5 Flash Lite
3. Giữ nguyên trọn vẹn 8.000 mẫu v2 sạch đã kiểm chứng + phân tầng chuẩn 70/15/15:
   - Train: 8.400 mẫu | Val: 1.800 mẫu | Test: 1.800 mẫu
   Lưu tại: backend/data/vietnamese_v3_{train,val,test}.csv
"""

import os
import re
import sys
import json
import html
import time
import random
import unicodedata
from typing import List, Dict, Any

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from datasets import load_dataset
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

load_dotenv(os.path.join(BASE_DIR, "..", ".env"))
load_dotenv(os.path.join(BASE_DIR, ".env"))

RANDOM_SEED = 2026
random.seed(RANDOM_SEED)

VI_DIACRITICS = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")
POST_2022_YEAR_ONLY = re.compile(r"\b(2023|2024|2025|2026)\b")
POST_2022_TERMS = re.compile(r"\b(2023|2024|2025|2026|chatgpt|gpt-4|gpt-3\.5|gemini|claude|openai)\b", re.IGNORECASE)
TECH_SCI_KEYWORDS = re.compile(
    r"(thuật toán|hệ thống|công nghệ|nghiên cứu|khoa học|dữ liệu|mô hình|phần mềm|máy tính|"
    r"kỹ thuật|giáo dục|đại học|sinh viên|thực nghiệm|phương pháp|phân tích|trí tuệ nhân tạo|"
    r"ứng dụng|thiết kế|mạng|điện tử|sinh học|vật lý|hóa học|toán học|y học)",
    re.IGNORECASE,
)


def has_good_vietnamese(text: str, min_alpha: int = 30) -> bool:
    if not text:
        return False
    low = text.lower()
    vi_cnt = sum(1 for ch in low if ch in VI_DIACRITICS)
    alpha_cnt = sum(1 for ch in low if ch.isalpha())
    if alpha_cnt < min_alpha:
        return False
    return (vi_cnt / alpha_cnt) >= 0.10


def clean_html_prose(text: str) -> str:
    """Làm sạch HTML entities (&agrave;), thẻ <p>, URL, Markdown thừa về tiếng Việt NFC sạch."""
    if not text or not isinstance(text, str):
        return ""
    t = html.unescape(text)
    t = re.sub(r"```[\s\S]*?```", " ", t)
    t = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"</?(p|div|br|li|h[1-6]|tr)[^>]*>", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"https?://\S+|www\.\S+", "", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"\*([^*\n]+)\*", r"\1", t)
    t = re.sub(r"^#{1,6}\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"^\s*[-•*]\s+", "", t, flags=re.MULTILINE)
    t = unicodedata.normalize("NFC", t)
    t = t.replace("\ufeff", "").replace("\u200b", "").replace("\u00a0", " ")
    return re.sub(r"\s+", " ", t).strip()


def slice_natural(text: str, min_words: int = 35, max_words: int = 320, short_mode: bool = False) -> str:
    words = text.split()
    if len(words) < min_words:
        return ""
    if short_mode:
        target = random.randint(min_words, min(max_words, 90))
    else:
        r = random.random()
        if r < 0.22:
            target = random.randint(45, 110)
        elif r < 0.65:
            target = random.randint(115, 225)
        else:
            target = random.randint(230, max_words)

    if len(words) <= target:
        return text

    sents = re.split(r"(?<=[.!?…])\s+", text)
    acc, cnt = [], 0
    for s in sents:
        sw = len(s.split())
        if cnt + sw > target and cnt >= min_words:
            break
        acc.append(s)
        cnt += sw
        if cnt >= target:
            break
    out = " ".join(acc).strip()
    if len(out.split()) < min_words or len(out.split()) > max_words:
        out = " ".join(words[:target]).strip()
    return out


def generate_gemini_humanized_ai(target_count: int = 200) -> List[str]:
    """Sinh các mẫu AI giọng tự nhiên/sinh viên đời thường (không văn mẫu) bằng Gemini 3.5 Flash Lite."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or "your_" in api_key.lower():
        print("  [!] Không có GEMINI_API_KEY, bỏ qua nhánh Gemini Humanized.")
        return []

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-3.5-flash-lite")
    except Exception as e:
        print(f"  [!] Không khởi tạo được Gemini: {e}")
        return []

    prompt_themes = [
        ("nhật ký sinh viên kể chuyện đi học, làm đồ án, chạy deadline, uống cà phê hoặc trời mưa", "đời thường, gần gũi như blog cá nhân"),
        ("giải thích ngắn về lập trình, cơ sở dữ liệu, mạng máy tính, trí tuệ nhân tạo hoặc kiến trúc phần mềm", "kỹ sư viết blog chia sẻ kinh nghiệm thực tế, pha thuật ngữ tiếng Anh tự nhiên"),
        ("đoạn báo cáo thực tập, khóa luận hoặc tiểu luận môn học về kinh tế, giáo dục, công nghệ", "giọng sinh viên nghiêm túc, rõ ràng, tuyệt đối KHÔNG dùng 'không thể phủ nhận', 'đóng vai trò then chốt', 'tóm lại'"),
        ("bài review sách, phim, quán ăn, chuyến đi du lịch hoặc cảm nhận cuộc sống", "tự nhiên, có cảm xúc cá nhân, nhịp câu ngắn dài xen kẽ"),
        ("email hoặc tin nhắn ngắn (30-75 từ) sinh viên gửi giảng viên hỏi bài, nộp báo cáo hoặc nhận xét nhóm", "ngắn gọn, lịch sự, tự nhiên"),
        ("ý kiến thảo luận về làm việc từ xa, học trực tuyến, đọc sách, kỹ năng mềm hoặc quản lý thời gian", "lập luận mềm mại kiểu bài viết Spiderum, không khuôn mẫu"),
    ]

    generated: List[str] = []
    batch_size = 5
    max_calls = (target_count // batch_size) + 8
    print(f"  [*] Đang sinh ~{target_count} mẫu AI Humanized giọng tự nhiên bằng Gemini 3.5 Flash Lite...")

    for call_idx in range(max_calls):
        if len(generated) >= target_count:
            break
        theme, tone = prompt_themes[call_idx % len(prompt_themes)]
        is_short = "30-75 từ" in theme
        len_hint = "mỗi đoạn 30 đến 80 từ" if is_short else "mỗi đoạn 90 đến 230 từ"
        prompt = (
            f"Hãy viết {batch_size} đoạn văn tiếng Việt hoàn toàn độc lập về chủ đề: {theme}.\n"
            f"Phong cách bắt buộc: {tone}. Độ dài: {len_hint}.\n"
            f"QUY TẮC: Không dùng gạch đầu dòng, không dùng Markdown (** hay #), không nhắc các năm 2023-2026 hay tên ChatGPT/Gemini, "
            f"không dùng các cụm sáo rỗng như 'không thể phủ nhận', 'đóng vai trò then chốt', 'tóm lại', 'vô cùng cấp thiết'.\n"
            f"Trả về ĐÚNG 1 mảng JSON gồm {batch_size} chuỗi văn bản thuần: [\"đoạn 1\", \"đoạn 2\", ...]"
        )
        try:
            resp = model.generate_content(
                prompt,
                generation_config={"temperature": 0.95, "response_mime_type": "application/json"},
            )
            raw = (resp.text or "").strip()
            items = json.loads(raw)
            if isinstance(items, list):
                for item in items:
                    txt = clean_html_prose(str(item))
                    wc = len(txt.split())
                    if (25 if is_short else 55) <= wc <= 320 and has_good_vietnamese(txt, min_alpha=25):
                        if not POST_2022_TERMS.search(txt):
                            generated.append(txt)
            if (call_idx + 1) % 8 == 0:
                print(f"      -> Gemini đã sinh {len(generated)}/{target_count} mẫu...")
            time.sleep(0.25)
        except Exception as e:
            err_s = str(e)
            print(f"      [!] Gemini batch #{call_idx+1} lỗi: {err_s[:120]}")
            if "429" in err_s or "quota" in err_s.lower() or "resource_exhausted" in err_s.lower():
                time.sleep(2.5)
                if call_idx > 10 and len(generated) >= 40:
                    break
            else:
                time.sleep(0.8)

    print(f"  [v] Hoàn tất Gemini Humanized AI: {len(generated)} mẫu.")
    return generated[:target_count]


def build_vietnamese_v3():
    print("=" * 80)
    print("  XÂY DỰNG BỘ DỮ LIỆU TIẾNG VIỆT v3 (12.000 MẪU — ACADEMIC PRE-2022 + HUMANIZED AI)")
    print("=" * 80)

    records: List[Dict[str, Any]] = []
    seen_fp = set()

    def add_record(text: str, label: int, source: str, genre: str, era: str) -> bool:
        if not text:
            return False
        fp = re.sub(r"\W+", "", text[:180].lower())
        if len(fp) < 25 or fp in seen_fp:
            return False
        seen_fp.add(fp)
        records.append({
            "text": text,
            "label": int(label),
            "label_name": "Human" if label == 0 else "AI-generated",
            "source": source,
            "genre": genre,
            "era_verified": era,
            "word_count": len(text.split()),
            "char_count": len(text),
        })
        return True

    # ==========================================================================
    # BƯỚC 0: NẠP TRỌN VẸN 8.000 MẪU PRE-2022 v2 SẠCH ĐÃ KIỂM CHỨNG
    # ==========================================================================
    print("\n[0/6] Nạp nền 8.000 mẫu v2 Pre-2022 đã kiểm chứng...")
    v2_parts = [
        os.path.join(DATA_DIR, "vietnamese_pre2022_train.csv"),
        os.path.join(DATA_DIR, "vietnamese_pre2022_val.csv"),
        os.path.join(DATA_DIR, "vietnamese_pre2022_test.csv"),
    ]
    base_cnt = 0
    for p in v2_parts:
        df_p = pd.read_csv(p)
        for _, r in df_p.iterrows():
            if add_record(str(r["text"]), int(r["label"]), str(r["source"]), str(r["genre"]), str(r["era_verified"])):
                base_cnt += 1
    print(f"  [v] Đã giữ nguyên {base_cnt:,} mẫu v2 (Human={sum(1 for x in records if x['label']==0)}, AI={sum(1 for x in records if x['label']==1)})")

    # ==========================================================================
    # PHẦN A: BỔ SUNG +2.000 MẪU HUMAN THUẦN NGƯỜI TIỀN 2022 (HỌC THUẬT / CNTT / NGẮN)
    # ==========================================================================
    print("\n[1/6] Thu thập BinhVQ News 2021 (<= 2021) chuyên mục Công nghệ / Khoa học / Giáo dục / Đời sống...")
    targets_binhvq = {
        "CongNghe_KhoaHoc": 800,
        "GiaoDuc_HocThuat": 600,
        "DoiSong_Ngan": 250,
    }
    got_binhvq = {k: 0 for k in targets_binhvq}

    ds_bvq = load_dataset("imthanhlv/binhvq_news21_raw", split="train", streaming=True)
    scanned = 0
    for row in ds_bvq:
        scanned += 1
        if all(got_binhvq[k] >= targets_binhvq[k] for k in targets_binhvq):
            break
        if scanned > 120000:
            break

        # Kiểm tra cứng mốc thời gian xuất bản <= 2021
        pub = row.get("publish")
        pub_year = None
        if isinstance(pub, dict) and "$date" in pub:
            d = pub["$date"]
            pub_year = getattr(d, "year", None)
            if pub_year is None:
                m = re.search(r"(19|20)\d{2}", str(d))
                pub_year = int(m.group(0)) if m else None
        if pub_year is None or pub_year > 2021:
            continue

        cate = str(row.get("cate") or "").strip()
        if cate in ("Công nghệ", "Khoa học"):
            bucket = "CongNghe_KhoaHoc"
        elif cate == "Giáo dục":
            bucket = "GiaoDuc_HocThuat"
        elif cate in ("Đời sống", "Văn hóa", "Kinh tế"):
            bucket = "DoiSong_Ngan"
        else:
            continue

        if got_binhvq[bucket] >= targets_binhvq[bucket]:
            continue

        sapo = clean_html_prose(str(row.get("sapo") or ""))
        body = clean_html_prose(str(row.get("body") or ""))
        full = f"{sapo} {body}".strip() if sapo and body else (body or sapo)
        if not has_good_vietnamese(full, min_alpha=40) or POST_2022_YEAR_ONLY.search(full):
            continue

        is_short = (bucket == "DoiSong_Ngan") and (got_binhvq[bucket] < 150)
        sliced = slice_natural(full, min_words=28 if is_short else 60, max_words=85 if is_short else 320, short_mode=is_short)
        if not sliced:
            continue

        if add_record(sliced, 0, f"BinhVQ-News-{pub_year} ({cate})", bucket, f"<={pub_year}"):
            got_binhvq[bucket] += 1

    print(f"  [v] BinhVQ Pre-2022 đã thu (quét {scanned:,} bài): {got_binhvq}")

    print("\n[2/6] Bổ sung 350 mẫu Bách khoa Khoa học & Kỹ thuật mới (UIT-ViQuAD 2.0 - VLSP 2021)...")
    viquad_target = 350 + sum(max(0, targets_binhvq[k] - got_binhvq[k]) for k in targets_binhvq)
    viquad_got = 0
    ds_vq = load_dataset("taidng/UIT-ViQuAD2.0", split="train", streaming=True)
    for row in ds_vq:
        if viquad_got >= viquad_target:
            break
        ctx = clean_html_prose(str(row.get("context") or ""))
        if not has_good_vietnamese(ctx, min_alpha=50) or POST_2022_YEAR_ONLY.search(ctx):
            continue
        if not TECH_SCI_KEYWORDS.search(ctx):
            continue
        sliced = slice_natural(ctx, min_words=55, max_words=320)
        if add_record(sliced, 0, "UIT-ViQuAD2.0-SciTech (2021)", "CongNghe_KhoaHoc", "<=2021"):
            viquad_got += 1
    print(f"  [v] Đã bổ sung {viquad_got} mẫu Bách khoa Khoa học/Kỹ thuật (<=2021).")

    total_human_now = sum(1 for r in records if r["label"] == 0)
    print(f"  => TỔNG LỚP 0 (HUMAN PRE-2022): {total_human_now:,} mẫu")

    # ==========================================================================
    # PHẦN B: BỔ SUNG +2.000 MẪU AI ĐA THẾ HỆ & HUMANIZED (KHỚP ĐÚNG 50/50)
    # ==========================================================================
    ai_needed_total = total_human_now - sum(1 for r in records if r["label"] == 1)
    print(f"\n[*] Cần bổ sung chính xác {ai_needed_total:,} mẫu AI mới để cân bằng 50/50.")

    # 3. Gemini 2.5 Humanized Student/Casual/Tech AI (200 mẫu)
    print("\n[3/6] Sinh mẫu AI Humanized giọng sinh viên / kỹ thuật tự nhiên (Gemini 2.5)...")
    gem_texts = generate_gemini_humanized_ai(target_count=min(200, ai_needed_total))
    gem_got = 0
    for txt in gem_texts:
        if add_record(txt, 1, "Gemini-2.5-Humanized", "AI_Humanized_TuNhien", "LLM-2025"):
            gem_got += 1
    print(f"  [v] Đã thêm {gem_got} mẫu AI_Humanized_TuNhien (Gemini 2.5).")

    # 4. ICCIES-2025-DetectAI Vietnamese Frontier LLM (550 mẫu)
    print("\n[4/6] Thu thập 550 mẫu AI đời mới (ICCIES-2025-DetectAI/vietnamese_news_human_ai)...")
    ic_target = min(550, ai_needed_total - gem_got)
    ic_got = 0
    try:
        ds_ic = load_dataset("ICCIES-2025-DetectAI/vietnamese_news_human_ai", split="train", streaming=True)
        for row in ds_ic:
            if ic_got >= ic_target:
                break
            if str(row.get("Label", "")).strip() != "1":
                continue
            txt = clean_html_prose(str(row.get("Text") or ""))
            if not has_good_vietnamese(txt, min_alpha=45) or POST_2022_TERMS.search(txt):
                continue
            sliced = slice_natural(txt, min_words=55, max_words=320)
            if add_record(sliced, 1, "ICCIES-2025-AI", "AI_Frontier_2025", "LLM-2025"):
                ic_got += 1
    except Exception as e:
        print(f"  [!] Lỗi ICCIES-2025: {e}")
    print(f"  [v] Đã thêm {ic_got} mẫu AI_Frontier_2025.")

    # 5. BKAI vi-alpaca (650 mẫu, gồm cả đoạn ngắn 30-85 từ và blog/kỹ thuật)
    print("\n[5/6] Thu thập 650 mẫu AI đa dạng độ dài (bkai-foundation-models/vi-alpaca)...")
    alp_target = min(650, total_human_now - sum(1 for r in records if r["label"] == 1))
    alp_got = 0
    try:
        ds_alp = load_dataset("bkai-foundation-models/vi-alpaca", split="train", streaming=True)
        for row in ds_alp:
            if alp_got >= alp_target:
                break
            out_txt = clean_html_prose(str(row.get("output") or ""))
            if not has_good_vietnamese(out_txt, min_alpha=30) or POST_2022_TERMS.search(out_txt):
                continue
            is_short = alp_got < 140
            sliced = slice_natural(out_txt, min_words=28 if is_short else 55, max_words=85 if is_short else 320, short_mode=is_short)
            if add_record(sliced, 1, "BKAI-vi-alpaca", "AI_Alpaca_DaDang", "LLM"):
                alp_got += 1
    except Exception as e:
        print(f"  [!] Lỗi vi-alpaca: {e}")
    print(f"  [v] Đã thêm {alp_got} mẫu AI_Alpaca_DaDang.")

    # 6. BKAI vi-self-chat-sharegpt-format (bù chính xác phần còn lại tới 50/50)
    rem_ai = total_human_now - sum(1 for r in records if r["label"] == 1)
    print(f"\n[6/6] Thu thập {rem_ai} mẫu AI hội thoại tự nhiên (bkai-foundation-models/vi-self-chat-sharegpt-format)...")
    sc_got = 0
    if rem_ai > 0:
        ds_sc = load_dataset("bkai-foundation-models/vi-self-chat-sharegpt-format", split="train", streaming=True)
        for row in ds_sc:
            if sc_got >= rem_ai:
                break
            convs = row.get("conversations") or []
            if isinstance(convs, str):
                try:
                    convs = json.loads(convs)
                except Exception:
                    continue
            for turn in convs:
                if sc_got >= rem_ai:
                    break
                if not isinstance(turn, dict) or turn.get("from") not in ("gpt", "assistant"):
                    continue
                txt = clean_html_prose(str(turn.get("value") or ""))
                if not has_good_vietnamese(txt, min_alpha=35) or POST_2022_TERMS.search(txt):
                    continue
                is_short = sc_got < 120
                sliced = slice_natural(txt, min_words=28 if is_short else 55, max_words=85 if is_short else 320, short_mode=is_short)
                if add_record(sliced, 1, "BKAI-SelfChat-VI", "AI_HoiThoai_TuNhien", "LLM"):
                    sc_got += 1
    print(f"  [v] Đã thêm {sc_got} mẫu AI_HoiThoai_TuNhien.")

    # ==========================================================================
    # CÂN BẰNG 50/50, XÁO TRỘN & CHIA PHÂN TẦNG 70 / 15 / 15
    # ==========================================================================
    df = pd.DataFrame(records)
    df_h = df[df["label"] == 0]
    df_a = df[df["label"] == 1]
    n_each = min(len(df_h), len(df_a))
    df = pd.concat([
        df_h.sample(n=n_each, random_state=RANDOM_SEED),
        df_a.sample(n=n_each, random_state=RANDOM_SEED),
    ], ignore_index=True).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    df.insert(0, "id", [f"VN_V3_{i+1:05d}" for i in range(len(df))])

    # Chia Train 70% / Val 15% / Test 15% phân tầng theo (label + genre)
    strat_key = df["label"].astype(str) + "_" + df["genre"].astype(str)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_SEED)
    train_idx, temp_idx = next(sss1.split(df, strat_key))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_temp = df.iloc[temp_idx].reset_index(drop=True)

    strat_temp = df_temp["label"].astype(str) + "_" + df_temp["genre"].astype(str)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_SEED)
    val_idx, test_idx = next(sss2.split(df_temp, strat_temp))
    df_val = df_temp.iloc[val_idx].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)

    train_path = os.path.join(DATA_DIR, "vietnamese_v3_train.csv")
    val_path = os.path.join(DATA_DIR, "vietnamese_v3_val.csv")
    test_path = os.path.join(DATA_DIR, "vietnamese_v3_test.csv")
    df_train.to_csv(train_path, index=False, encoding="utf-8-sig")
    df_val.to_csv(val_path, index=False, encoding="utf-8-sig")
    df_test.to_csv(test_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print(f"  HOÀN TẤT BỘ DỮ LIỆU TIẾNG VIỆT v3: {len(df):,} MẪU (50/50 = {n_each:,} Human + {n_each:,} AI)")
    print(f"  - Train : {len(df_train):,} -> {train_path}")
    print(f"  - Val   : {len(df_val):,} -> {val_path}")
    print(f"  - Test  : {len(df_test):,} -> {test_path}")
    print("=" * 80)
    print("\nPhân bố chi tiết theo Thể loại (Genre):")
    summary = df.groupby(["label_name", "genre"]).agg(
        count=("id", "count"),
        avg_words=("word_count", "mean"),
        min_words=("word_count", "min"),
        max_words=("word_count", "max"),
    )
    summary["avg_words"] = summary["avg_words"].round(1)
    print(summary.to_string())


if __name__ == "__main__":
    build_vietnamese_v3()
