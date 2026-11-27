"""
Nâng cấp Bộ Dữ Liệu Tiếng Việt v3 -> v3.1 (15.000 mẫu — 50/50 = 7.500 Human Pre-2022 + 7.500 AI)
Giải quyết triệt để điểm mù "Blog ngôi thứ nhất / Kỹ thuật CNTT / Học thuật .edu.vn" phát hiện qua
kiểm chuẩn 100 văn bản Người viết thật Pre-2020 trên C4-vi Shard 0:
- Lấy nguồn Human Pre-2020 ĐỘC LẬP từ allenai/c4 (vi) SHARD 1 (c4-vi.tfrecord-00001-of-01024.json.gz),
  tuyệt đối KHÔNG chạm vào SHARD 0 để giữ nguyên tập kiểm chuẩn Unseen Real-Human độc lập!
- Bổ sung +1.500 mẫu Human Pre-2020 sạch (lọc bỏ navbar/menu/spam/quảng cáo):
  + 600 mẫu Blog cá nhân / Tự thuật ngôi thứ nhất / Review đời thường (mình, em, tôi)
  + 500 mẫu Kỹ thuật CNTT / Phần mềm / Công nghệ (tinhte, quantrimang, viblo, freetuts, genk, ...)
  + 400 mẫu Học thuật / Giáo dục / Báo chí chất lượng (.edu.vn, tailieu, vnexpress, tuoitre, ...)
- Bổ sung +1.500 mẫu AI cân bằng (ICCIES-2025 + vi-alpaca + vi-self-chat + 100 Gemini 3.5 Humanized đa miền)
- Giữ trọn 12.000 mẫu v3 đã kiểm chứng -> Tổng 15.000 mẫu (Train 10.500 / Val 2.250 / Test 2.250).
"""

import os
import re
import sys
import gzip
import json
import random
import unicodedata
from typing import List, Dict, Any

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from huggingface_hub import hf_hub_download
from datasets import load_dataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
sys.path.insert(0, BASE_DIR)

from build_vietnamese_v3_dataset import (
    has_good_vietnamese,
    clean_html_prose,
    slice_natural,
    generate_gemini_humanized_ai,
    POST_2022_YEAR_ONLY,
    POST_2022_TERMS,
)

RANDOM_SEED = 2026
random.seed(RANDOM_SEED)

SPAM_URL_RE = re.compile(
    r"(xsmt|xsmb|xsmn|soi-?cau|lo-?de|kubet|fun88|w88|188bet|alibaba|shopee|lazada|tiki|sendo|"
    r"batdongsan|nhadat|muaban|chotot|timviec|vieclam|simso|phongthuy|tuvi|boitoan|sex|porn|xxx)",
    re.I,
)
NAV_NOISE_RE = re.compile(
    r"(trang chủ|đăng nhập|đăng ký|liên hệ|bản quyền|copyright|menu|giỏ hàng|hotline|"
    r"xem thêm|bình luận|chia sẻ bài viết|you are here|home\s*/|thảo luận trong)",
    re.I,
)
IT_URL_HINTS = (
    "quantrimang", "tinhte.vn", "genk.vn", "viblo", "freetuts", "vnreview",
    "kipalog", "topdev", "techtalk", "toidicode", "daynhauhoc", "cafedev",
    "howkteam", "kungfutech", "stdio.vn", "techrum", "ictnews", "sohoa", "congnghe",
)
EDU_URL_HINTS = (
    ".edu.vn", "tailieu.vn", "luanvan", "hocmai", "vietjack", "loigiaihay",
    "123doc", "vnexpress.net", "tuoitre.vn", "thanhnien.vn", "dantri.com.vn",
    "vietnamnet.vn", "nhandan", "baochinhphu", "vtv.vn",
)
BLOG_URL_HINTS = (
    "wordpress.com", "blogspot.com", "spiderum.com", "webtretho.com",
    "otofun", "voz", "foody.vn", "tripadvisor", "goodreads", "blog", "tamsu", "review",
)
IT_KW_RE = re.compile(
    r"\b(lập trình|cơ sở dữ liệu|thuật toán|phần mềm|hệ thống|mạng máy tính|máy chủ|trí tuệ nhân tạo|"
    r"ứng dụng|giao diện|bảo mật|mã nguồn|trình duyệt|điện thoại|máy tính|bộ vi xử lý|python|java|sql|linux|windows|android|ios)\b",
    re.I,
)
FIRST_PERSON_RE = re.compile(r"\b(mình|tụi mình|bọn mình|nhóm mình|chúng em|em|tôi|chúng tôi)\b", re.I)


def extract_clean_c4_prose(raw_text: str) -> str:
    """Lọc bỏ dòng tiêu đề/navbar/footer, chỉ giữ các đoạn văn xuôi hoàn chỉnh sạch."""
    if not raw_text:
        return ""
    good_paras = []
    lines = [l.strip() for l in raw_text.split("\n")]
    # Bỏ dòng đầu và dòng cuối (thường là tiêu đề site / breadcrumb / footer)
    core_lines = lines[1:-1] if len(lines) >= 4 else lines
    for ln in core_lines:
        if not ln:
            continue
        if NAV_NOISE_RE.search(ln):
            continue
        if "|" in ln or "»" in ln or "›" in ln or ">>>" in ln or "http://" in ln or "https://" in ln:
            continue
        words = ln.split()
        if len(words) < 15:
            continue
        # Phải kết thúc bằng dấu câu tự nhiên (câu văn xuôi, không phải danh mục/menu)
        if ln[-1] not in ".!?…\"”')":
            continue
        # Loại dòng nhiều chữ IN HOA bất thường (banner quảng cáo)
        upper_ratio = sum(1 for c in ln if c.isupper()) / max(sum(1 for c in ln if c.isalpha()), 1)
        if upper_ratio > 0.25:
            continue
        good_paras.append(ln)

    if len(good_paras) < 2:
        return ""

    merged = clean_html_prose(" ".join(good_paras))
    if not has_good_vietnamese(merged, min_alpha=50):
        return ""
    if POST_2022_YEAR_ONLY.search(merged):
        return ""
    return slice_natural(merged, min_words=60, max_words=310)


def upgrade_v31():
    print("=" * 80)
    print("  NÂNG CẤP TIẾNG VIỆT v3 -> v3.1 (15.000 MẪU: BỔ SUNG C4-VI SHARD 1 PRE-2020)")
    print("=" * 80)

    # 0. Nạp 12.000 mẫu v3 hiện có
    dfs = [pd.read_csv(os.path.join(DATA_DIR, f"vietnamese_v3_{sp}.csv")) for sp in ("train", "val", "test")]
    df_base = pd.concat(dfs, ignore_index=True).drop(columns=["id"], errors="ignore")
    records: List[Dict[str, Any]] = df_base.to_dict(orient="records")
    seen_fp = set(re.sub(r"\W+", "", str(r["text"])[:180].lower()) for r in records)
    print(f"[0/4] Nạp nền v3: {len(records):,} mẫu (Human={sum(1 for r in records if r['label']==0)}, AI={sum(1 for r in records if r['label']==1)})")

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

    # 1. Tải C4-vi SHARD 1 (Pre-2020, độc lập hoàn toàn với Shard 0 dùng để benchmark!)
    print("\n[1/4] Tải & lọc +1.500 mẫu Human Pre-2020 từ allenai/c4 (vi) SHARD 1...")
    shard_path = hf_hub_download("allenai/c4", "multilingual/c4-vi.tfrecord-00001-of-01024.json.gz", repo_type="dataset")
    targets = {
        "Blog_TuThuat_Ngoi1": 600,
        "CNTT_KyThuat_ThucTe": 500,
        "HocThuat_Web_Pre2020": 400,
    }
    got = {k: 0 for k in targets}
    scanned = 0
    with gzip.open(shard_path, "rt", encoding="utf-8") as f:
        for line in f:
            scanned += 1
            if all(got[k] >= targets[k] for k in targets):
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            url = (obj.get("url") or "").lower()
            ts = (obj.get("timestamp") or "")[:4]
            if not ts.isdigit() or int(ts) > 2020:
                continue
            if SPAM_URL_RE.search(url):
                continue

            # Xác định nhóm mục tiêu ưu tiên
            is_it_url = any(h in url for h in IT_URL_HINTS)
            is_edu_url = any(h in url for h in EDU_URL_HINTS)
            is_blog_url = any(h in url for h in BLOG_URL_HINTS)

            if not (is_it_url or is_edu_url or is_blog_url):
                # Vẫn cho phép trang web sạch nếu cần đủ Blog ngôi 1 hoặc IT
                if got["Blog_TuThuat_Ngoi1"] >= targets["Blog_TuThuat_Ngoi1"] and got["CNTT_KyThuat_ThucTe"] >= targets["CNTT_KyThuat_ThucTe"]:
                    continue

            prose = extract_clean_c4_prose(obj.get("text") or "")
            if not prose:
                continue

            fp_cnt = len(FIRST_PERSON_RE.findall(prose))
            it_cnt = len(IT_KW_RE.findall(prose))

            bucket = None
            if got["CNTT_KyThuat_ThucTe"] < targets["CNTT_KyThuat_ThucTe"] and (is_it_url or it_cnt >= 4):
                bucket = "CNTT_KyThuat_ThucTe"
            elif got["Blog_TuThuat_Ngoi1"] < targets["Blog_TuThuat_Ngoi1"] and ((is_blog_url and fp_cnt >= 2) or fp_cnt >= 4):
                bucket = "Blog_TuThuat_Ngoi1"
            elif got["HocThuat_Web_Pre2020"] < targets["HocThuat_Web_Pre2020"] and is_edu_url:
                bucket = "HocThuat_Web_Pre2020"

            if bucket and add_record(prose, 0, f"C4-VI-Shard1-{ts}", bucket, f"<={ts}"):
                got[bucket] += 1

    print(f"  [v] C4-vi Shard 1 (quét {scanned:,} tài liệu <=2020): {got}")
    total_human = sum(1 for r in records if r["label"] == 0)
    ai_needed = total_human - sum(1 for r in records if r["label"] == 1)
    print(f"  => Tổng Human Pre-2022: {total_human:,} | Cần bổ sung chính xác {ai_needed:,} mẫu AI mới.")

    # 2. Bổ sung 100 mẫu Gemini 3.5 Humanized AI bổ sung
    print("\n[2/4] Sinh thêm 100 mẫu AI Humanized giọng tự nhiên (Gemini 3.5)...")
    gem_new = generate_gemini_humanized_ai(target_count=min(100, ai_needed))
    g_cnt = 0
    for txt in gem_new:
        if add_record(txt, 1, "Gemini-3.5-Humanized", "AI_Humanized_TuNhien", "LLM-2026"):
            g_cnt += 1
    print(f"  [v] Đã thêm {g_cnt} mẫu AI_Humanized_TuNhien.")

    # 3. Bổ sung AI Frontier 2025 (ICCIES-2025) + BKAI vi-alpaca + BKAI vi-self-chat để khớp 50/50
    print("\n[3/4] Thu thập AI đa nguồn (ICCIES-2025 + vi-alpaca + vi-self-chat) để cân bằng 50/50...")
    ic_target = min(600, total_human - sum(1 for r in records if r["label"] == 1))
    ic_got = 0
    ds_ic = load_dataset("ICCIES-2025-DetectAI/vietnamese_news_human_ai", split="train", streaming=True)
    for row in ds_ic:
        if ic_got >= ic_target:
            break
        if str(row.get("Label", "")).strip() != "1":
            continue
        txt = clean_html_prose(str(row.get("Text") or ""))
        if not has_good_vietnamese(txt, min_alpha=45) or POST_2022_TERMS.search(txt):
            continue
        sliced = slice_natural(txt, min_words=55, max_words=310)
        if add_record(sliced, 1, "ICCIES-2025-AI", "AI_Frontier_2025", "LLM-2025"):
            ic_got += 1
    print(f"  [v] Đã thêm {ic_got} mẫu ICCIES-2025 AI.")

    alp_target = min(500, total_human - sum(1 for r in records if r["label"] == 1))
    alp_got = 0
    ds_alp = load_dataset("bkai-foundation-models/vi-alpaca", split="train", streaming=True)
    for row in ds_alp:
        if alp_got >= alp_target:
            break
        txt = clean_html_prose(str(row.get("output") or ""))
        if not has_good_vietnamese(txt, min_alpha=35) or POST_2022_TERMS.search(txt):
            continue
        sliced = slice_natural(txt, min_words=50, max_words=310)
        if add_record(sliced, 1, "BKAI-vi-alpaca", "AI_Alpaca_DaDang", "LLM"):
            alp_got += 1
    print(f"  [v] Đã thêm {alp_got} mẫu vi-alpaca AI.")

    rem_ai = total_human - sum(1 for r in records if r["label"] == 1)
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
                sliced = slice_natural(txt, min_words=45, max_words=310)
                if add_record(sliced, 1, "BKAI-SelfChat-VI", "AI_HoiThoai_TuNhien", "LLM"):
                    sc_got += 1
    print(f"  [v] Đã thêm {sc_got} mẫu vi-self-chat AI.")

    # 4. Cân bằng chính xác 50/50 & chia phân tầng 70 / 15 / 15
    print("\n[4/4] Cân bằng 50/50 & xuất bộ dữ liệu Tiếng Việt v3.1 (15.000 mẫu)...")
    df = pd.DataFrame(records)
    df_h = df[df["label"] == 0]
    df_a = df[df["label"] == 1]
    n_each = min(len(df_h), len(df_a))
    df = pd.concat([
        df_h.sample(n=n_each, random_state=RANDOM_SEED),
        df_a.sample(n=n_each, random_state=RANDOM_SEED),
    ], ignore_index=True).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    df.insert(0, "id", [f"VN_V31_{i+1:05d}" for i in range(len(df))])

    strat_key = df["label"].astype(str) + "_" + df["genre"].astype(str)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_SEED)
    tr_idx, tmp_idx = next(sss1.split(df, strat_key))
    df_train = df.iloc[tr_idx].reset_index(drop=True)
    df_tmp = df.iloc[tmp_idx].reset_index(drop=True)

    strat_tmp = df_tmp["label"].astype(str) + "_" + df_tmp["genre"].astype(str)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_SEED)
    va_idx, te_idx = next(sss2.split(df_tmp, strat_tmp))
    df_val = df_tmp.iloc[va_idx].reset_index(drop=True)
    df_test = df_tmp.iloc[te_idx].reset_index(drop=True)

    df_train.to_csv(os.path.join(DATA_DIR, "vietnamese_v3_train.csv"), index=False, encoding="utf-8-sig")
    df_val.to_csv(os.path.join(DATA_DIR, "vietnamese_v3_val.csv"), index=False, encoding="utf-8-sig")
    df_test.to_csv(os.path.join(DATA_DIR, "vietnamese_v3_test.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print(f"  HOÀN TẤT TIẾNG VIỆT v3.1: {len(df):,} MẪU ({n_each:,} Human Pre-2022 + {n_each:,} AI)")
    print(f"  - Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")
    print("=" * 80)
    print(df.groupby(["label_name", "genre"]).agg(count=("id", "count"), avg_words=("word_count", "mean")).round(1).to_string())


if __name__ == "__main__":
    upgrade_v31()
