"""
Module Quản lý Cơ sở Dữ liệu Lịch sử Thẩm định Văn bản (SQLite Database)
Đồ án Chuyên ngành Kỹ thuật Phần mềm: Hệ thống Phát hiện Văn bản do LLM sinh (Ensemble PLMs)
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from contextlib import contextmanager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(BASE_DIR, "output")):
    DB_DIR = os.path.join(BASE_DIR, "output")
elif os.path.exists(os.path.join(BASE_DIR, "..", "output")):
    DB_DIR = os.path.join(BASE_DIR, "..", "output")
else:
    DB_DIR = os.path.join(BASE_DIR, "output")

DB_PATH = os.path.join(DB_DIR, "scan_history.db")


@contextmanager
def get_connection():
    """Khởi tạo kết nối SQLite với row_factory dạng Row và tự động đóng kết nối khi hoàn tất."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Khởi tạo cấu trúc bảng CSDL nếu chưa tồn tại."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                title TEXT,
                filename TEXT,
                text_snippet TEXT,
                full_text TEXT,
                word_count INTEGER,
                char_count INTEGER,
                ai_probability REAL,
                verdict TEXT,
                confidence TEXT,
                inference_time_ms REAL,
                score_bert REAL,
                score_roberta REAL,
                score_longformer REAL,
                details_json TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON scans (timestamp DESC)")
        conn.commit()


def save_scan(
    title: str,
    full_text: str,
    word_count: int,
    char_count: int,
    ai_probability: float,
    verdict: str,
    confidence: str,
    inference_time_ms: float,
    score_bert: float,
    score_roberta: float,
    score_longformer: float,
    details_json: Dict[str, Any],
    filename: Optional[str] = None,
) -> int:
    """Lưu trữ một lượt quét mới vào CSDL."""
    snippet = full_text[:180] + ("..." if len(full_text) > 180 else "")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO scans (
                title, filename, text_snippet, full_text, word_count, char_count,
                ai_probability, verdict, confidence, inference_time_ms,
                score_bert, score_roberta, score_longformer, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            title or "Văn bản không tên",
            filename or "",
            snippet,
            full_text,
            word_count,
            char_count,
            round(ai_probability, 4),
            verdict,
            confidence,
            round(inference_time_ms, 2),
            round(score_bert, 4),
            round(score_roberta, 4),
            round(score_longformer, 4),
            json.dumps(details_json, ensure_ascii=False),
        ))
        conn.commit()
        return cursor.lastrowid


def get_history(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """Truy xuất danh sách lịch sử thẩm định."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, title, filename, text_snippet, word_count, char_count,
                   ai_probability, verdict, confidence, inference_time_ms,
                   score_bert, score_roberta, score_longformer
            FROM scans
            ORDER BY id DESC
            LIMIT ? OFFSET ?
        """, (limit, offset))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_scan_by_id(scan_id: int) -> Optional[Dict[str, Any]]:
    """Truy xuất chi tiết một lượt quét theo ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM scans WHERE id = ?", (scan_id,))
        row = cursor.fetchone()
        if not row:
            return None
        res = dict(row)
        if res.get("details_json"):
            try:
                res["details"] = json.loads(res["details_json"])
            except Exception:
                res["details"] = {}
        return res


def delete_scan(scan_id: int) -> bool:
    """Xóa một lượt quét khỏi CSDL."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
        conn.commit()
        return cursor.rowcount > 0


def get_statistics() -> Dict[str, Any]:
    """Tính toán thống kê tổng quan các lượt quét trên Dashboard."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS total FROM scans")
        total = cursor.fetchone()["total"]

        if total == 0:
            return {
                "total_scans": 0,
                "ai_count": 0,
                "human_count": 0,
                "ai_percentage": 0.0,
                "avg_inference_time_ms": 0.0,
                "total_words_scanned": 0,
            }

        cursor.execute("SELECT COUNT(*) AS ai_cnt FROM scans WHERE verdict = 'AI-generated'")
        ai_cnt = cursor.fetchone()["ai_cnt"]

        cursor.execute("SELECT AVG(inference_time_ms) AS avg_time, SUM(word_count) AS total_words FROM scans")
        stats = cursor.fetchone()

        return {
            "total_scans": total,
            "ai_count": ai_cnt,
            "human_count": total - ai_cnt,
            "ai_percentage": round((ai_cnt / total) * 100, 1),
            "avg_inference_time_ms": round(stats["avg_time"] or 0, 1),
            "total_words_scanned": stats["total_words"] or 0,
        }


# Tự động khởi tạo CSDL khi import
init_db()
