"""
Module Quản lý Cơ sở Dữ liệu Microsoft SQL Server (Database: VeriTextAI)
Đồ án Chuyên ngành Kỹ thuật Phần mềm: Hệ thống Phát hiện Văn bản do LLM sinh (Ensemble PLMs)
Hỗ trợ:
- Xác thực người dùng, phân quyền Giảng viên & Sinh viên (Role-based access).
- Lưu trữ lịch sử thẩm định chi tiết theo từng người dùng (User-specific audit logs).
- Quản lý các lô chấm bài tập hàng loạt của lớp học (Classroom batches).
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
import pyodbc
import bcrypt

# Đảm bảo in tiếng Việt UTF-8 chuẩn trên Windows Console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ODBC_DRIVERS = [
    "ODBC Driver 17 for SQL Server",
    "ODBC Driver 18 for SQL Server",
    "SQL Server",
]

def get_connection_string() -> str:
    """Tự động phát hiện driver ODBC khả dụng và tạo connection string."""
    available_drivers = pyodbc.drivers()
    selected_driver = "ODBC Driver 17 for SQL Server"
    for d in ODBC_DRIVERS:
        if d in available_drivers:
            selected_driver = d
            break

    server = os.environ.get("DB_SERVER", "localhost")
    database = os.environ.get("DB_NAME", "VeriTextAI")
    trust_cert = ";TrustServerCertificate=yes" if "18" in selected_driver else ""
    return f"DRIVER={{{selected_driver}}};SERVER={server};DATABASE={database};Trusted_Connection=yes{trust_cert};"

DB_CONNECTION_STRING = get_connection_string()
_db_initialized = False


@contextmanager
def get_connection():
    """Khởi tạo kết nối tới Microsoft SQL Server và tự động đóng khi hoàn tất."""
    conn = pyodbc.connect(DB_CONNECTION_STRING, autocommit=False)
    try:
        yield conn
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def init_db(force: bool = False):
    """Khởi tạo cấu trúc bảng CSDL và nâng cấp schema an toàn trong VeriTextAI."""
    global _db_initialized
    if _db_initialized and not force:
        return

    with get_connection() as conn:
        cursor = conn.cursor()

        # 1. Bảng Người dùng (users)
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='users' AND xtype='U')
            BEGIN
                CREATE TABLE users (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    email NVARCHAR(255) NOT NULL UNIQUE,
                    password_hash NVARCHAR(255) NOT NULL,
                    full_name NVARCHAR(255) NOT NULL,
                    role NVARCHAR(50) NOT NULL DEFAULT 'student',
                    is_active BIT NOT NULL DEFAULT 1,
                    created_at DATETIME2 DEFAULT GETDATE()
                );
            END;
        """)

        # 2. Bảng Lịch sử thẩm định (detection_history)
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='detection_history' AND xtype='U')
            BEGIN
                CREATE TABLE detection_history (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    user_id INT NULL FOREIGN KEY REFERENCES users(id) ON DELETE SET NULL,
                    title NVARCHAR(255) NULL,
                    text_snippet NVARCHAR(MAX) NOT NULL,
                    full_text NVARCHAR(MAX) NULL,
                    language NVARCHAR(10) NOT NULL DEFAULT 'en',
                    label NVARCHAR(50) NOT NULL,
                    ai_probability FLOAT NOT NULL,
                    human_probability FLOAT NOT NULL,
                    confidence NVARCHAR(50) NOT NULL,
                    word_count INT NOT NULL,
                    char_count INT NOT NULL,
                    sentence_count INT NOT NULL,
                    model_breakdown NVARCHAR(MAX) NULL,
                    sentences_data NVARCHAR(MAX) NULL,
                    perplexity_score FLOAT NULL,
                    burstiness_score FLOAT NULL,
                    has_bypass_warning BIT NOT NULL DEFAULT 0,
                    bypass_details NVARCHAR(MAX) NULL,
                    inference_time_ms FLOAT NOT NULL,
                    created_at DATETIME2 DEFAULT GETDATE()
                );
            END;
        """)

        # Bổ sung các cột nâng cấp (full_text, authorship_tier, xai_metrics) và index nếu bảng đã tạo từ trước
        for col_name, col_def in [
            ("full_text", "NVARCHAR(MAX) NULL"),
            ("authorship_tier", "NVARCHAR(60) NULL"),
            ("xai_metrics", "NVARCHAR(MAX) NULL"),
        ]:
            cursor.execute(f"""
                IF COL_LENGTH('detection_history', '{col_name}') IS NULL
                BEGIN
                    ALTER TABLE detection_history ADD {col_name} {col_def};
                END;
            """)
        cursor.execute("""
            IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_detection_history_user_id' AND object_id = OBJECT_ID('detection_history'))
            BEGIN
                CREATE INDEX IX_detection_history_user_id ON detection_history(user_id);
            END;
        """)

        # 3. Bảng Lô bài tập lớp học (classroom_batches)
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='classroom_batches' AND xtype='U')
            BEGIN
                CREATE TABLE classroom_batches (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    teacher_id INT NOT NULL FOREIGN KEY REFERENCES users(id) ON DELETE CASCADE,
                    batch_name NVARCHAR(255) NOT NULL,
                    total_files INT NOT NULL DEFAULT 0,
                    high_risk_count INT NOT NULL DEFAULT 0,
                    moderate_risk_count INT NOT NULL DEFAULT 0,
                    safe_count INT NOT NULL DEFAULT 0,
                    created_at DATETIME2 DEFAULT GETDATE()
                );
            END;
        """)
        conn.commit()

    # Nạp 2 tài khoản mẫu nếu chưa có (chỉ check email, không chạy bcrypt lặp lại)
    seed_sample_users()
    _db_initialized = True


def seed_sample_users():
    """Tạo sẵn 2 tài khoản mẫu để demo vấn đáp thuận tiện: 1 Giảng viên + 1 Sinh viên."""
    sample_accounts = [
        ("giangvien@veritext.ai", "123456", "TS. Hoàng Văn Thầy (Giảng viên)", "teacher"),
        ("sinhvien@veritext.ai", "123456", "Nguyễn Văn Sinh Viên (K16)", "student"),
    ]
    with get_connection() as conn:
        cursor = conn.cursor()
        for email, pwd, full_name, role in sample_accounts:
            cursor.execute("SELECT 1 FROM users WHERE email = ?;", (email,))
            if not cursor.fetchone():
                try:
                    create_user(email=email, password=pwd, full_name=full_name, role=role)
                except Exception:
                    pass


# ==============================================================================
# 1. QUẢN LÝ NGƯỜI DÙNG & XÁC THỰC (USERS & AUTH)
# ==============================================================================

def hash_password(password: str) -> str:
    """Mã hóa mật khẩu bằng thuật toán bcrypt an toàn tuyệt đối."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Kiểm tra tính trùng khớp của mật khẩu."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


def create_user(email: str, password: str, full_name: str, role: str = "student") -> Dict[str, Any]:
    """Đăng ký tài khoản người dùng mới (Sinh viên hoặc Giảng viên)."""
    email_clean = email.strip().lower()
    full_name_clean = full_name.strip()
    role_clean = "teacher" if role.lower() in ["teacher", "giangvien", "giảng viên"] else "student"
    pwd_hash = hash_password(password)

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email = ?;", (email_clean,))
        if cursor.fetchone():
            raise ValueError(f"Email '{email_clean}' đã được đăng ký trong hệ thống!")

        cursor.execute("""
            INSERT INTO users (email, password_hash, full_name, role)
            OUTPUT INSERTED.id, INSERTED.created_at
            VALUES (?, ?, ?, ?);
        """, (email_clean, pwd_hash, full_name_clean, role_clean))
        row = cursor.fetchone()
        conn.commit()

        return {
            "id": row[0],
            "email": email_clean,
            "full_name": full_name_clean,
            "role": role_clean,
            "created_at": row[1].strftime("%Y-%m-%d %H:%M:%S") if row[1] else "",
        }


def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Xác thực đăng nhập người dùng."""
    email_clean = email.strip().lower()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, email, password_hash, full_name, role, is_active, created_at
            FROM users
            WHERE email = ?;
        """, (email_clean,))
        row = cursor.fetchone()
        if not row:
            return None

        user_id, user_email, pwd_hash, full_name, role, is_active, created_at = row
        if not is_active:
            return None

        if verify_password(password, pwd_hash):
            return {
                "id": user_id,
                "email": user_email,
                "full_name": full_name,
                "role": role,
                "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S") if created_at else "",
            }
        return None


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Lấy thông tin người dùng theo ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, email, full_name, role, is_active, created_at
            FROM users WHERE id = ?;
        """, (user_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "email": row[1],
            "full_name": row[2],
            "role": row[3],
            "is_active": bool(row[4]),
            "created_at": row[5].strftime("%Y-%m-%d %H:%M:%S") if row[5] else "",
        }


# ==============================================================================
# 2. QUẢN LÝ LỊCH SỬ THẨM ĐỊNH (DETECTION HISTORY)
# ==============================================================================

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
    user_id: Optional[int] = None,
    language: str = "en",
    perplexity_score: Optional[float] = None,
    burstiness_score: Optional[float] = None,
    has_bypass_warning: bool = False,
    bypass_details: Optional[str] = None,
    authorship_tier: Optional[str] = None,
    xai_metrics: Optional[Dict[str, Any]] = None,
) -> int:
    """Lưu trữ một lượt quét mới vào CSDL SQL Server gắn liền với tài khoản (nếu có)."""
    snippet = full_text[:180] + ("..." if len(full_text) > 180 else "")
    sentences_data_json = json.dumps(details_json.get("sentences", []), ensure_ascii=False)
    model_breakdown_json = json.dumps(details_json.get("model_breakdown", {}), ensure_ascii=False)
    xai_metrics_json = json.dumps(xai_metrics, ensure_ascii=False) if xai_metrics else None
    sentence_count = len(details_json.get("sentences", []))

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO detection_history (
                user_id, title, text_snippet, full_text, language, label,
                ai_probability, human_probability, confidence,
                word_count, char_count, sentence_count,
                model_breakdown, sentences_data,
                perplexity_score, burstiness_score,
                has_bypass_warning, bypass_details,
                authorship_tier, xai_metrics,
                inference_time_ms
            )
            OUTPUT INSERTED.id
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            user_id,
            title or filename or "Văn bản không tên",
            snippet,
            full_text,
            language or "en",
            verdict,
            round(float(ai_probability), 4),
            round(1.0 - float(ai_probability), 4),
            confidence,
            int(word_count),
            int(char_count),
            int(sentence_count),
            model_breakdown_json,
            sentences_data_json,
            perplexity_score,
            burstiness_score,
            1 if has_bypass_warning else 0,
            bypass_details,
            authorship_tier,
            xai_metrics_json,
            round(float(inference_time_ms), 2),
        ))
        inserted_id = cursor.fetchone()[0]
        conn.commit()
        return inserted_id


def get_history(limit: int = 50, offset: int = 0, user_id: Optional[int] = None, role: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Truy xuất danh sách lịch sử thẩm định theo phân quyền:
    - Sinh viên: Chỉ xem lịch sử cá nhân (user_id = ?).
    - Giảng viên / Quản trị viên: Xem toàn bộ lịch sử của sinh viên và lớp học.
    - Khách chưa đăng nhập: Xem các bài quét công khai gần nhất.
    """
    with get_connection() as conn:
        cursor = conn.cursor()

        select_cols = """
                SELECT h.id, h.created_at, h.title, h.text_snippet, h.word_count, h.char_count,
                       h.ai_probability, h.label, h.confidence, h.inference_time_ms,
                       h.language, h.perplexity_score, h.burstiness_score, h.has_bypass_warning,
                       u.full_name as author_name, u.role as author_role, h.authorship_tier
                FROM detection_history h
                LEFT JOIN users u ON h.user_id = u.id
        """
        if role == "student" and user_id is not None:
            query = select_cols + """
                WHERE h.user_id = ?
                ORDER BY h.id DESC
                OFFSET ? ROWS FETCH NEXT ? ROWS ONLY;
            """
            cursor.execute(query, (user_id, offset, limit))
        else:
            query = select_cols + """
                ORDER BY h.id DESC
                OFFSET ? ROWS FETCH NEXT ? ROWS ONLY;
            """
            cursor.execute(query, (offset, limit))

        rows = cursor.fetchall()
        result = []
        for r in rows:
            result.append({
                "id": r[0],
                "timestamp": r[1].strftime("%Y-%m-%d %H:%M:%S") if r[1] else "",
                "title": r[2],
                "text_snippet": r[3],
                "word_count": r[4],
                "char_count": r[5],
                "ai_probability": r[6],
                "verdict": r[7],
                "confidence": r[8],
                "inference_time_ms": r[9],
                "language": r[10],
                "perplexity_score": r[11],
                "burstiness_score": r[12],
                "has_bypass_warning": bool(r[13]),
                "author_name": r[14] or "Khách",
                "author_role": r[15] or "guest",
                "authorship_tier": r[16] or "",
            })
        return result


def get_scan_by_id(scan_id: int, user_id: Optional[int] = None, role: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Truy xuất chi tiết một lượt quét theo ID kèm kiểm tra quyền truy cập."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT h.*, u.full_name as author_name, u.email as author_email
            FROM detection_history h
            LEFT JOIN users u ON h.user_id = u.id
            WHERE h.id = ?;
        """, (scan_id,))
        row = cursor.fetchone()
        if not row:
            return None

        # Sinh viên không được xem bài của sinh viên khác (nếu bài đó gắn user_id khác)
        if role == "student" and user_id is not None:
            if row.user_id is not None and row.user_id != user_id:
                return None

        sentences = []
        model_breakdown = {}
        xai_metrics = None
        bypass_report = None
        if row.sentences_data:
            try:
                sentences = json.loads(row.sentences_data)
            except Exception:
                sentences = []
        if row.model_breakdown:
            try:
                model_breakdown = json.loads(row.model_breakdown)
            except Exception:
                model_breakdown = {}
        raw_xai = getattr(row, "xai_metrics", None)
        if raw_xai:
            try:
                xai_metrics = json.loads(raw_xai)
            except Exception:
                xai_metrics = None
        if row.bypass_details:
            try:
                bypass_report = json.loads(row.bypass_details)
            except Exception:
                bypass_report = {"has_bypass_warning": bool(row.has_bypass_warning), "summary": str(row.bypass_details)}

        created_str = row.created_at.strftime("%Y-%m-%d %H:%M:%S") if row.created_at else ""
        full_text_val = getattr(row, "full_text", None) or row.text_snippet
        authorship_tier_val = getattr(row, "authorship_tier", None)

        return {
            "id": row.id,
            "user_id": row.user_id,
            "author_name": row.author_name or "Khách",
            "author_email": row.author_email or "",
            "title": row.title,
            "text_snippet": row.text_snippet,
            "full_text": full_text_val,
            "language": row.language,
            "label": row.label,
            "verdict": row.label,
            "ai_probability": row.ai_probability,
            "human_probability": row.human_probability,
            "confidence": row.confidence,
            "word_count": row.word_count,
            "char_count": row.char_count,
            "sentence_count": row.sentence_count,
            "perplexity_score": row.perplexity_score,
            "burstiness_score": row.burstiness_score,
            "has_bypass_warning": bool(row.has_bypass_warning),
            "bypass_details": row.bypass_details,
            "bypass_report": bypass_report,
            "authorship_tier": authorship_tier_val,
            "xai_metrics": xai_metrics,
            "inference_time_ms": row.inference_time_ms,
            "timestamp": created_str,
            "created_at": created_str,
            "details": {
                "sentences": sentences,
                "model_breakdown": model_breakdown,
                "xai_metrics": xai_metrics,
                "bypass_report": bypass_report,
            }
        }


def delete_scan(scan_id: int, user_id: Optional[int] = None, role: Optional[str] = None) -> bool:
    """Xóa một lượt quét (Sinh viên chỉ được xóa bài của mình, Giảng viên có thể xóa bất kỳ)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        if role == "student" and user_id is not None:
            cursor.execute("DELETE FROM detection_history WHERE id = ? AND user_id = ?;", (scan_id, user_id))
        else:
            cursor.execute("DELETE FROM detection_history WHERE id = ?;", (scan_id,))
        conn.commit()
        return cursor.rowcount > 0


def get_statistics(user_id: Optional[int] = None, role: Optional[str] = None) -> Dict[str, Any]:
    """Tính toán thống kê Dashboard theo quyền người dùng trong 1 truy vấn SQL duy nhất."""
    with get_connection() as conn:
        cursor = conn.cursor()
        where_clause = ""
        params = []
        if role == "student" and user_id is not None:
            where_clause = "WHERE user_id = ?"
            params.append(user_id)

        cursor.execute(
            f"""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN label = 'AI-generated' THEN 1 ELSE 0 END) AS ai_cnt,
                AVG(inference_time_ms) AS avg_time,
                SUM(word_count) AS total_words
            FROM detection_history {where_clause};
            """,
            tuple(params),
        )
        row = cursor.fetchone()
        total = int(row[0] or 0) if row else 0
        if total == 0:
            return {
                "total_scans": 0,
                "ai_count": 0,
                "human_count": 0,
                "ai_percentage": 0.0,
                "avg_inference_time_ms": 0.0,
                "total_words_scanned": 0,
            }

        ai_cnt = int(row[1] or 0)
        avg_time = float(row[2] or 0.0)
        total_words = int(row[3] or 0)

        return {
            "total_scans": total,
            "ai_count": ai_cnt,
            "human_count": total - ai_cnt,
            "ai_percentage": round((ai_cnt / total) * 100, 1),
            "avg_inference_time_ms": round(avg_time, 1),
            "total_words_scanned": total_words,
        }


# Tự động khởi tạo database khi import
init_db()
