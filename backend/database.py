"""
Module Quản lý Cơ sở Dữ liệu Microsoft SQL Server (Database: VeriTextAI)
Đồ án Chuyên ngành Kỹ thuật Phần mềm: Hệ thống Phát hiện Văn bản do LLM sinh (Ensemble PLMs)
Hỗ trợ:
- Xác thực người dùng, phân quyền Giảng viên & Sinh viên (Role-based access).
- Lưu trữ lịch sử thẩm định chi tiết theo từng người dùng (User-specific audit logs).
- Quản lý các lô chấm bài tập hàng loạt của lớp học (Classroom batches).
"""

import os
import re
import secrets
import sys
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
import pyodbc
import bcrypt

try:
    from dotenv import load_dotenv
    _DB_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(_DB_BASE_DIR, "..", ".env"))
    load_dotenv(os.path.join(_DB_BASE_DIR, ".env"))
except Exception:
    pass

EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

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
    conn = pyodbc.connect(DB_CONNECTION_STRING, autocommit=False, timeout=15)
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
                    auth_provider NVARCHAR(30) NOT NULL DEFAULT 'local',
                    google_id NVARCHAR(255) NULL,
                    avatar_url NVARCHAR(500) NULL,
                    is_active BIT NOT NULL DEFAULT 1,
                    last_login DATETIME2 NULL,
                    created_at DATETIME2 DEFAULT GETDATE()
                );
            END;
        """)

        # Bổ sung các cột Google OAuth & Hồ sơ nếu bảng users đã tạo từ trước
        for col_name, col_def in [
            ("auth_provider", "NVARCHAR(30) NOT NULL DEFAULT 'local'"),
            ("google_id", "NVARCHAR(255) NULL"),
            ("avatar_url", "NVARCHAR(500) NULL"),
            ("last_login", "DATETIME2 NULL"),
        ]:
            cursor.execute(f"""
                IF COL_LENGTH('users', '{col_name}') IS NULL
                BEGIN
                    ALTER TABLE users ADD {col_name} {col_def};
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
                    bypass_count INT NOT NULL DEFAULT 0,
                    avg_ai_probability FLOAT NOT NULL DEFAULT 0.0,
                    created_at DATETIME2 DEFAULT GETDATE()
                );
            END;
        """)

        # Nâng cấp schema classroom_batches & liên kết batch_id từ detection_history
        for col_name, col_def in [
            ("bypass_count", "INT NOT NULL DEFAULT 0"),
            ("avg_ai_probability", "FLOAT NOT NULL DEFAULT 0.0"),
        ]:
            cursor.execute(f"""
                IF COL_LENGTH('classroom_batches', '{col_name}') IS NULL
                BEGIN
                    ALTER TABLE classroom_batches ADD {col_name} {col_def};
                END;
            """)

        for col_name, col_def in [
            ("batch_id", "INT NULL"),
            ("student_label", "NVARCHAR(255) NULL"),
            ("risk_level", "NVARCHAR(20) NULL"),
        ]:
            cursor.execute(f"""
                IF COL_LENGTH('detection_history', '{col_name}') IS NULL
                BEGIN
                    ALTER TABLE detection_history ADD {col_name} {col_def};
                END;
            """)

        cursor.execute("""
            IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_detection_history_batch_id' AND object_id = OBJECT_ID('detection_history'))
            BEGIN
                CREATE INDEX IX_detection_history_batch_id ON detection_history(batch_id);
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
    missing_accounts = []
    with get_connection() as conn:
        cursor = conn.cursor()
        for acc in sample_accounts:
            cursor.execute("SELECT 1 FROM users WHERE email = ?;", (acc[0],))
            if not cursor.fetchone():
                missing_accounts.append(acc)

    for email, pwd, full_name, role in missing_accounts:
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


def create_user(
    email: str,
    password: str,
    full_name: str,
    role: str = "student",
    auth_provider: str = "local",
    google_id: Optional[str] = None,
    avatar_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Đăng ký tài khoản người dùng mới (Sinh viên hoặc Giảng viên)."""
    email_clean = email.strip().lower()
    if not EMAIL_REGEX.match(email_clean):
        raise ValueError("Định dạng email không hợp lệ! Vui lòng kiểm tra lại.")
    full_name_clean = full_name.strip()
    if len(full_name_clean) < 2:
        raise ValueError("Họ và tên phải có ít nhất 2 ký tự.")
    role_clean = "teacher" if role.lower() in ["teacher", "giangvien", "giảng viên"] else "student"
    pwd_hash = hash_password(password)

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email = ?;", (email_clean,))
        if cursor.fetchone():
            raise ValueError(f"Email '{email_clean}' đã được đăng ký trong hệ thống!")

        cursor.execute("""
            INSERT INTO users (email, password_hash, full_name, role, auth_provider, google_id, avatar_url, last_login)
            OUTPUT INSERTED.id, INSERTED.created_at, INSERTED.last_login
            VALUES (?, ?, ?, ?, ?, ?, ?, GETDATE());
        """, (email_clean, pwd_hash, full_name_clean, role_clean, auth_provider, google_id, avatar_url))
        row = cursor.fetchone()
        conn.commit()

        return {
            "id": row[0],
            "email": email_clean,
            "full_name": full_name_clean,
            "role": role_clean,
            "auth_provider": auth_provider,
            "is_google_linked": bool(google_id),
            "avatar_url": avatar_url or "",
            "created_at": row[1].strftime("%Y-%m-%d %H:%M:%S") if row[1] else "",
            "last_login": row[2].strftime("%Y-%m-%d %H:%M:%S") if row[2] else "",
        }


def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Xác thực đăng nhập người dùng bằng email & mật khẩu (bcrypt)."""
    email_clean = email.strip().lower()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, email, password_hash, full_name, role, is_active, created_at,
                   auth_provider, google_id, avatar_url
            FROM users
            WHERE email = ?;
        """, (email_clean,))
        row = cursor.fetchone()
        if not row:
            return None

        user_id, user_email, pwd_hash, full_name, role, is_active, created_at, provider, google_id, avatar_url = row
        if not is_active:
            return None

        if verify_password(password, pwd_hash):
            cursor.execute("UPDATE users SET last_login = GETDATE() WHERE id = ?;", (user_id,))
            conn.commit()
            return {
                "id": user_id,
                "email": user_email,
                "full_name": full_name,
                "role": role,
                "auth_provider": provider or "local",
                "is_google_linked": bool(google_id),
                "avatar_url": avatar_url or "",
                "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S") if created_at else "",
                "last_login": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        return None


def upsert_google_user(
    google_id: str,
    email: str,
    full_name: str,
    avatar_url: str = "",
    role: str = "student",
) -> Dict[str, Any]:
    """
    Đăng nhập hoặc tự động đăng ký tài khoản bằng Google OAuth 2.0 (ID Token):
    - Nếu email/google_id đã tồn tại: liên kết google_id, cập nhật avatar & last_login, giữ nguyên vai trò hiện tại.
    - Nếu là người dùng Google mới: tạo tài khoản mới trong SQL Server với auth_provider='google'.
    """
    email_clean = email.strip().lower()
    name_clean = (full_name or email_clean.split("@")[0]).strip()
    role_clean = "teacher" if role.lower() in ["teacher", "giangvien", "giảng viên"] else "student"

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, email, full_name, role, is_active, created_at, auth_provider, google_id, avatar_url
            FROM users
            WHERE google_id = ? OR email = ?;
        """, (google_id, email_clean))
        row = cursor.fetchone()

        if row:
            uid, u_email, u_name, u_role, is_active, created_at, u_provider, u_gid, u_avatar = row
            if not is_active:
                raise ValueError("Tài khoản này đã bị vô hiệu hóa.")
            new_avatar = avatar_url or u_avatar or ""
            cursor.execute("""
                UPDATE users
                SET google_id = ?, avatar_url = ?, last_login = GETDATE()
                WHERE id = ?;
            """, (google_id, new_avatar, uid))
            conn.commit()
            return {
                "id": uid,
                "email": u_email,
                "full_name": u_name or name_clean,
                "role": u_role,
                "auth_provider": u_provider or "google",
                "is_google_linked": True,
                "avatar_url": new_avatar,
                "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S") if created_at else "",
                "last_login": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

    # Tạo tài khoản mới cho người dùng đăng nhập Google lần đầu
    random_pwd = secrets.token_urlsafe(24)
    return create_user(
        email=email_clean,
        password=random_pwd,
        full_name=name_clean,
        role=role_clean,
        auth_provider="google",
        google_id=google_id,
        avatar_url=avatar_url,
    )


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Lấy thông tin người dùng theo ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, email, full_name, role, is_active, created_at,
                   auth_provider, google_id, avatar_url, last_login
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
            "auth_provider": row[6] or "local",
            "is_google_linked": bool(row[7]),
            "avatar_url": row[8] or "",
            "last_login": row[9].strftime("%Y-%m-%d %H:%M:%S") if row[9] else "",
        }


def update_user_profile(user_id: int, full_name: str, role: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Cập nhật họ tên và vai trò tài khoản trong SQL Server."""
    clean_name = full_name.strip()
    if len(clean_name) < 2:
        raise ValueError("Họ và tên phải có ít nhất 2 ký tự.")

    with get_connection() as conn:
        cursor = conn.cursor()
        if role is not None:
            role_clean = "teacher" if role.lower() in ["teacher", "giangvien", "giảng viên"] else "student"
            cursor.execute("UPDATE users SET full_name = ?, role = ? WHERE id = ?;", (clean_name, role_clean, user_id))
        else:
            cursor.execute("UPDATE users SET full_name = ? WHERE id = ?;", (clean_name, user_id))
        conn.commit()

    return get_user_by_id(user_id)


def change_user_password(user_id: int, old_password: Optional[str], new_password: str) -> None:
    """Đổi mật khẩu tài khoản (yêu cầu mật khẩu cũ với tài khoản local; hỗ trợ đặt mật khẩu cho tài khoản Google)."""
    if not new_password or len(new_password) < 6:
        raise ValueError("Mật khẩu mới phải có ít nhất 6 ký tự.")

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash, auth_provider FROM users WHERE id = ?;", (user_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError("Tài khoản không tồn tại.")

        pwd_hash, provider = row
        # Cả tài khoản 'local' lẫn 'hybrid' (Google đã từng đặt mật khẩu) đều bắt buộc xác thực mật khẩu cũ
        if (provider or "local") != "google":
            if not old_password or not verify_password(old_password, pwd_hash):
                raise ValueError("Mật khẩu hiện tại không chính xác!")

        new_hash = hash_password(new_password)
        if (provider or "local") == "google":
            cursor.execute(
                "UPDATE users SET password_hash = ?, auth_provider = 'hybrid' WHERE id = ?;",
                (new_hash, user_id),
            )
        else:
            cursor.execute("UPDATE users SET password_hash = ? WHERE id = ?;", (new_hash, user_id))
        conn.commit()


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
    batch_id: Optional[int] = None,
    student_label: Optional[str] = None,
    risk_level: Optional[str] = None,
) -> int:
    """Lưu trữ một lượt quét mới vào CSDL SQL Server gắn liền với tài khoản (và đợt chấm lớp học nếu có)."""
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
                batch_id, student_label, risk_level,
                inference_time_ms
            )
            OUTPUT INSERTED.id
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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
            batch_id,
            student_label,
            risk_level,
            round(float(inference_time_ms), 2),
        ))
        inserted_id = cursor.fetchone()[0]
        conn.commit()
        return inserted_id


def get_history(
    limit: int = 50,
    offset: int = 0,
    user_id: Optional[int] = None,
    role: Optional[str] = None,
    search_query: Optional[str] = None,
    language: Optional[str] = None,
    verdict: Optional[str] = None,
    bypass_only: bool = False,
) -> List[Dict[str, Any]]:
    """
    Truy xuất danh sách lịch sử thẩm định theo phân quyền RBAC kết hợp tìm kiếm & lọc đa tiêu chí:
    - Sinh viên: Chỉ xem lịch sử cá nhân (user_id = ?).
    - Giảng viên / Quản trị viên: Xem toàn bộ lịch sử của sinh viên và lớp học.
    - Khách chưa đăng nhập: Chỉ xem các bài quét ẩn danh (user_id IS NULL).
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
        conditions: List[str] = []
        params: List[Any] = []

        if role in ("teacher", "admin"):
            pass
        elif role == "student" and user_id is not None:
            conditions.append("h.user_id = ?")
            params.append(user_id)
        else:
            conditions.append("h.user_id IS NULL")

        if search_query and search_query.strip():
            q_raw = search_query.strip()
            # Escape các ký tự wildcard của T-SQL LIKE (\, %, _, [) để tìm chính xác chuỗi như "[LMS/Web]"
            q_escaped = (
                q_raw.replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
                .replace("[", "\\[")
            )
            like_pat = f"%{q_escaped}%"
            if q_raw.lstrip("#").isdigit():
                conditions.append(
                    "(h.id = ? OR h.title LIKE ? ESCAPE '\\' OR u.full_name LIKE ? ESCAPE '\\')"
                )
                params.extend([int(q_raw.lstrip("#")), like_pat, like_pat])
            else:
                conditions.append(
                    "(h.title LIKE ? ESCAPE '\\' OR u.full_name LIKE ? ESCAPE '\\' OR h.student_label LIKE ? ESCAPE '\\')"
                )
                params.extend([like_pat, like_pat, like_pat])

        if language in ("vi", "en"):
            conditions.append("h.language = ?")
            params.append(language)

        if verdict in ("AI-generated", "Human-written"):
            conditions.append("h.label = ?")
            params.append(verdict)

        if bypass_only:
            conditions.append("h.has_bypass_warning = 1")

        where_sql = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        query = (
            select_cols
            + where_sql
            + """
                ORDER BY h.id DESC
                OFFSET ? ROWS FETCH NEXT ? ROWS ONLY;
            """
        )
        params.extend([offset, limit])
        cursor.execute(query, tuple(params))

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

        # Kiểm tra phân quyền RBAC chặt chẽ:
        # - Giảng viên / Admin: xem được toàn bộ
        # - Sinh viên: chỉ xem bài của mình (hoặc bài ẩn danh công khai)
        # - Khách chưa đăng nhập: chỉ xem bài ẩn danh (user_id IS NULL), không đọc được bài riêng của GV/SV
        if role not in ("teacher", "admin"):
            if role == "student" and user_id is not None:
                if row.user_id is not None and row.user_id != user_id:
                    return None
            else:
                if row.user_id is not None:
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
                "text_stats": model_breakdown.get("text_stats") if isinstance(model_breakdown, dict) else None,
                "xai_metrics": xai_metrics,
                "bypass_report": bypass_report,
            }
        }


def get_scan_public_verification(scan_id: int) -> Optional[Dict[str, Any]]:
    """
    Lấy thông tin tóm tắt chứng chỉ kiểm định phục vụ cổng Tra Cứu Xác Thực Mã QR / Hash.
    Tuyệt đối KHÔNG trả về nội dung bài viết gốc (full_text) để bảo mật quyền riêng tư học thuật.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT h.id, h.title, h.language, h.label, h.ai_probability, h.human_probability,
                   h.confidence, h.word_count, h.char_count, h.has_bypass_warning,
                   h.authorship_tier, h.created_at, COALESCE(u.full_name, N'Khách vãng lai') as author_name
            FROM detection_history h
            LEFT JOIN users u ON h.user_id = u.id
            WHERE h.id = ?;
        """, (scan_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "id": int(row[0]),
            "title": row[1] or "Tài liệu thẩm định",
            "language": row[2] or "en",
            "verdict": row[3],
            "ai_probability": float(row[4] or 0.0),
            "human_probability": float(row[5] or 0.0),
            "confidence": row[6] or "",
            "word_count": int(row[7] or 0),
            "char_count": int(row[8] or 0),
            "has_bypass_warning": bool(row[9]),
            "authorship_tier": row[10] or "",
            "timestamp": row[11].strftime("%Y-%m-%d %H:%M:%S") if row[11] else "",
            "author_name": row[12] or "Khách vãng lai",
        }


def delete_scan(scan_id: int, user_id: Optional[int] = None, role: Optional[str] = None) -> bool:
    """Xóa một lượt quét (Giảng viên xóa được mọi bài, Sinh viên chỉ xóa bài mình, Khách chỉ xóa bài ẩn danh)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        if role in ("teacher", "admin"):
            cursor.execute("DELETE FROM detection_history WHERE id = ?;", (scan_id,))
        elif role == "student" and user_id is not None:
            cursor.execute("DELETE FROM detection_history WHERE id = ? AND user_id = ?;", (scan_id, user_id))
        else:
            cursor.execute("DELETE FROM detection_history WHERE id = ? AND user_id IS NULL;", (scan_id,))
        conn.commit()
        return cursor.rowcount > 0


def get_statistics(user_id: Optional[int] = None, role: Optional[str] = None) -> Dict[str, Any]:
    """Tính toán thống kê Dashboard theo quyền người dùng trong 1 truy vấn SQL duy nhất."""
    with get_connection() as conn:
        cursor = conn.cursor()
        where_clause = ""
        params = []
        if role in ("teacher", "admin"):
            where_clause = ""
        elif role == "student" and user_id is not None:
            where_clause = "WHERE user_id = ?"
            params.append(user_id)
        else:
            where_clause = "WHERE user_id IS NULL"

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


# ==============================================================================
# 3. QUẢN LÝ ĐỢT CHẤM BÀI LỚP HỌC (CLASSROOM BATCHES)
# ==============================================================================

def create_classroom_batch(teacher_id: int, batch_name: str) -> int:
    """Tạo một đợt chấm bài lớp học mới trong SQL Server."""
    clean_name = (batch_name or "Đợt chấm bài lớp học").strip()[:250]
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO classroom_batches (teacher_id, batch_name)
            OUTPUT INSERTED.id
            VALUES (?, ?);
        """, (teacher_id, clean_name))
        batch_id = cursor.fetchone()[0]
        conn.commit()
        return int(batch_id)


def finalize_classroom_batch(
    batch_id: int,
    total_files: int,
    high_risk_count: int,
    moderate_risk_count: int,
    safe_count: int,
    bypass_count: int,
    avg_ai_probability: float,
) -> None:
    """Cập nhật số liệu thống kê tổng hợp sau khi quét xong toàn bộ các tệp của lớp."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE classroom_batches
            SET total_files = ?,
                high_risk_count = ?,
                moderate_risk_count = ?,
                safe_count = ?,
                bypass_count = ?,
                avg_ai_probability = ?
            WHERE id = ?;
        """, (
            int(total_files),
            int(high_risk_count),
            int(moderate_risk_count),
            int(safe_count),
            int(bypass_count),
            round(float(avg_ai_probability), 4),
            int(batch_id),
        ))
        conn.commit()


def get_classroom_batches(teacher_id: Optional[int] = None, limit: int = 30) -> List[Dict[str, Any]]:
    """Lấy danh sách các đợt chấm bài lớp học gần nhất."""
    with get_connection() as conn:
        cursor = conn.cursor()
        base_sql = """
            SELECT b.id, b.teacher_id, u.full_name as teacher_name, b.batch_name,
                   b.total_files, b.high_risk_count, b.moderate_risk_count, b.safe_count,
                   b.bypass_count, b.avg_ai_probability, b.created_at
            FROM classroom_batches b
            LEFT JOIN users u ON b.teacher_id = u.id
        """
        if teacher_id is not None:
            cursor.execute(
                base_sql + " WHERE b.teacher_id = ? ORDER BY b.id DESC OFFSET 0 ROWS FETCH NEXT ? ROWS ONLY;",
                (teacher_id, limit),
            )
        else:
            cursor.execute(
                base_sql + " ORDER BY b.id DESC OFFSET 0 ROWS FETCH NEXT ? ROWS ONLY;",
                (limit,),
            )
        rows = cursor.fetchall()
        batches = []
        for r in rows:
            batches.append({
                "id": r[0],
                "teacher_id": r[1],
                "teacher_name": r[2] or "Giảng viên",
                "batch_name": r[3],
                "total_files": int(r[4] or 0),
                "high_risk_count": int(r[5] or 0),
                "moderate_risk_count": int(r[6] or 0),
                "safe_count": int(r[7] or 0),
                "bypass_count": int(r[8] or 0),
                "avg_ai_probability": round(float(r[9] or 0.0), 4),
                "created_at": r[10].strftime("%Y-%m-%d %H:%M:%S") if r[10] else "",
            })
        return batches


def get_classroom_batch_detail(batch_id: int) -> Optional[Dict[str, Any]]:
    """Lấy thông tin tổng quan đợt chấm kèm danh sách bài sinh viên xếp hạng rủi ro giảm dần."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT b.id, b.teacher_id, u.full_name as teacher_name, b.batch_name,
                   b.total_files, b.high_risk_count, b.moderate_risk_count, b.safe_count,
                   b.bypass_count, b.avg_ai_probability, b.created_at
            FROM classroom_batches b
            LEFT JOIN users u ON b.teacher_id = u.id
            WHERE b.id = ?;
        """, (batch_id,))
        b = cursor.fetchone()
        if not b:
            return None

        cursor.execute("""
            SELECT id, title, student_label, language, label, ai_probability, human_probability,
                   confidence, word_count, perplexity_score, burstiness_score,
                   has_bypass_warning, authorship_tier, risk_level, inference_time_ms, created_at
            FROM detection_history
            WHERE batch_id = ?
            ORDER BY has_bypass_warning DESC, ai_probability DESC, id ASC;
        """, (batch_id,))
        scan_rows = cursor.fetchall()
        submissions = []
        for s in scan_rows:
            ai_prob = float(s[5] or 0.0)
            has_bp = bool(s[11])
            r_level = s[13] or ("high" if (ai_prob >= 0.75 or has_bp) else "moderate" if ai_prob >= 0.45 else "safe")
            submissions.append({
                "scan_id": s[0],
                "title": s[1],
                "student_label": s[2] or s[1],
                "language": s[3] or "en",
                "verdict": s[4],
                "ai_probability": ai_prob,
                "human_probability": float(s[6] or (1.0 - ai_prob)),
                "confidence": s[7],
                "word_count": int(s[8] or 0),
                "perplexity_score": s[9],
                "burstiness_score": s[10],
                "has_bypass_warning": has_bp,
                "authorship_tier": s[12] or "",
                "risk_level": r_level,
                "inference_time_ms": float(s[14] or 0.0),
                "created_at": s[15].strftime("%Y-%m-%d %H:%M:%S") if s[15] else "",
            })

        return {
            "id": b[0],
            "teacher_id": b[1],
            "teacher_name": b[2] or "Giảng viên",
            "batch_name": b[3],
            "total_files": int(b[4] or len(submissions)),
            "high_risk_count": int(b[5] or 0),
            "moderate_risk_count": int(b[6] or 0),
            "safe_count": int(b[7] or 0),
            "bypass_count": int(b[8] or 0),
            "avg_ai_probability": round(float(b[9] or 0.0), 4),
            "created_at": b[10].strftime("%Y-%m-%d %H:%M:%S") if b[10] else "",
            "submissions": submissions,
        }


def delete_classroom_batch(batch_id: int) -> bool:
    """Xóa một đợt chấm lớp học và gỡ liên kết batch_id khỏi bảng detection_history."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE detection_history SET batch_id = NULL WHERE batch_id = ?;", (batch_id,))
        cursor.execute("DELETE FROM classroom_batches WHERE id = ?;", (batch_id,))
        conn.commit()
        return cursor.rowcount > 0


# Tự động khởi tạo database khi import
init_db()
