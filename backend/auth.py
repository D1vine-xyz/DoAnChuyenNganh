"""
Module Xác thực & Quản lý Phiên Người Dùng (JWT Authentication & RBAC)
Đồ án Chuyên ngành Kỹ thuật Phần mềm: Hệ thống Phát hiện Văn bản do LLM sinh (Ensemble PLMs)
"""

import os
import re
import hmac
import hashlib
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
import jwt
from fastapi import HTTPException, status, Header, Depends, Query

try:
    from dotenv import load_dotenv
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(_BASE_DIR, "..", ".env"))
    load_dotenv(os.path.join(_BASE_DIR, ".env"))
except Exception:
    pass

# Khóa bí mật ký JWT (ưu tiên biến môi trường, fallback key mặc định cho môi trường local)
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "veritext-ai-super-secret-jwt-key-2026-academic-thesis")
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_DAYS = 7  # Token có hiệu lực 7 ngày


def get_google_client_id() -> str:
    """Đọc GOOGLE_CLIENT_ID từ biến môi trường / file .env (bỏ qua giá trị placeholder)."""
    cid = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    if not cid or "your_google_client_id" in cid.lower():
        return ""
    return cid


def verify_google_id_token(credential: str) -> Dict[str, Any]:
    """
    Xác thực chữ ký số Google ID Token (Google Identity Services / OAuth 2.0)
    bằng thư viện chính chủ `google-auth`.
    """
    client_id = get_google_client_id()
    if not client_id:
        raise ValueError(
            "Chưa cấu hình GOOGLE_CLIENT_ID trong tệp .env. "
            "Vui lòng dán Google OAuth Client ID vào .env và khởi động lại server."
        )

    from google.oauth2 import id_token as google_id_token
    from google.auth.transport import requests as google_requests

    try:
        idinfo = google_id_token.verify_oauth2_token(
            credential,
            google_requests.Request(),
            client_id,
        )
    except Exception as e:
        raise ValueError(f"Google ID Token không hợp lệ hoặc đã hết hạn: {str(e)}")

    if not idinfo.get("email") or not idinfo.get("email_verified", False):
        raise ValueError("Tài khoản Google chưa xác minh địa chỉ email.")

    return {
        "google_id": str(idinfo["sub"]),
        "email": str(idinfo["email"]).strip().lower(),
        "full_name": str(idinfo.get("name") or idinfo["email"].split("@")[0]).strip(),
        "avatar_url": str(idinfo.get("picture") or ""),
    }


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Tạo mã JWT Access Token với payload chứa ID, email và vai trò người dùng."""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta if expires_delta else timedelta(days=JWT_ACCESS_TOKEN_EXPIRE_DAYS))

    to_encode.update({
        "exp": expire,
        "iat": now,
    })
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Giải mã và xác thực tính hợp lệ của JWT Access Token."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


def extract_bearer_token(authorization_header: Optional[str]) -> Optional[str]:
    """Bóc tách Bearer token từ Authorization header."""
    if not authorization_header:
        return None
    parts = authorization_header.strip().split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return authorization_header.strip()


async def get_current_user_optional(
    authorization: Optional[str] = Header(None, alias="Authorization")
) -> Optional[Dict[str, Any]]:
    """
    Dependency FastAPI: Trả về thông tin người dùng nếu có token hợp lệ.
    Nếu là khách vãng lai (không gửi token), trả về None thay vì báo lỗi.
    """
    token = extract_bearer_token(authorization)
    if not token:
        return None
    payload = decode_access_token(token)
    if not payload:
        return None
    return {
        "id": payload.get("id"),
        "email": payload.get("email"),
        "full_name": payload.get("full_name"),
        "role": payload.get("role", "student"),
    }


async def get_current_user(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> Dict[str, Any]:
    """
    Dependency FastAPI: Bắt buộc người dùng phải đăng nhập.
    Ném lỗi 401 Unauthorized nếu chưa đăng nhập hoặc token hết hạn.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Phiên đăng nhập đã hết hạn hoặc không hợp lệ. Vui lòng đăng nhập lại!",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user


async def get_current_user_from_header_or_query(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    token: Optional[str] = Query(None, description="Optional JWT token for direct download links"),
) -> Optional[Dict[str, Any]]:
    """Hỗ trợ xác thực qua Header hoặc query ?token= (dùng riêng cho xuất file PDF trực tiếp)."""
    raw_token = extract_bearer_token(authorization) or token
    if not raw_token:
        return None
    payload = decode_access_token(raw_token)
    if not payload:
        return None
    return {
        "id": payload.get("id"),
        "email": payload.get("email"),
        "full_name": payload.get("full_name"),
        "role": payload.get("role", "student"),
    }


def require_role(allowed_roles: List[str]):
    """Kiểm tra phân quyền người dùng (Role-Based Access Control)."""
    async def role_checker(current_user: Dict[str, Any] = Depends(get_current_user)):
        user_role = current_user.get("role", "student")
        if user_role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Quyền truy cập bị từ chối. Tính năng này yêu cầu quyền: {', '.join(allowed_roles)}!",
            )
        return current_user
    return role_checker


AUDIT_CODE_REGEX = re.compile(r"^VT-(\d{1,10})-([0-9A-Fa-f]{8})$")


def compute_audit_code(scan_id: int, ai_probability: float, word_count: int) -> str:
    """
    Sinh Mã Chứng Thực Chống Giả Mạo (HMAC-SHA256 rút gọn) từ dữ liệu gốc bất biến trong SQL Server.
    Nếu kẻ gian sửa điểm % AI hoặc số từ trên file PDF rồi giữ nguyên mã VT-..., đối chiếu sẽ phát hiện ngay.
    """
    canonical = f"VERITEXT|{int(scan_id)}|{round(float(ai_probability), 4):.4f}|{int(word_count)}"
    digest = hmac.new(
        JWT_SECRET_KEY.encode("utf-8"),
        canonical.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:8].upper()
    return f"VT-{int(scan_id):06d}-{digest}"


def parse_and_verify_audit_code(audit_code: str, scan_record: Optional[Dict[str, Any]]) -> bool:
    """Kiểm tra chữ ký HMAC-SHA256 của mã chứng thực đối với bản ghi gốc trong SQL Server."""
    if not audit_code or not scan_record:
        return False
    code_clean = audit_code.strip().upper()
    m = AUDIT_CODE_REGEX.match(code_clean)
    if not m:
        return False
    parsed_id = int(m.group(1))
    if parsed_id != int(scan_record.get("id", -1)):
        return False
    expected = compute_audit_code(
        scan_id=parsed_id,
        ai_probability=float(scan_record.get("ai_probability", 0.0)),
        word_count=int(scan_record.get("word_count", 0)),
    )
    return hmac.compare_digest(code_clean, expected)
