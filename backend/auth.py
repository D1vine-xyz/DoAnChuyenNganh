"""
Module Xác thực & Quản lý Phiên Người Dùng (JWT Authentication & RBAC)
Đồ án Chuyên ngành Kỹ thuật Phần mềm: Hệ thống Phát hiện Văn bản do LLM sinh (Ensemble PLMs)
"""

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
import jwt
from fastapi import HTTPException, status, Header, Depends, Query

# Khóa bí mật ký JWT (ưu tiên biến môi trường, fallback key mặc định cho môi trường local)
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "veritext-ai-super-secret-jwt-key-2026-academic-thesis")
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_DAYS = 7  # Token có hiệu lực 7 ngày


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
