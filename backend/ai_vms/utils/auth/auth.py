import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import jwt
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from ai_vms.models.users import User
from ai_vms.database.database_connection import get_db
from ai_vms.models.roles import Role
from ai_vms.models.resource import Resource 
from ai_vms.models.scope import Scope  
from ai_vms.models.rolepermission import RolePermission 
from ai_vms.crud.auth import (
    get_user_by_email,
    get_role_permissions,
)

# Configuration for JWT and Password Hashing
SECRET_KEY = os.getenv("SECRET_KEY", "ai_vms-secret-key-12345")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 24 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def verify_jwt_token(token: str = Depends(oauth2_scheme)) -> Dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        return {"email": email}
    except jwt.PyJWTError:
        raise credentials_exception

def get_role_permissions(db: Session, role_id: int) -> List[str]:
    """
    Returns a flat list of permission strings like 'resource:scope' based on the role_id.
    """
    if not role_id:
        return []

    rows = (
        db.query(Resource.name.label("resource"), Scope.name.label("scope"))
        .join(RolePermission, RolePermission.resource_id == Resource.id)
        .join(Scope, Scope.id == RolePermission.scope_id)
        .filter(
            RolePermission.role_id == role_id,
            Resource.is_active.is_(True),
            Scope.is_active.is_(True),
        )
        .all()
    )

    permissions: List[str] = [f"{res}:{sc}" for res, sc in rows]
    return sorted(set(permissions))


def get_current_user(
    token_user: Dict = Depends(verify_jwt_token),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:

    email = token_user["email"].lower()
    db_user = get_user_by_email(db, email)
    
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    permissions = get_role_permissions(db, db_user.role_id)

    return {
        "user": db_user,
        "role": db_user.role,
        "permissions": permissions
    }

def require_permission(req_per: str):
    async def _dep(
        current=Depends(get_current_user),
    ):
        user_permissions: List[str] = current["permissions"]

        if "*:*" in user_permissions:
            return current
        if req_per in user_permissions:
            return current

        try:
            resource, scope = req_per.split(":")
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Invalid permission format for '{req_per}'. Expected 'resource:scope'."
            )

        if f"{resource}:*" in user_permissions:
            return current
        if f"*:{scope}" in user_permissions:
            return current

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Missing permission: {req_per}"
        )

    return _dep
