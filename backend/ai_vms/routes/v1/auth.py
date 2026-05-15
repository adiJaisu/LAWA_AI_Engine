"""
API routes for Authentication module.

Includes secured endpoints that validate user identity and return
user details along with permissions based on assigned roles.

Author: HCLTech
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from ai_vms.database.database_connection import get_db
from ai_vms.schemas.users import LoginRequest, SignUpRequest, UserDetailsResponse
from ai_vms.utils.auth.auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user
)
from ai_vms.crud.users import get_user_by_email, create_user

router = APIRouter()

@router.post("/login")
async def login(login_data: LoginRequest, db: Session = Depends(get_db)):
    user = get_user_by_email(db, login_data.email)
    if not user or not user.hashed_password or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user.email})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "role_id": user.role_id
        }
    }

@router.post("/signup")
async def signup(signup_data: SignUpRequest, db: Session = Depends(get_db)):
    existing_user = get_user_by_email(db, signup_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    hashed_password = get_password_hash(signup_data.password)
    # Default role_id = 2 (standard user, assuming 1 is admin)
    new_user = create_user(
        db,
        email=signup_data.email,
        first_name=signup_data.firstName,
        last_name=signup_data.lastName,
        role_id=2, 
        hashed_password=hashed_password
    )
    
    return {"message": "User created successfully", "user_id": new_user.id}

@router.get("/secure")
async def secure_endpoint(current = Depends(get_current_user)):
    role = current["user"].role or current.get("role")

    return {
        "code": 200,
        "message": "Authenticated successfully",
        "user": {
            "id": current["user"].id,
            "email": current["user"].email,
            "first_name": current["user"].first_name,
            "last_name": current["user"].last_name,
            "role_id": current["user"].role_id,
            "role": role.name if role else None
        },
        "permissions": current.get("permissions")
    }
