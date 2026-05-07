
# src/utils/auth/app_user_auth.py

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from ai_vms.database.database_connection import get_db
from ai_vms.utils.auth.azure_ad_bearer import AzureADBearer
from ai_vms.models.users import User  # your DB model

azure_bearer = AzureADBearer(scopes=["access_as_user"])

async def get_app_user(
    token_user=Depends(azure_bearer),
    db: Session = Depends(get_db)
):
    email = token_user.get("email")

    user = db.query(User).filter(User.email == email).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User not authorized for this application",
        )

    return user   # return DB user object
