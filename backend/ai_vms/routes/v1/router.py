"""
Defines the main API router for version 1 (v1) of the application.

Includes:
- User-related routes.
- Versioned API structure for better maintainability.

Author: HCLTech
"""
from fastapi import APIRouter
from ai_vms.routes.v1.users import router as users_router
from ai_vms.routes.v1.cameras import router as camera_router
from ai_vms.routes.v1.auth import router as auth_router
from ai_vms.routes.v1.roles import router as role_router
from ai_vms.routes.v1.roi import router as roi_router
from ai_vms.routes.v1.usecases import router as usecase_router

v1_router = APIRouter()

v1_router.include_router(users_router, prefix="/users", tags=["Users"])
v1_router.include_router(camera_router, prefix="/camera", tags=["Camera"])
v1_router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
v1_router.include_router(role_router, prefix="/roles", tags=["Roles"])
v1_router.include_router(roi_router, prefix="/roi", tags=["ROI"])
v1_router.include_router(usecase_router, prefix="/usecases", tags=["Usecases"])