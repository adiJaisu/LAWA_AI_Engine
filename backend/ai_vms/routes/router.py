"""
Defines the main API router for the application.

Includes:
- Versioned API routing for better scalability.
- Integration of all versioned route modules.

Author: HCLTech
"""
from fastapi import APIRouter
from ai_vms.routes.v1.router import v1_router

api_router = APIRouter()

# Include all versioned routers
api_router.include_router(v1_router, prefix="/api/v1")
