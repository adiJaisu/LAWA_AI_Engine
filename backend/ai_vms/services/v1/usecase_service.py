"""
Defines the service for handling user usecases and usecases management.
Includes:
- Retrieval of camera usecases.
- Update of camera usecases.
- Database transaction handling with SQLAlchemy.
- Error handling for database and server exceptions.

Author: HCLTech
"""
from typing import Union, List
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from fastapi import HTTPException
from ai_vms.models.camera import Camera
from ai_vms.models.usecases import Usecase
from ai_vms.schemas.usecases import *
from ai_vms.crud.usecases import get_all_usecases
from datetime import datetime, timedelta
from ai_vms.constant.constants import Constants as c
from ai_vms.config.logging_config import LoggingConfig

logger = LoggingConfig().setup_logging()

class UsecaseService:
    def __init__(self, db: Session):
        self.db = db

    def getusecases(self) -> Union[UsecaseDetailsResponse, GetAllUsecaseErrorResponse, GetAllUsecasesSuccessResponse]:
        """
        Retrieves camera Usecases.
        
        Returns:
            - UsecaseDetailsResponse: Success response with user Usecase.
            - GetAllUsecaseErrorResponse: Error response if there is a database failure.
            - GetAllRoleSuccessResponse: Success response with user roles.
        """
        try:
            logger.info("Retrieving Usecases...")
            usecases = get_all_usecases(self.db)
            
            filtered_usecases = [usecase for usecase in usecases]
            
            if not filtered_usecases:
                return GetAllUsecaseErrorResponse(code=500, message="No  Usecases found in the system")
            
            logger.info("Usecases retrieved successfully")
            usecases_list = [
                UsecaseDetailsResponse(
                    usecaseId=usecase.id,
                    usecaseName=usecase.name,
                    classes=[item.class_name for item in (usecase.classes or [])],
                    tracking=bool(usecase.tracking),
                    fps=usecase.fps,
                    batch=usecase.batch,
                    frameSkip=usecase.frame_skip,
                    aiResource=usecase.ai_resource.value if hasattr(usecase.ai_resource, "value") else str(usecase.ai_resource)
                )
                for usecase in filtered_usecases
            ]
            return GetAllUsecasesSuccessResponse(code=200, message="usecases retrieved successfully", usecaseDetails=usecases_list)
        
        except SQLAlchemyError as e:
            logger.error(f"Database error: {e}")
            raise GetAllUsecaseErrorResponse(code=500, message="Something went wrong | Database error")
        
        except Exception as e:
            logger.error(f"Internal Server error: {e}")
            raise GetAllUsecaseErrorResponse(code=500, message="Something went wrong | Internal Server error")        
