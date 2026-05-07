"""
Defines the user service for handling user-related operations.

Includes:
- Getting All users info from database along with their roles
- Database transaction handling with SQLAlchemy.
- Error handling for database and server exceptions.

Author: HCLTech
"""

from typing import Any, Union
from sqlalchemy.exc import IntegrityError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from ai_vms.schemas.users import (
    GetAllUserFailureResponse, GetAllUserSuccessResponse,
    AllUsersInfoResponse, AddUserSuccessResponse, AddUserFailureResponse,
    AddUserRequest, UpdateUserRequest, UpdateUserSuccessResponse, UpdateUserFailureResponse, 
    GetUserDetailSuccessResponse, GetUserDetailFailureResponse, DeleteUserSuccessResponse, 
    DeleteUserFailureResponse, UserInfoResponse
)
from ai_vms.config.logging_config import LoggingConfig
from ai_vms.crud.users import (
    get_all_users, get_user_by_id, get_user_by_email, create_user,
    update_user, soft_delete_user_by_id
)
from ai_vms.crud.roles import role_exists


logger = LoggingConfig().setup_logging()

class UserService:
    def __init__(self, db: Session):
        self.db = db

    def get_all_users(self) -> Union[GetAllUserSuccessResponse, GetAllUserFailureResponse]:
        try:
            logger.info("API requested to fetch all users.")
            users = get_all_users(self.db)
            
            user_list = [
                AllUsersInfoResponse(
                    userId=user.id,
                    username=user.email,
                    firstName=user.first_name,
                    lastName=user.last_name,
                    roleId=user.role_id,
                    roleName=user.role.name,
                    status=user.is_active,
                    createdAt=user.created_at,
                    updatedAt=user.updated_at,
                )
                for user in users
            ]
            logger.info("Users fetched successfully")
            return GetAllUserSuccessResponse(code=200, message="Users fetched successfully", users=user_list)

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database Error: {str(e)}")
            return GetAllUserFailureResponse(code=500, message="Database Error Occurred")

        except Exception as e:
            logger.error(f"An error occurred while fetching users: {str(e)}")
            return GetAllUserFailureResponse(code=500, message="Internal Server Error")


    def add_user(self, payload: AddUserRequest, current: dict[str, Any]):
        try:
            logger.info("Add user requested")
            created_by = current.get("user")
            if created_by is None:
                return AddUserFailureResponse(code=401, message="Unauthorized user context")

            if not role_exists(self.db, payload.roleId):
                return AddUserFailureResponse(code=400, message="Invalid roleId")

            if get_user_by_email(self.db, payload.email):
                return AddUserFailureResponse(code=409, message="Email already exists")

            user = create_user(
                self.db,
                email=payload.email.lower(),
                first_name=payload.firstName,
                last_name=payload.lastName,
                role_id=payload.roleId,
                is_active=bool(payload.isActive) if payload.isActive is not None else True,
                created_by=created_by.id,
            )

            self.db.commit()
            return AddUserSuccessResponse(code=200, message="User added successfully")
        
        except IntegrityError as e:
            self.db.rollback()
            logger.error(f"Integrity Error (add_user): {str(e)}")
            return AddUserFailureResponse(code=409, message="Database integrity error")

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database Error (add_user): {str(e)}")
            return AddUserFailureResponse(code=500, message="Database Error Occurred")

        except Exception as e:
            self.db.rollback()
            logger.error(f"Internal Error (add_user): {str(e)}")
            return AddUserFailureResponse(code=500, message="Internal Server Error")

    def update_user(self, payload: UpdateUserRequest, current: dict[str, Any]):
        try:
            logger.info("Update user requested")
            updated_by = current.get("user")
            if updated_by is None:
                return UpdateUserFailureResponse(code=401, message="Unauthorized user context")

            user = get_user_by_id(self.db, payload.userId)
            if not user:
                return UpdateUserFailureResponse(code=404, message="User not found")

            if not role_exists(self.db, payload.roleId):
                return UpdateUserFailureResponse(code=400, message="Invalid roleId")

            if payload.email != user.email:
                existing = get_user_by_email(self.db, payload.email)
                if existing and existing.id != user.id:
                    return UpdateUserFailureResponse(code=409, message="Email already exists")

            user = update_user(
                self.db,
                user=user,
                email=payload.email,
                first_name=payload.firstName,
                last_name=payload.lastName,
                role_id=payload.roleId,
                is_active=payload.isActive,
                updated_by=updated_by.id,
            )

            self.db.commit()
            return UpdateUserSuccessResponse(userId=user.id)
        
        except IntegrityError as e:
            self.db.rollback()
            logger.error(f"Integrity Error (update_user): {str(e)}")
            return UpdateUserFailureResponse(code=409, message="Database integrity error")

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database Error (update_user): {str(e)}")
            return UpdateUserFailureResponse(code=500, message="Database Error Occurred")

        except Exception as e:
            self.db.rollback()
            logger.error(f"Internal Error (update_user): {str(e)}")
            return UpdateUserFailureResponse(code=500, message="Internal Server Error")

    def get_user_detail(self, user_id: int) -> Union[GetUserDetailSuccessResponse, GetUserDetailFailureResponse]:
        try:
            logger.info(f"API requested to fetch user detail userId={user_id}")
            user = get_user_by_id(self.db, user_id)

            if not user:
                return GetUserDetailFailureResponse(code=404, message="User not found")

            user_info = UserInfoResponse(
                userId=user.id,
                username=user.email,
                firstName=user.first_name,
                lastName=user.last_name,
                roleId=user.role_id,
                roleName=user.role.name if user.role else None,
                status=user.is_active,
                createdAt=user.created_at,
                updatedAt=user.updated_at
            )

            return GetUserDetailSuccessResponse(code=200, message="User fetched successfully", user=user_info)

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database Error: {str(e)}")
            return GetUserDetailFailureResponse(code=500, message="Database Error Occurred")

        except Exception as e:
            logger.error(f"Internal Error: {str(e)}")
            return GetUserDetailFailureResponse(code=500, message="Internal Server Error")


    def delete_user(self, user_id: int, current: dict[str, Any]) -> Union[DeleteUserSuccessResponse, DeleteUserFailureResponse]:
        try:
            logger.info(f"API requested to delete user userId={user_id}")
            updated_by = current.get("user")
            if updated_by is None:
                return DeleteUserFailureResponse(code=401, message="Unauthorized user context")
            deleted = soft_delete_user_by_id(self.db, user_id, updated_by.id)
            if not deleted:
                return DeleteUserFailureResponse(code=404, message="User not found")

            return DeleteUserSuccessResponse(code=200, message="User deleted successfully")

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database Error: {str(e)}")
            return DeleteUserFailureResponse(code=500, message="Database Error Occurred")

        except Exception as e:
            logger.error(f"Internal Error: {str(e)}")
            return DeleteUserFailureResponse(code=500, message="Internal Server Error")
