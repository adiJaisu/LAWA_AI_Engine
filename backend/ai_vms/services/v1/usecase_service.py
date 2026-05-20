from sqlalchemy.orm import Session
import ai_vms.crud.usecases as usecases_crud
from ai_vms.models.usecases import Usecase
from fastapi import HTTPException

from ai_vms.crud.usecases import (
    create_usecase,
    get_all_usecases,
    get_usecase_by_id,
    update_usecase,
    delete_usecase,
)

from ai_vms.schemas.usecases import (
    UsecaseCreateRequest,
    UsecaseUpdateRequest,UsecaseResponse
)


class AddUsecaseService:

    def __init__(self, db: Session):

        self.db = db


    def create_usecase(
        self,
        payload: UsecaseCreateRequest,
        actor_id: int
    ):
        
        existing_usecase = usecases_crud.get_usecase_by_name(
        self.db,
        payload.name
        )

        if existing_usecase:

            raise HTTPException(
                status_code=400,
                detail="Usecase already exists"
            )
            

        usecase = Usecase(
            name=payload.name,
            description=payload.description,
            tracking=payload.tracking,
            fps=payload.fps,
            batch=payload.batch,
            frame_skip=payload.frame_skip,
            ai_resource=payload.ai_resource,
            is_active=payload.is_active,
            created_by=actor_id,
            updated_by=actor_id,
        )

        return usecases_crud.create_usecase(
            self.db,
            usecase
        )


    # def get_all_usecases(self):

    #     usecases = usecases_crud.get_all_usecases(
    #         self.db
    #     )

    #     if not usecases:

    #         raise HTTPException(
    #             status_code=404,
    #             detail="No usecases found"
    #         )

    #     return {
    #         "code": 200,
    #         "message": "Usecases fetched successfully",
    #         "data": usecases
    #     }


    def get_all_usecases(self):

        usecases = usecases_crud.get_all_usecases(self.db)

        if not usecases:

            raise HTTPException(
                status_code=404,
                detail="No usecases found"
            )

        response_data = [
            UsecaseResponse(
                usecaseId=usecase.id,
                usecaseName=usecase.name,
                description=usecase.description,

                classes=[cls.class_name for cls in usecase.classes],

                tracking=usecase.tracking,
                fps=usecase.fps,
                batch=usecase.batch,

                frameSkip=usecase.frame_skip,
                aiResource=usecase.ai_resource,

                is_active=usecase.is_active,
                created_at=usecase.created_at,
                updated_at=usecase.updated_at
            )
            for usecase in usecases
        ]

        return {
            "code": 200,
            "message": "Usecases fetched successfully",
            "usecaseDetails": response_data
        }


    def get_usecase_detail(
        self,
        usecase_id: int
    ):

        usecase = usecases_crud.get_usecase_by_id(
        self.db,
        usecase_id
        )

        if not usecase:
            raise HTTPException(
                status_code=404,
                detail="Usecase not found"
            )

        return usecase


    def update_usecase(
        self,
        payload: UsecaseUpdateRequest,
        actor_id: int
    ):

        usecase = usecases_crud.get_usecase_by_id(
            self.db,
            payload.usecase_id
        )

        if not usecase:
            raise HTTPException(
                status_code=404,
                detail="Usecase not found"
            )

        if payload.name is not None:
            usecase.name = payload.name

        if payload.description is not None:
            usecase.description = payload.description

        if payload.tracking is not None:
            usecase.tracking = payload.tracking

        if payload.fps is not None:
            usecase.fps = payload.fps

        if payload.batch is not None:
            usecase.batch = payload.batch

        if payload.frame_skip is not None:
            usecase.frame_skip = payload.frame_skip

        if payload.ai_resource is not None:
            usecase.ai_resource = payload.ai_resource

        if payload.is_active is not None:
            usecase.is_active = payload.is_active

        usecase.updated_by = actor_id

        return usecases_crud.update_usecase(
            self.db,
            usecase
        )


    def delete_usecase(
        self,
        usecase_id: int
    ):

        usecase = usecases_crud.get_usecase_by_id(
            self.db,
            usecase_id
        )

        if not usecase:
            raise HTTPException(
                status_code=404,
                detail="Usecase not found"
            )

        usecases_crud.delete_usecase(
            self.db,
            usecase
        )

        return {
            "code": 200,
            "message": "Deleted successfully"
        }