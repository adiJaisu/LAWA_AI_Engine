from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ai_vms.database.database_connection import get_db

from ai_vms.services.v1.usecase_service import AddUsecaseService
from typing import List

from ai_vms.schemas.usecases import (
    UsecaseCreateRequest,
    UsecaseUpdateRequest,
    UsecaseResponse,
    GetAllUsecasesResponse,
)

router = APIRouter()


@router.post("/addusecase",response_model=UsecaseResponse)
def add_usecase(
    payload: UsecaseCreateRequest,
    db: Session = Depends(get_db)
):

    service = AddUsecaseService(db)

    return service.create_usecase(
        payload,
        actor_id=1
    )


@router.get("/getallusecases", response_model=GetAllUsecasesResponse)
def get_all_usecases(
    db: Session = Depends(get_db)
):

    service = AddUsecaseService(db)

    return service.get_all_usecases()


@router.get("/getusecase/{usecase_id}", response_model=UsecaseResponse)
def get_usecase(
    usecase_id: int,
    db: Session = Depends(get_db)
):

    service = AddUsecaseService(db)

    return service.get_usecase_detail(
        usecase_id
    )


@router.put("/updateusecase", response_model=UsecaseResponse)
def update_usecase(
    payload: UsecaseUpdateRequest,
    db: Session = Depends(get_db)
):

    service = AddUsecaseService(db)

    return service.update_usecase(
        payload,
        actor_id=1
    )


@router.delete("/deleteusecase/{usecase_id}")
def delete_usecase(
    usecase_id: int,
    db: Session = Depends(get_db)
):

    service = AddUsecaseService(db)

    return service.delete_usecase(
        usecase_id
    )







## Below code is with auth, uncomment to use auth and comment above code

# from fastapi import APIRouter, Depends
# from sqlalchemy.orm import Session

# from ai_vms.database.database_connection import get_db

# from ai_vms.services.v1.add_usecases_service import AddUsecaseService
# from typing import List

# from ai_vms.schemas.add_usecases import (
#     UsecaseCreateRequest,
#     UsecaseUpdateRequest,
#     UsecaseResponse,
#     GetAllUsecasesResponse
# )

# from ai_vms.utils.auth.auth import require_permission

# router = APIRouter()


# @router.post("/addusecase",response_model=UsecaseResponse)
# def add_usecase(
#     payload: UsecaseCreateRequest,
#     db: Session = Depends(get_db),
#     current_user=Depends(require_permission("usecase:create")),
# ):

#     service = AddUsecaseService(db)

#     return service.create_usecase(
#         payload,
#         actor_id=current_user["user"].id
#     )


# @router.get("/getallusecases", response_model=GetAllUsecasesResponse)
# def get_all_usecases(
#     db: Session = Depends(get_db),
#     current_user=Depends(require_permission("usecase:read")),
# ):

#     service = AddUsecaseService(db)

#     return service.get_all_usecases()


# @router.get("/getusecase/{usecase_id}", response_model=UsecaseResponse)
# def get_usecase(
#     usecase_id: int,
#     db: Session = Depends(get_db),
#     current_user=Depends(require_permission("usecase:read")),
# ):

#     service = AddUsecaseService(db)

#     return service.get_usecase_detail(
#         usecase_id
#     )


# @router.put("/updateusecase", response_model=UsecaseResponse)
# def update_usecase(
#     payload: UsecaseUpdateRequest,
#     db: Session = Depends(get_db),
#     current_user=Depends(require_permission("usecase:update")),
# ):

#     service = AddUsecaseService(db)

#     return service.update_usecase(
#         payload,
#         actor_id=current_user["user"].id
#     )


# @router.delete("/deleteusecase/{usecase_id}")
# def delete_usecase(
#     usecase_id: int,
#     db: Session = Depends(get_db),
#     current_user=Depends(require_permission("usecase:delete")),
# ):

#     service = AddUsecaseService(db)

#     return service.delete_usecase(
#         usecase_id
#     )