"""
Repository layer for Camera operations.

Includes camera CRUD, validations, and usecase mapping updates using SQLAlchemy ORM.

Author: HCLTech
"""

from typing import List, Optional, Any, Dict

from sqlalchemy.orm import Session, joinedload

from ai_vms.models.camera import Camera
from ai_vms.models.camera_usecase import CameraUsecase
from ai_vms.models.usecases import Usecase

# Fetch all cameras
def get_all_cameras(db: Session) -> List[Camera]:
    return (
        db.query(Camera)
        .options(
            joinedload(Camera.usecase_links)
                .joinedload(CameraUsecase.usecase),
        )
        .filter(Camera.is_delete.is_(False))
        .order_by(Camera.id.desc())
        .all()
    )


# Fetch a single camera by ID
def get_camera_by_id(db: Session, camera_id: int) -> Optional[Camera]:
    return (
        db.query(Camera)
        .options(
            joinedload(Camera.usecase_links)
                .joinedload(CameraUsecase.usecase),
        )
        .filter(Camera.id == camera_id)
        .one_or_none()
    )


# Fetch a single camera by its name
def get_camera_by_name(db: Session, name: str) -> Optional[Camera]:
    return (
        db.query(Camera)
        .filter(Camera.name == name)
        .one_or_none()
    )


# Insert a new camera record and return the saved entity
def create_camera(db: Session, camera: Camera) -> Camera:
    db.add(camera)
    db.commit()
    db.refresh(camera)
    return camera


# Persist updates on the given camera entity and return the refreshed entity
def update_camera(db: Session, camera: Camera) -> Camera:
    db.commit()
    db.refresh(camera)
    return camera


# Soft-delete a camera
def soft_delete_camera(db: Session, camera: Camera) -> Camera:
    camera.name = camera.name + '-'+ str(camera.id)
    camera.is_delete = True
    db.commit()
    db.refresh(camera)
    return camera


# Replace all usecases mapped to a camera
def replace_camera_usecases(db: Session, camera_id: int, usecase_ids: List[int], actor_id: int):
    # Remove existing mappings for this camera
    db.query(CameraUsecase).filter(CameraUsecase.camera_id == camera_id).delete(
        synchronize_session=False
    )

    # Create new mappings
    for uc_id in set(usecase_ids or []):
        db.add(
            CameraUsecase(
                camera_id=camera_id,
                usecase_id=uc_id,
                created_by=actor_id,
                updated_by=actor_id,
            )
        )
    db.commit()

#fetch respective camera roi from db
def get_camera_roi(db: Session, camera_id: int) -> Optional[Dict[str, Any]]:
    camera = (
        db.query(Camera)
        .filter(Camera.id == camera_id, Camera.is_delete.is_(False))
        .one_or_none()
    )
    if not camera:
        return None
    return camera.roi

#create/save respective camera roi
def set_camera_roi(db: Session, camera_id: int, roi_payload: Dict[str, Any], actor_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    camera = (
        db.query(Camera)
        .filter(Camera.id == camera_id, Camera.is_delete.is_(False))
        .one_or_none()
    )
    if not camera:
        return None

    # Overwrite the JSON column
    camera.roi = roi_payload
    if hasattr(camera, "updated_by") and actor_id is not None:
        camera.updated_by = actor_id
    
    db.commit()
    db.refresh(camera)
    return camera.roi

#set respective camere roi to null
def clear_camera_roi(db: Session, camera_id: int, actor_id: Optional[int] = None) -> bool:
    camera = (
        db.query(Camera)
        .filter(Camera.id == camera_id, Camera.is_delete.is_(False))
        .one_or_none()
    )
    if not camera:
        return False

    camera.roi = None
    if hasattr(camera, "updated_by") and actor_id is not None:
        camera.updated_by = actor_id

    db.commit()
    return True

def get_camera_frame_blob(db: Session, camera_id: int) -> Optional[bytes]:
    camera = (
        db.query(Camera)
        .filter(Camera.id == camera_id, Camera.is_delete.is_(False))
        .one_or_none()
    )
    if not camera:
        return None
    return getattr(camera, "roi_frame_blob", None)

def set_camera_frame_blob(
    db: Session,
    camera_id: int,
    frame_blob: Optional[bytes],
    actor_id: Optional[int] = None
) -> bool:
    camera = (
        db.query(Camera)
        .filter(Camera.id == camera_id, Camera.is_delete.is_(False))
        .one_or_none()
    )
    if not camera or not hasattr(camera, "roi_frame_blob"):
        return False

    camera.roi_frame_blob = frame_blob
    if hasattr(camera, "updated_by") and actor_id is not None:
        camera.updated_by = actor_id
    db.commit()
    db.refresh(camera)
    return True


def validate_usecase_ids(db: Session, usecase_ids: List[int]) -> List[int]:
    """
    Validates that the given usecase IDs exist in the database.
    
    Args:
        db: Database session
        usecase_ids: List of usecase IDs to validate
        
    Returns:
        List of valid usecase IDs that exist in the database
    """
    if not usecase_ids:
        return []
    
    valid_usecases = db.query(Usecase.id).filter(Usecase.id.in_(usecase_ids)).all()
    return [uc.id for uc in valid_usecases]
