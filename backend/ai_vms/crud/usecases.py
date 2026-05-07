"""
Repository layer for Usecase operations.

Includes retrieval of all defined usecases using SQLAlchemy ORM.

Author: HCLTech
"""

from sqlalchemy.orm import joinedload, Session
from ai_vms.models.usecases import Usecase

# Fetch all available usecases sorted by ID
def get_all_usecases(db: Session):
    return (
        db.query(Usecase)
        .options(joinedload(Usecase.classes))
        .order_by(Usecase.id.asc())
        .all()
    )
