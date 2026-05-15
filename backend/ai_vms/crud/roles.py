"""
Repository layer for Role operations.

Includes role retrieval and existence validation using SQLAlchemy ORM.

Author: HCLTech
"""


from sqlalchemy.orm import joinedload, Session
from ai_vms.models.roles import Role

# Fetch all roles ordered by latest created first
def get_all_roles(db: Session):
    return (
        db.query(Role)
        .order_by(Role.id.desc())
        .all()
    )


# Check if a role exists in the database
def role_exists(db: Session, role_id: int) -> bool:
    return db.query(Role.id).filter(Role.id == role_id).first() is not None

