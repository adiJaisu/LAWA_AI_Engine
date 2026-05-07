"""
Auth repository layer.

Handles:
- user lookup
- role permissions
"""

from typing import List, Optional, Dict
from sqlalchemy.orm import Session

from ai_vms.models.users import User
from ai_vms.models.roles import Role
from ai_vms.models.resource import Resource
from ai_vms.models.scope import Scope
from ai_vms.models.rolepermission import RolePermission


# ---------------------------------------------------------
# USER
# ---------------------------------------------------------
def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return (
        db.query(User)
        .filter(User.email == email)
        .filter(User.is_active.is_(True))
        .filter(User.is_delete.is_(False))
        .one_or_none()
    )


# ---------------------------------------------------------
# ROLE PERMISSIONS
# ---------------------------------------------------------
def get_role_permissions(db: Session, role_id: int) -> List[str]:
    if not role_id:
        return []

    role = (
        db.query(Role)
        .filter(Role.id == role_id, Role.is_active.is_(True))
        .one_or_none()
    )
    if not role:
        return []

    rows = (
        db.query(Resource.name, Scope.name)
        .join(RolePermission, RolePermission.resource_id == Resource.id)
        .join(Scope, Scope.id == RolePermission.scope_id)
        .filter(
            RolePermission.role_id == role.id,
            Resource.is_active.is_(True),
            Scope.is_active.is_(True),
        )
        .all()
    )

    return sorted(set([f"{r}:{s}" for r, s in rows]))


# ---------------------------------------------------------
# ADMIN CHECK
# ---------------------------------------------------------
def is_super_admin(permissions: List[str]) -> bool:
    return "*:*" in permissions