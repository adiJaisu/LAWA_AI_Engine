"""
Repository layer for User operations.

Includes user retrieval, creation, updates, and soft deletion using SQLAlchemy ORM.

Author: HCLTech
"""

from sqlalchemy.orm import Session, joinedload
from ai_vms.models.users import User


# Fetch all users with role
# Only returns users that are not soft-deleted
def get_all_users(db: Session):
    return (
        db.query(User)
        .options(
            joinedload(User.role),
        )
        .filter(User.is_delete.is_(False))
        .order_by(User.id.desc())
        .all()
    )


# Fetch a single user by ID
def get_user_by_id(db: Session, user_id: int) -> User | None:
    return (
        db.query(User)
        .options(
            joinedload(User.role),
        )
        .filter(User.id == user_id)
        .one_or_none()
    )


# Fetch a single active user by email
def get_user_by_email(db: Session, email: str) -> User | None:
    return (
        db.query(User)
        .options(
            joinedload(User.role),
        )
        .filter(User.email == email, User.is_active == True)
        .one_or_none()
    )


# Soft-delete a user by marking the record as deleted
# Returns True if user exists and was deleted, otherwise returns False
def soft_delete_user_by_id(db: Session, user_id: int, deleted_by: int | None = None) -> bool:
    # Find the user only if not already deleted
    user = db.query(User).filter(User.id == user_id, User.is_delete == False).one_or_none()
    if not user:
        return False

    # Mark user as deleted and update audit fields
    user.email = user.email+'-'+str(user.id)
    user.is_delete = True
    user.updated_by = deleted_by
    db.commit()
    return True


# Create a new user record and return the saved entity
def create_user(
    db: Session,
    *,
    email: str,
    first_name: str | None,
    last_name: str | None,
    role_id: int,
    is_active: bool = True,
    created_by: int | None = None,
    hashed_password: str | None = None
) -> User:

    # Build the user object with provided details
    user = User(
        email=email,
        hashed_password=hashed_password,
        first_name=first_name,
        last_name=last_name,
        role_id=role_id,
        is_active=is_active,
        created_by=created_by,
        updated_by=created_by
    )

    # Save the user to database
    db.add(user)
    db.commit()
    db.refresh(user)

    return user


# Update an existing user record and return the updated entity
def update_user(
    db: Session,
    *,
    user: User,
    email: str,
    first_name: str | None,
    last_name: str | None,
    role_id: int,
    is_active: bool | None = None,
    updated_by: int | None = None
) -> User:

    # Update basic user fields
    user.email = email
    user.first_name = first_name
    user.last_name = last_name
    user.role_id = role_id

    # Update active flag only when explicitly passed
    if is_active is not None:
        user.is_active = is_active

    # Update audit field
    user.updated_by = updated_by

    # Persist updates
    db.commit()
    db.refresh(user)

    return user