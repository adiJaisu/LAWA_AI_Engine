from sqlalchemy.orm import Session,joinedload

from ai_vms.models.usecases import Usecase


def create_usecase(db: Session, usecase: Usecase):

    db.add(usecase)

    db.commit()

    db.refresh(usecase)

    return usecase

def get_usecase_by_name(
    db: Session,
    name: str
):

    return (
        db.query(Usecase)
        .filter(Usecase.name == name)
        .first()
    )

def get_all_usecases(db: Session):

    return (
        db.query(Usecase)
        .options(joinedload(Usecase.classes))
        .order_by(Usecase.id.desc())
        .all()
    )


def get_usecase_by_id(
    db: Session,
    usecase_id: int
):

    return (
        db.query(Usecase)
        .filter(Usecase.id == usecase_id)
        .first()
    )


def update_usecase(
    db: Session,
    usecase: Usecase
):

    db.commit()

    db.refresh(usecase)

    return usecase


def delete_usecase(
    db: Session,
    usecase: Usecase
):

    db.delete(usecase)

    db.commit()