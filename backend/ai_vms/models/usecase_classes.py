from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ai_vms.database.database_connection import Base


class UsecaseClass(Base):
    __tablename__ = "usecase_classes"

    id = Column(BigInteger, primary_key=True)
    usecase_id = Column(
        BigInteger,
        ForeignKey("usecases.id", ondelete="CASCADE"),
        nullable=False,
    )
    class_name = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    usecase = relationship("Usecase", back_populates="classes")

    __table_args__ = (
        UniqueConstraint("usecase_id", "class_name", name="uq_usecase_classes_usecase_class_name"),
    )
