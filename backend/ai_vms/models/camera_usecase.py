from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, UniqueConstraint, Index
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ai_vms.database.database_connection import Base

class CameraUsecase(Base):
    __tablename__ = "camera_usecase"

    id = Column(BigInteger, primary_key=True)

    camera_id = Column(BigInteger, ForeignKey("cameras.id"), nullable=False)
    usecase_id = Column(BigInteger, ForeignKey("usecases.id"), nullable=False)  
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    created_by = Column(BigInteger, nullable=True)
    updated_by = Column(BigInteger, nullable=True)

    __table_args__ = (
        UniqueConstraint("camera_id", "usecase_id", name="uq_camera_usecase"),  
        Index("ix_camera_usecase_camera", "camera_id"),
        Index("ix_camera_usecase_usecase", "usecase_id"),
    )

    camera = relationship("Camera", back_populates="usecase_links")
    usecase = relationship("Usecase", back_populates="camera_links")