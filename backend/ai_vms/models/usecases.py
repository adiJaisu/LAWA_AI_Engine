from sqlalchemy import BigInteger, Boolean, Column, DateTime, String, Enum, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ai_vms.database.database_connection import Base
from ai_vms.models.enums import AIResource

class Usecase(Base):
    __tablename__ = "usecases"

    id = Column(BigInteger, primary_key=True)
    name = Column(String(120), nullable=False, unique=True)
    description = Column(String(500), nullable=True)
    tracking = Column(Boolean, nullable=False, server_default="false")
    fps = Column(Float, nullable=True)
    batch = Column(BigInteger, nullable=True)
    frame_skip = Column(BigInteger, nullable=True)
    ai_resource = Column(
        Enum(AIResource, name="ai_resource_type", native_enum=True),
        nullable=False,
        server_default=AIResource.cpu.name,
    )
    is_active = Column(Boolean, nullable=False, server_default="true")

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    created_by = Column(BigInteger, nullable=True)
    updated_by = Column(BigInteger, nullable=True)

    camera_links = relationship("CameraUsecase", back_populates="usecase", cascade="all, delete-orphan")
    classes = relationship("UsecaseClass", back_populates="usecase", cascade="all, delete-orphan")
