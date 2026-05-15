from sqlalchemy import BigInteger, Boolean, Column, DateTime, ForeignKey, String, Enum, Index, LargeBinary, Float
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ai_vms.database.database_connection import Base
from .enums import CameraType, DecodingResource, DecodingPipeline, DockerMode

class Camera(Base):
    __tablename__ = "cameras"

    id = Column(BigInteger, primary_key=True)
    name = Column(String(150), nullable=False, unique=True)

    type = Column(Enum(CameraType, name="camera_type", native_enum=True), nullable=False)
    rtsp_url = Column(String(500), nullable=True)
    roi_frame_blob = Column(LargeBinary, nullable=True)
    resolution = Column(String(50), nullable=True)
    height = Column(Float, nullable=True)
    resolution_width = Column(Float, nullable=True)
    resolution_height = Column(Float, nullable=True)
    decoding_resource = Column(
        Enum(DecodingResource, name="decoding_resource_type", native_enum=True),
        nullable=False,
        server_default=DecodingResource.cpu.name,
    )
    decoding_pipeline = Column(
        Enum(DecodingPipeline, name="decoding_pipeline_type", native_enum=True),
        nullable=False,
        server_default=DecodingPipeline.ffmpeg.name,
    )
    docker_mode = Column(
        Enum(DockerMode, name="docker_mode_type", native_enum=True),
        nullable=False,
        server_default=DockerMode.load.name,
    )
    fps = Column(BigInteger, nullable=True)
    codec = Column(String(50), nullable=True)

    roi = Column(JSONB, nullable=True)

    is_active = Column(Boolean, nullable=False, server_default="true")

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    created_by = Column(BigInteger, nullable=True)
    updated_by = Column(BigInteger, nullable=True)
    is_delete = Column(Boolean, nullable=False, server_default="false")

    usecase_links = relationship("CameraUsecase", back_populates="camera", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_cameras_created_at", "created_at"),
    )
