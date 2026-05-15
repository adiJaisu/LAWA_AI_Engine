from sqlalchemy import BigInteger, Boolean, Column, DateTime, String
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ai_vms.database.database_connection import Base

class Resource(Base):
    __tablename__ = "resources"

    id = Column(BigInteger, primary_key=True)
    name = Column(String(120), nullable=False, unique=True)
    is_active = Column(Boolean, nullable=False, server_default="true")

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    created_by = Column(BigInteger, nullable=True)
    updated_by = Column(BigInteger, nullable=True)

    permissions = relationship("RolePermission", back_populates="resource", cascade="all, delete-orphan")