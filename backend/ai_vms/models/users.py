from sqlalchemy import BigInteger, Boolean, Column, DateTime, ForeignKey, String, Index
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ai_vms.database.database_connection import Base

class User(Base):
    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True)
    email = Column(String(255), nullable=False, unique=True)
    hashed_password = Column(String(255), nullable=True) # Allow null for transition if needed, but should be required for local auth
    first_name = Column(String(120), nullable=True)
    last_name = Column(String(120), nullable=True)

    role_id = Column(BigInteger, ForeignKey("roles.id"), nullable=False)
    is_active = Column(Boolean, nullable=False, server_default="true")

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    created_by = Column(BigInteger, nullable=True)
    updated_by = Column(BigInteger, nullable=True)

    is_delete = Column(Boolean, nullable=False, server_default="false")

    role = relationship("Role", back_populates="users")

    __table_args__ = (
        Index("ix_users_role", "role_id"),
    )