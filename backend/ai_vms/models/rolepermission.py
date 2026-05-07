from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ai_vms.database.database_connection import Base

class RolePermission(Base):
    __tablename__ = "role_permissions"

    id = Column(BigInteger, primary_key=True)

    role_id = Column(BigInteger, ForeignKey("roles.id"), nullable=False)
    resource_id = Column(BigInteger, ForeignKey("resources.id"), nullable=False)
    scope_id = Column(BigInteger, ForeignKey("scopes.id"), nullable=False)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    created_by = Column(BigInteger, nullable=True)
    updated_by = Column(BigInteger, nullable=True)

    __table_args__ = (
        UniqueConstraint("role_id", "resource_id", "scope_id", name="uq_role_permissions"),
    )

    role = relationship("Role", back_populates="permissions")
    resource = relationship("Resource", back_populates="permissions")
    scope = relationship("Scope", back_populates="permissions")