from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Float, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base


class User(Base):
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    email           = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())

    model_records   = relationship("ModelRecord", back_populates="owner", cascade="all, delete-orphan")


class ModelRecord(Base):
    """Metadata of a trained model. Binary files live on disk under MODEL_DIR/{user_id}/."""
    __tablename__ = "model_records"

    id           = Column(Integer, primary_key=True, index=True)
    user_id      = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    dataset_name = Column(String, nullable=True)
    target_col   = Column(String, nullable=False)
    feature_cols = Column(Text, nullable=False)   # JSON-encoded list
    n_classes    = Column(Integer, nullable=False)
    accuracy     = Column(Float, nullable=False)
    n_samples    = Column(Integer, nullable=False)
    is_active    = Column(Boolean, default=True)  # marks the current model for the user
    created_at   = Column(DateTime(timezone=True), server_default=func.now())

    owner = relationship("User", back_populates="model_records")
