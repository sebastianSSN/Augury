from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional


# ── Auth ──────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    email: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type: str


# ── Model records ─────────────────────────────────────────────────────────────

class ModelRecordOut(BaseModel):
    id: int
    dataset_name: Optional[str]
    target_col: str
    n_classes: int
    accuracy: float
    n_samples: int
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}
