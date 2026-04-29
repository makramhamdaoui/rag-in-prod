import uuid
import enum
from datetime import datetime, timezone

from sqlalchemy import String, Integer, BigInteger, DateTime, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column

from src.database.connection import Base


class DocumentStatus(str, enum.Enum):
    pending = "pending"
    indexed = "indexed"
    failed  = "failed"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    path: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[DocumentStatus] = mapped_column(
        SAEnum(DocumentStatus), default=DocumentStatus.pending, nullable=False
    )
    num_chunks: Mapped[int] = mapped_column(Integer, default=0)
    num_pages:  Mapped[int] = mapped_column(Integer, default=0)
    file_size:  Mapped[int] = mapped_column(BigInteger, default=0)
    indexed_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self):
        return {
            "id":         self.id,
            "name":       self.name,
            "path":       self.path,
            "status":     self.status,
            "num_chunks": self.num_chunks,
            "num_pages":  self.num_pages,
            "file_size":  self.file_size,
            "indexed_at": self.indexed_at.isoformat() if self.indexed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
