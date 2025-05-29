from sqlalchemy import Column, Integer, String
from .database import Base

class Item(Base):
    __tablename__ = "items"  # Explicitly define table name

    id = Column(Integer, primary_key=True, index=True) # Add index=True for frequently queried columns
    name = Column(String(80), nullable=False, index=True)
    description = Column(String(200), nullable=True)

    # __repr__ is optional for FastAPI but good for debugging
    def __repr__(self):
        return f"<Item(id={self.id}, name='{self.name}')>"
