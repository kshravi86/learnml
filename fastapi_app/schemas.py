from pydantic import BaseModel
from typing import Optional

# Schema for item attributes that are common
class ItemBase(BaseModel):
    name: str
    description: Optional[str] = None

# Schema for creating an item (inherits from ItemBase)
class ItemCreate(ItemBase):
    pass # name is required (from ItemBase), description is optional (from ItemBase)

# Schema for updating an item (all fields optional)
# If we want to allow partial updates, all fields here must be Optional.
# Since ItemBase already makes description optional, we only need to consider name.
class ItemUpdate(BaseModel): # Or inherit from ItemBase and make fields optional as needed
    name: Optional[str] = None
    description: Optional[str] = None

# Schema for representing an item in API responses (includes id)
class Item(ItemBase):
    id: int

    class Config:
        orm_mode = True # Allows Pydantic to work with ORM objects
