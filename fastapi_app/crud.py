from sqlalchemy.orm import Session

# Import SQLAlchemy models and Pydantic schemas
from . import models # This refers to sqlalchemy models (e.g., models.Item)
from . import schemas # This refers to pydantic models (e.g., schemas.ItemCreate, schemas.ItemUpdate)

def get_item(db: Session, item_id: int):
    """
    Retrieves a single item by its ID.
    """
    return db.query(models.Item).filter(models.Item.id == item_id).first()

def get_item_by_name(db: Session, name: str):
    """
    Retrieves a single item by its name.
    Useful for checking for duplicates.
    """
    return db.query(models.Item).filter(models.Item.name == name).first()

def get_items(db: Session, skip: int = 0, limit: int = 100):
    """
    Retrieves a list of items with pagination.
    """
    return db.query(models.Item).offset(skip).limit(limit).all()

def create_item(db: Session, item: schemas.ItemCreate):
    """
    Creates a new item in the database.
    """
    # Create an instance of the SQLAlchemy model from the Pydantic schema data
    db_item = models.Item(name=item.name, description=item.description)
    db.add(db_item)
    db.commit()
    db.refresh(db_item) # Refresh to get any DB-generated values like the ID
    return db_item

def update_item(db: Session, item_id: int, item_update_schema: schemas.ItemUpdate):
    """
    Updates an existing item in the database.
    Allows for partial updates.
    """
    db_item = db.query(models.Item).filter(models.Item.id == item_id).first()
    if db_item:
        # Get data from the Pydantic schema, excluding unset fields for partial updates
        update_data = item_update_schema.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_item, key, value)
        db.commit()
        db.refresh(db_item)
    return db_item # Returns the updated item, or None if not found

def delete_item(db: Session, item_id: int):
    """
    Deletes an item from the database.
    """
    db_item = db.query(models.Item).filter(models.Item.id == item_id).first()
    if db_item:
        db.delete(db_item)
        db.commit()
    # It's common to return the deleted item (or its representation) or None/True/False
    # Returning the item allows the route to potentially return its data.
    # If the item was deleted, db_item will hold its data just before deletion.
    # However, after commit, it might be expired. For this pattern, we rely on what was fetched.
    # If the goal is to confirm deletion and nothing more, return True or some status.
    # The example returns the item, so we follow that.
    return db_item # Returns the item that was deleted, or None if not found
