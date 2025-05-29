from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

# Correctly import from local packages/modules
from . import crud # This will have the CRUD functions
from . import models # This refers to sqlalchemy models (e.g., models.Item)
from . import schemas # This refers to pydantic models (e.g., schemas.Item)
from .database import get_db, engine # Removed SessionLocal, Base as they are not directly used here

# Table creation is handled by database_setup.py or alembic migrations in a real app
# models.Base.metadata.create_all(bind=engine) # This line can be used for quick setup if not using migrations

app = FastAPI(title="CRUD API with FastAPI and PostgreSQL")

@app.post("/items/", response_model=schemas.Item, status_code=201)
def create_new_item(item: schemas.ItemCreate, db: Session = Depends(get_db)):
    # Optional: Check for duplicate item name
    # db_item_by_name = crud.get_item_by_name(db, name=item.name) # Assuming get_item_by_name exists in crud
    # if db_item_by_name:
    #     raise HTTPException(status_code=400, detail="Item with this name already exists")
    return crud.create_item(db=db, item=item)

@app.get("/items/", response_model=List[schemas.Item])
def read_all_items(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    items = crud.get_items(db=db, skip=skip, limit=limit)
    return items

@app.get("/items/{item_id}", response_model=schemas.Item)
def read_single_item(item_id: int, db: Session = Depends(get_db)):
    db_item = crud.get_item(db=db, item_id=item_id)
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return db_item

@app.put("/items/{item_id}", response_model=schemas.Item)
def update_existing_item(item_id: int, item: schemas.ItemUpdate, db: Session = Depends(get_db)):
    # The crud.update_item function should handle the logic of fetching the existing item
    # and then updating it, or returning None if not found.
    db_item = crud.update_item(db=db, item_id=item_id, item_update_schema=item)
    if db_item is None:
        # This means the item to update was not found by crud.update_item
        raise HTTPException(status_code=404, detail="Item not found, cannot update")
    return db_item

@app.delete("/items/{item_id}", response_model=schemas.Item)
def delete_existing_item(item_id: int, db: Session = Depends(get_db)):
    # The crud.delete_item function should handle the logic of fetching the existing item
    # and then deleting it, or returning None if not found.
    db_item = crud.delete_item(db=db, item_id=item_id)
    if db_item is None:
        # This means the item to delete was not found by crud.delete_item
        raise HTTPException(status_code=404, detail="Item not found, cannot delete")
    return db_item # Returns the deleted item as confirmation (FastAPI will serialize it via schemas.Item)
