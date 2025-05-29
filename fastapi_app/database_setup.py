# Import engine and Base from the local database module
from .database import engine, Base
# Import models to ensure they are registered with Base's metadata
from . import models # This ensures models.Item (and others) are known to Base.metadata

def create_db_tables():
    """
    Creates all database tables defined in the SQLAlchemy models
    that are registered with the 'Base' metadata.
    """
    print("Attempting to create database tables...")
    try:
        # Base.metadata should contain all table definitions from models
        # that inherited from this Base.
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully (if they didn't exist already)!")
    except Exception as e:
        print(f"Error creating database tables: {e}")

if __name__ == "__main__":
    print("Running database setup script...")
    create_db_tables()
