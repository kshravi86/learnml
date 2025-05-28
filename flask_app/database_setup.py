from flask_app.app import app, db
from flask_app.models import Item # Assuming Item is your primary model, add others if exist

def create_tables():
    """Creates all database tables defined in the models."""
    with app.app_context():
        try:
            print("Attempting to create database tables...")
            db.create_all()
            print("Database tables created successfully!")
        except Exception as e:
            print(f"Error creating database tables: {e}")

if __name__ == '__main__':
    print("Running database setup script...")
    create_tables()
