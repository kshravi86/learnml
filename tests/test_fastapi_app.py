import unittest
import json
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Adjust imports based on your project structure
from fastapi_app.main import app # Your FastAPI app instance
from fastapi_app.database import Base, get_db # Base for tables, get_db for overriding
from fastapi_app import models # Your SQLAlchemy models
from fastapi_app import schemas # Your Pydantic models

SQLALCHEMY_DATABASE_URL_TEST = "sqlite:///:memory:" # Use in-memory SQLite for tests

test_engine = create_engine(SQLALCHEMY_DATABASE_URL_TEST)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

# Dependency override for get_db
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

class FastAPITests(unittest.TestCase):

    def setUp(self):
        # Create all tables in the test database
        Base.metadata.create_all(bind=test_engine)
        self.client = TestClient(app)
        # Helper to add an item directly to DB for testing GET/PUT/DELETE
        # This item will typically have id=1 if it's the first one.
        self.initial_item = self._add_item_to_db("Initial Item", "Initial Description")

    def tearDown(self):
        # Drop all tables in the test database
        Base.metadata.drop_all(bind=test_engine)
        
    def _add_item_to_db(self, name: str, description: str) -> models.Item:
        db = TestingSessionLocal() # Create a new session for this helper
        item = models.Item(name=name, description=description)
        db.add(item)
        db.commit()
        db.refresh(item)
        db.close()
        return item

    def test_create_item_api(self):
        payload = {"name": "New Test Item", "description": "New Test Description"}
        response = self.client.post("/items/", json=payload)
        self.assertEqual(response.status_code, 201, f"Response JSON: {response.json()}")
        data = response.json()
        self.assertEqual(data["name"], payload["name"])
        self.assertEqual(data["description"], payload["description"])
        self.assertTrue("id" in data)
        
        # Check if item exists in DB
        db = TestingSessionLocal()
        db_item = db.query(models.Item).filter(models.Item.id == data["id"]).first()
        self.assertIsNotNone(db_item)
        self.assertEqual(db_item.name, payload["name"])
        db.close()

    def test_read_items_api(self):
        # Add another item to ensure we test reading multiple items
        self._add_item_to_db("Second Item", "Second Description")
        
        response = self.client.get("/items/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        # We added one in setUp and one here
        self.assertEqual(len(data), 2, f"Expected 2 items, got {len(data)}. Response: {data}")
        
        # Check if the initial item is present
        found_initial = any(item['name'] == self.initial_item.name for item in data)
        self.assertTrue(found_initial, "Initial item not found in the list")

    def test_read_one_item_api(self):
        # Use the ID of the item created in setUp
        item_id = self.initial_item.id
        response = self.client.get(f"/items/{item_id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["name"], self.initial_item.name)
        self.assertEqual(data["id"], item_id)

        response_not_found = self.client.get("/items/9999") # Assuming 9999 does not exist
        self.assertEqual(response_not_found.status_code, 404)

    def test_update_item_api(self):
        item_id = self.initial_item.id
        update_payload = {"name": "Updated Item Name", "description": "Updated Description"}
        response = self.client.put(f"/items/{item_id}", json=update_payload)
        self.assertEqual(response.status_code, 200, f"Response JSON: {response.json()}")
        data = response.json()
        self.assertEqual(data["name"], update_payload["name"])
        self.assertEqual(data["description"], update_payload["description"])
        
        # Check DB
        db = TestingSessionLocal()
        updated_db_item = db.query(models.Item).filter(models.Item.id == item_id).first()
        self.assertIsNotNone(updated_db_item)
        self.assertEqual(updated_db_item.name, update_payload["name"])
        self.assertEqual(updated_db_item.description, update_payload["description"])
        db.close()
        
        response_not_found = self.client.put("/items/9999", json=update_payload) # Assuming 9999 does not exist
        self.assertEqual(response_not_found.status_code, 404)

    def test_delete_item_api(self):
        item_id = self.initial_item.id
        response = self.client.delete(f"/items/{item_id}")
        self.assertEqual(response.status_code, 200, f"Response JSON: {response.json()}")
        data = response.json() # Check what the delete operation returns
        self.assertEqual(data["id"], item_id) # Assuming the deleted item's data is returned
        
        # Verify item is deleted from DB
        db = TestingSessionLocal()
        deleted_item = db.query(models.Item).filter(models.Item.id == item_id).first()
        self.assertIsNone(deleted_item)
        db.close()

        response_not_found = self.client.delete("/items/9999") # Assuming 9999 does not exist
        self.assertEqual(response_not_found.status_code, 404)

if __name__ == "__main__":
    unittest.main()
