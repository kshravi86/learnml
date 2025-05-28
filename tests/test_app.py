import unittest
import json
from flask_app.app import app, db
from flask_app.models import Item

class BasicTests(unittest.TestCase):

    def setUp(self):
        """Set up test variables."""
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        app.config['WTF_CSRF_ENABLED'] = False # Disable CSRF for testing forms if applicable
        self.app_context = app.app_context()
        self.app_context.push()
        db.create_all()
        self.client = app.test_client()

    def tearDown(self):
        """Tear down all initialized variables."""
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    # --- Helper Methods ---
    def _create_item_in_db(self, name="Test Item", description="Test Description"):
        item = Item(name=name, description=description)
        db.session.add(item)
        db.session.commit()
        return item

    # --- JSON API Tests ---
    def test_create_item_json(self):
        payload = {'name': 'New Item JSON', 'description': 'Created via JSON API'}
        response = self.client.post('/items', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode())
        self.assertEqual(data['name'], payload['name'])
        self.assertTrue(Item.query.filter_by(name=payload['name']).first() is not None)

    def test_get_all_items_json(self):
        self._create_item_in_db(name="Item 1")
        self._create_item_in_db(name="Item 2")
        response = self.client.get('/items')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode())
        self.assertEqual(len(data), 2)

    def test_get_one_item_json(self):
        item = self._create_item_in_db(name="Specific Item")
        response = self.client.get(f'/items/{item.id}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode())
        self.assertEqual(data['name'], item.name)

        # Test 404
        response_404 = self.client.get('/items/9999')
        self.assertEqual(response_404.status_code, 404)

    def test_update_item_json(self):
        item = self._create_item_in_db(name="Old Name")
        payload = {'name': 'Updated Name JSON', 'description': 'Updated via JSON API'}
        response = self.client.put(f'/items/{item.id}', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode())
        self.assertEqual(data['name'], payload['name'])
        updated_item = Item.query.get(item.id)
        self.assertEqual(updated_item.name, payload['name'])

        # Test 404
        response_404 = self.client.put('/items/9999', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response_404.status_code, 404)

    def test_delete_item_json(self):
        item = self._create_item_in_db(name="To Be Deleted")
        response = self.client.delete(f'/items/{item.id}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode())
        self.assertEqual(data['message'], 'Item deleted successfully')
        self.assertIsNone(Item.query.get(item.id))

        # Test 404
        response_404 = self.client.delete('/items/9999')
        self.assertEqual(response_404.status_code, 404)

    # --- HTML UI Route Tests ---
    def test_list_items_html(self):
        self._create_item_in_db(name="HTML Item")
        response = self.client.get('/ui/items')
        self.assertEqual(response.status_code, 200)
        content = response.data.decode()
        self.assertIn("HTML Item", content)
        self.assertIn("<h1>Items</h1>", content)

    def test_create_item_html_form_submission(self):
        payload = {'name': 'New Item HTML', 'description': 'Created via HTML form'}
        response = self.client.post('/ui/items', data=payload, follow_redirects=True)
        self.assertEqual(response.status_code, 200) # After redirect
        self.assertIn(payload['name'], response.data.decode())
        self.assertTrue(Item.query.filter_by(name=payload['name']).first() is not None)

    def test_update_item_html_form_submission(self):
        item = self._create_item_in_db(name="Old HTML Name")
        payload = {'name': 'Updated HTML Name', 'description': 'Updated via HTML form'}
        response = self.client.post(f'/ui/items/{item.id}/update', data=payload, follow_redirects=True)
        self.assertEqual(response.status_code, 200) # After redirect
        self.assertIn(payload['name'], response.data.decode())
        updated_item = Item.query.get(item.id)
        self.assertEqual(updated_item.name, payload['name'])

    def test_delete_item_html_form_submission(self):
        item = self._create_item_in_db(name="To Be Deleted HTML")
        response = self.client.post(f'/ui/items/{item.id}/delete', follow_redirects=True)
        self.assertEqual(response.status_code, 200) # After redirect
        self.assertNotIn("To Be Deleted HTML", response.data.decode())
        self.assertIsNone(Item.query.get(item.id))

if __name__ == "__main__":
    unittest.main()
