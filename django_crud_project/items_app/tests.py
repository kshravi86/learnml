from django.test import TestCase, Client
from django.urls import reverse
from .models import Item
from .forms import ItemForm

class ItemModelTests(TestCase):
    def test_item_creation(self):
        item = Item.objects.create(name="Test Item", description="A test description.")
        self.assertEqual(item.name, "Test Item")
        self.assertEqual(item.description, "A test description.")
        self.assertEqual(str(item), "Test Item")

class ItemFormTests(TestCase):
    def test_item_form_valid(self):
        form_data = {'name': 'Valid Item', 'description': 'Valid description.'}
        form = ItemForm(data=form_data)
        self.assertTrue(form.is_valid())

    def test_item_form_invalid_name_missing(self):
        form_data = {'description': 'Description without name.'} # Name is required
        form = ItemForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn('name', form.errors)

class ItemViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.item1 = Item.objects.create(name="Item 1", description="First item")
        self.list_url = reverse('items_app:item_list')
        # Note: The example in the prompt for detail_url, update_url, delete_url
        # uses self.item1.pk. If item1 is the only item, its pk might be 1.
        # If other tests create items, this might not be guaranteed.
        # However, for the scope of setUp, self.item1.pk is reliable.
        self.detail_url = reverse('items_app:item_detail', args=[self.item1.pk])
        self.create_url = reverse('items_app:item_create')
        self.update_url = reverse('items_app:item_update', args=[self.item1.pk])
        self.delete_url = reverse('items_app:item_delete', args=[self.item1.pk])

    def test_item_list_view(self):
        # Create another item to test listing multiple items
        Item.objects.create(name="Item 2", description="Second item")
        response = self.client.get(self.list_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'items_app/item_list.html')
        self.assertContains(response, self.item1.name)
        self.assertContains(response, "Item 2") # Check for the second item

    def test_item_detail_view(self):
        response = self.client.get(self.detail_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'items_app/item_detail.html')
        self.assertEqual(response.context['item'], self.item1)
        
        # Test 404 for non-existent item
        non_existent_url = reverse('items_app:item_detail', args=[999]) # Assuming 999 does not exist
        response_404 = self.client.get(non_existent_url)
        self.assertEqual(response_404.status_code, 404)

    def test_item_create_view_get(self):
        response = self.client.get(self.create_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'items_app/item_form.html')
        self.assertIsInstance(response.context['form'], ItemForm)
        self.assertEqual(response.context['form_title'], 'Create New Item')


    def test_item_create_view_post_valid(self):
        item_count_before = Item.objects.count()
        response = self.client.post(self.create_url, {
            'name': 'New Item Created',
            'description': 'Description for new item.'
        })
        self.assertEqual(response.status_code, 302) # Should redirect after successful creation
        self.assertRedirects(response, self.list_url)
        self.assertEqual(Item.objects.count(), item_count_before + 1)
        self.assertTrue(Item.objects.filter(name='New Item Created').exists())

    def test_item_create_view_post_invalid(self):
        item_count_before = Item.objects.count()
        response = self.client.post(self.create_url, {'description': 'Only description'}) # Missing name
        self.assertEqual(response.status_code, 200) # Should re-render form
        self.assertFormError(response, 'form', 'name', 'This field is required.')
        self.assertEqual(Item.objects.count(), item_count_before) # No item should be created

    def test_item_update_view_get(self):
        response = self.client.get(self.update_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'items_app/item_form.html')
        self.assertEqual(response.context['item'], self.item1)
        self.assertIsInstance(response.context['form'], ItemForm)
        self.assertEqual(response.context['form_title'], 'Edit Item')


    def test_item_update_view_post_valid(self):
        updated_name = "Updated Item 1 Name"
        updated_description = "Updated item description."
        response = self.client.post(self.update_url, {
            'name': updated_name,
            'description': updated_description
        })
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, self.list_url)
        self.item1.refresh_from_db()
        self.assertEqual(self.item1.name, updated_name)
        self.assertEqual(self.item1.description, updated_description)

    def test_item_update_view_post_invalid(self):
        original_name = self.item1.name
        response = self.client.post(self.update_url, {
            'name': '', # Invalid: name cannot be empty
            'description': 'Trying to update with invalid name'
        })
        self.assertEqual(response.status_code, 200) # Should re-render form
        self.assertFormError(response, 'form', 'name', 'This field is required.')
        self.item1.refresh_from_db()
        self.assertEqual(self.item1.name, original_name) # Name should not have changed

    def test_item_delete_view_get(self):
        response = self.client.get(self.delete_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'items_app/item_confirm_delete.html')
        self.assertEqual(response.context['item'], self.item1)

    def test_item_delete_view_post(self):
        item_count_before = Item.objects.count()
        response = self.client.post(self.delete_url)
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, self.list_url)
        self.assertEqual(Item.objects.count(), item_count_before - 1)
        self.assertFalse(Item.objects.filter(pk=self.item1.pk).exists())
