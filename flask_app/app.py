# Main Flask application logic
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_app.models import Item # Adjusted import

app = Flask(__name__)
# Replace with your actual database URI
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:password@localhost/mydatabase'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # Optional: Suppresses a warning

db = SQLAlchemy(app)

# Create Route
@app.route('/items', methods=['POST'])
def create_item():
    data = request.get_json()
    if not data or not 'name' in data:
        return jsonify({'message': 'Name is required'}), 400
    new_item = Item(name=data['name'], description=data.get('description'))
    db.session.add(new_item)
    db.session.commit()
    return jsonify({'id': new_item.id, 'name': new_item.name, 'description': new_item.description}), 201

# Read All Route
@app.route('/items', methods=['GET'])
def get_items():
    items = Item.query.all()
    return jsonify([{'id': item.id, 'name': item.name, 'description': item.description} for item in items])

# Read One Route
@app.route('/items/<int:item_id>', methods=['GET'])
def get_item(item_id):
    item = Item.query.get(item_id)
    if item:
        return jsonify({'id': item.id, 'name': item.name, 'description': item.description})
    return jsonify({'message': 'Item not found'}), 404

# Update Route
@app.route('/items/<int:item_id>', methods=['PUT'])
def update_item(item_id):
    item = Item.query.get(item_id)
    if not item:
        return jsonify({'message': 'Item not found'}), 404
    data = request.get_json()
    if not data:
        return jsonify({'message': 'No input data provided'}), 400
    item.name = data.get('name', item.name)
    item.description = data.get('description', item.description)
    db.session.commit()
    return jsonify({'id': item.id, 'name': item.name, 'description': item.description})

# Delete Route
@app.route('/items/<int:item_id>', methods=['DELETE'])
def delete_item(item_id):
    item = Item.query.get(item_id)
    if not item:
        return jsonify({'message': 'Item not found'}), 404
    db.session.delete(item)
    db.session.commit()
    return jsonify({'message': 'Item deleted successfully'})

# --- HTML Serving Routes ---

# List all items (HTML)
@app.route('/ui/items', methods=['GET'])
def list_items_html():
    items = Item.query.all()
    return render_template('index.html', items=items)

# Show form to create a new item (HTML)
@app.route('/ui/items/new', methods=['GET'])
def create_item_form():
    return render_template('item_form.html', form_title="Create Item", form_action=url_for('create_item_html'))

# Handle creation of a new item from form (HTML)
@app.route('/ui/items', methods=['POST'])
def create_item_html():
    name = request.form.get('name')
    description = request.form.get('description')
    if name: # Basic validation
        new_item = Item(name=name, description=description)
        db.session.add(new_item)
        db.session.commit()
    return redirect(url_for('list_items_html'))

# Show item details (HTML)
@app.route('/ui/items/<int:item_id>', methods=['GET'])
def get_item_detail(item_id):
    item = Item.query.get_or_404(item_id)
    return render_template('item_detail.html', item=item)

# Show form to edit an existing item (HTML)
@app.route('/ui/items/<int:item_id>/edit', methods=['GET'])
def edit_item_form(item_id):
    item = Item.query.get_or_404(item_id)
    return render_template('item_form.html', form_title="Edit Item", item=item, form_action=url_for('update_item_html', item_id=item.id))

# Handle update of an existing item from form (HTML)
@app.route('/ui/items/<int:item_id>/update', methods=['POST'])
def update_item_html(item_id):
    item = Item.query.get_or_404(item_id)
    item.name = request.form.get('name', item.name)
    item.description = request.form.get('description', item.description)
    db.session.commit()
    return redirect(url_for('list_items_html'))

# Handle deletion of an item (HTML)
@app.route('/ui/items/<int:item_id>/delete', methods=['POST']) # Using POST for delete as forms typically submit POST
def delete_item_html(item_id):
    item = Item.query.get_or_404(item_id)
    db.session.delete(item)
    db.session.commit()
    return redirect(url_for('list_items_html'))

if __name__ == '__main__':
    app.run(debug=True)
