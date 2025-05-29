from django.shortcuts import render, redirect, get_object_or_404
from .models import Item
from .forms import ItemForm

def item_list_view(request):
    items = Item.objects.all()
    context = {'items': items}
    return render(request, 'items_app/item_list.html', context)

def item_detail_view(request, pk):
    item = get_object_or_404(Item, pk=pk)
    context = {'item': item}
    return render(request, 'items_app/item_detail.html', context)

def item_create_view(request):
    if request.method == 'POST':
        form = ItemForm(request.POST)
        if form.is_valid():
            form.save()
            # Assuming 'item_list' is the name of the URL pattern for the item list view
            return redirect('item_list') 
    else:
        form = ItemForm()
    context = {
        'form': form,
        'form_title': 'Create New Item',
        # 'form_action_url_name': 'item_create' # We'll use this in the template with {% url 'item_create' %}
    }
    return render(request, 'items_app/item_form.html', context)

def item_update_view(request, pk):
    item = get_object_or_404(Item, pk=pk)
    if request.method == 'POST':
        form = ItemForm(request.POST, instance=item)
        if form.is_valid():
            form.save()
            return redirect('item_list')
    else:
        form = ItemForm(instance=item)
    context = {
        'form': form,
        'item': item, # Useful for displaying item info or for action URL
        'form_title': 'Edit Item',
        # 'form_action_url_name': 'item_update' # We'll use this in template with {% url 'item_update' item.pk %}
    }
    return render(request, 'items_app/item_form.html', context)

def item_delete_view(request, pk):
    item = get_object_or_404(Item, pk=pk)
    if request.method == 'POST':
        item.delete()
        return redirect('item_list')
    context = {'item': item}
    return render(request, 'items_app/item_confirm_delete.html', context)
