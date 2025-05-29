from django.urls import path
from . import views

app_name = 'items_app'  # Namespacing for URLs

urlpatterns = [
    path('', views.item_list_view, name='item_list'),
    path('<int:pk>/', views.item_detail_view, name='item_detail'),
    path('new/', views.item_create_view, name='item_create'),
    path('<int:pk>/edit/', views.item_update_view, name='item_update'),
    path('<int:pk>/delete/', views.item_delete_view, name='item_delete'),
]
