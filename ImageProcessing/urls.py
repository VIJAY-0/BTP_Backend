
from django.urls import path  # Ensure path is imported from django.urls
from ImageProcessing import views

urlpatterns = [
    path('process_image/', views.process_image, name='process_image'),
]
