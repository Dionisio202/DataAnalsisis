from django.urls import path
from .views import upload_measurements

urlpatterns = [
    path('upload/', upload_measurements, name='upload_measurements'),
]
