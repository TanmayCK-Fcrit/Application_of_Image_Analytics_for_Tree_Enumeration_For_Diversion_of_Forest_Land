from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('process/', views.process_image_view, name='process_image'),
    path('download_report/', views.download_pdf, name='download_pdf'),
]
