from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='main_site-home'),
    path('about/', views.about, name='main_site-about'),
]