
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    # path("", ), # TODO: Display Profile
    path("login/", views.login),
    path("signup/", views.signup),
    path("delete_account/", views.remove_user),
    path("test_token/", views.test_token),
]
