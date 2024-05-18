"""
TODO: implement the following permissions

PatientPermission: Used to determine the actions that can be performed by patients
DoctorPermission: Used to determine the actions that can be performed by doctors

"""
from rest_framework.permissions import BasePermission


class DoctorPermission(BasePermission):

    def has_permission(self, request, view):
        return request.user.is_authenticated and request.user.user_type == 'Dr' and request.user.is_verified == True


class PatientPermission(BasePermission):

    def has_permission(self, request, view):
        return request.user.is_authenticated and request.user.user_type == 'Pt'

