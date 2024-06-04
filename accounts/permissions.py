from rest_framework.permissions import BasePermission


class DoctorPermission(BasePermission):

    def has_permission(self, request, view):
        return request.user.is_authenticated and request.user.user_type == 'Dr'  # and request.user.is_verified == True

    def has_object_permission(self, request, view, obj):
        return obj.owner == request.user


class PatientPermission(BasePermission):

    def has_permission(self, request, view):
        return request.user.is_authenticated and request.user.user_type == 'Pt'

    def has_object_permission(self, request, view, obj):
        return obj.owner == request.user
