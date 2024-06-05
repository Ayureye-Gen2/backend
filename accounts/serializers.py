from rest_framework import serializers
from .models import UserProfile


class UserSerializer(serializers.ModelSerializer):
    class Meta(object):
        model = UserProfile
        fields = ["id", "username", "password", "email", "user_type", "first_name", "last_name"]

        extra_kwargs = {
            "password": {"write_only": True},
        }

class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        exclude = ("is_verified",)
        extra_kwargs = {
            "password": {"write_only": True},
            "license_photo": {"required": False},
            "hospital_name": {"required": False},
            "user_type": {"required": True},
            "contact_number": {"required": False},
            "country": {"required": True},
            "city": {"required": True},
        }

    def validate(self, data):
        data_dict = dict(data)
        data_keys = data.keys()
        account_type = data_dict.get("user_type")
        if account_type == "Dr" and "license_photo" not in data_keys:
            raise serializers.ValidationError(
                "License photo required to submit form as Doctor."
            )
        if account_type == "Dr" and "hospital_name" not in data_keys:
            raise serializers.ValidationError(
                "Hospital name required to submit form as Doctor."
            )
        if account_type == "Dr" and "contact_number" not in data_keys:
            raise serializers.ValidationError(
                "Doctors need to have a contact number."
            )
        if account_type == "Pt" and "license_photo" in data_keys:
            raise serializers.ValidationError("Patients cannot have a license photo")

        if account_type == "Pt" and "contact_number" not in data_keys:
            raise serializers.ValidationError("Patients must have a contact number")
        
        if account_type == "Gst" and ("license_photo" in data_keys or "contact_number" in data_keys):
            raise serializers.ValidationError("Guests cannot have a contact number or a license photo.")

        return data

    def create(self, validated_data):
        return UserProfile(**validated_data)
