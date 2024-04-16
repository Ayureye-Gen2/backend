from .models import XRayImage, XRayPrediction
from rest_framework import serializers


class XRayImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = XRayImage
        fields = "__all__"


class XRayPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = XRayPrediction
        fields = "__all__"
