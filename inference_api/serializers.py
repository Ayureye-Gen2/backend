from .models import XRayImage, XRayPrediction
from rest_framework import serializers


class XRayPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = XRayPrediction
        fields = "__all__"


class XRayImageSerializer(serializers.ModelSerializer):
    predictions = serializers.PrimaryKeyRelatedField(
        many=True, queryset=XRayPrediction.objects.all()
    )

    class Meta:
        model = XRayImage
        fields = ["name", "img_file", "upload_date", "predictions"]
