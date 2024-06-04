from .models import XRayImage, XRayPrediction, PredictedXRayImage
from rest_framework import serializers


class XRayPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = XRayPrediction
        fields = "__all__"


class XRayImageSerializer(serializers.ModelSerializer):
    predictions = serializers.PrimaryKeyRelatedField(
        many=True, queryset=XRayPrediction.objects.all(), allow_null=True, required=False
    )

    class Meta:
        model = XRayImage
        fields = ["id", "name", "img_file", "upload_date", "predictions"]

class XRayPredictedImageSerializer(serializers.ModelSerializer):
    predictions = serializers.PrimaryKeyRelatedField(
        many=True, queryset=XRayPrediction.objects.all(), allow_null=True, required=False
    )

    class Meta:
        model = PredictedXRayImage
        fields = ["id", "name", "img_file", "upload_date", "predictions"]
