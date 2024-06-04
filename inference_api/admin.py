from django.contrib import admin
from .models import XRayImage, XRayPrediction, PredictedXRayImage

admin.site.register(XRayImage)
admin.site.register(XRayPrediction)
admin.site.register(PredictedXRayImage)
