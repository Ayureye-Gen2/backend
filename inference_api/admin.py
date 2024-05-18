from django.contrib import admin
from .models import XRayImage, XRayPrediction

admin.site.register(XRayImage)
admin.site.register(XRayPrediction)
