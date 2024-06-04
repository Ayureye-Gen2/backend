from django.db import models
from accounts.models import UserProfile


class XRayImage(models.Model):
    """
    Storing XRayImage only, there need not be an associated doctor

    One image can have multiple predictions.


    """

    name = models.CharField(max_length=255)
    img_file = models.ImageField(upload_to="images/")
    upload_date = models.DateTimeField(auto_now_add=True)
    patient = models.ForeignKey(
        UserProfile,
        null=True,  # this must be removed in the future
        related_name="patient_xray_image",
        on_delete=models.CASCADE,
        limit_choices_to={
            "user_type": "Pt"
        },  # only allow users with user_type Patient to be stored here
    )  # remove the associated image if user is removed


class PredictedXRayImage(models.Model):
    """

    Storing Predicted XRayImage, there need to be an associated doctor

    All the associated predictions are applied

    """

    name = models.CharField(max_length=255)
    img_file = models.ImageField(upload_to="images/")
    upload_date = models.DateTimeField(auto_now_add=True)
    patient = models.ForeignKey(
        UserProfile,
        null=True,  # this must be removed in the future
        related_name="patient_predicted_xray_image",
        on_delete=models.CASCADE,
        limit_choices_to={
            "user_type": "Pt"
        },  # only allow users with user_type Patient to be stored here
    )  # remove the associated image if user is removed
    diagnosing_doctor = models.ForeignKey(
        UserProfile,
        null=True,
        related_name="doctor_predicted_xray_image",
        on_delete=models.DO_NOTHING,
        limit_choices_to={
            "user_type": "Dr"
        },  # only allow users with user_type Doctor to be stored here
    )  # todo: decide if image not having a defined doctor should be kept


class XRayPrediction(models.Model):
    """
    One prediction needs to be associated with only one infection.
    There are no bounds on the co-ordinates
    """

    DETECTION_PREDICTION = "DET"
    SEGMENTATION_PREDICTION = "SEG"

    PREDITION_CHOICES = {
        DETECTION_PREDICTION: "Detection",
        SEGMENTATION_PREDICTION: "Segmentation",
    }

    prediction_type = models.CharField(
        max_length=3, choices=PREDITION_CHOICES, default=SEGMENTATION_PREDICTION
    )
    infection_type = models.CharField(max_length=255)  # using string for now
    confidence = models.FloatField(default=0.0)
    bounding_box_coordinates = models.JSONField(
        max_length=1000
    )  # todo: need to check the max length
    original_image = models.ForeignKey(
        XRayImage, related_name="original_image_to_predictions", on_delete=models.DO_NOTHING
    )
    predicted_image = models.ForeignKey(
        PredictedXRayImage, related_name="predicted_image_to_predictions", on_delete=models.DO_NOTHING, null=True
    )
