from django.db import models
from accounts.models import UserProfile


class XRayImage(models.Model):
    """
    One image can have multiple predictions.
    """

    name = models.CharField(max_length=255)
    img_file = models.ImageField(upload_to="images/")
    upload_date = models.DateTimeField(auto_now_add=True)
    patient = models.ForeignKey(
        UserProfile,
        null=True, # this must be removed in the future
        related_name="patient_xray_image",
        on_delete=models.CASCADE,
        limit_choices_to={
            "user_type": "Pt"
        },  # only allow users with user_type Patient to be stored here
    )  # remove the associated image if user is removed
    diagnosing_doctor = models.ForeignKey(
        UserProfile,
        null=True,
        related_name="doctor_xray_image",
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

    infection_type = models.CharField(max_length=255)  # using string for now
    bounding_box_coordinates = models.JSONField(
        max_length=1000
    )  # todo: need to check the max length
    image = models.ForeignKey(
        XRayImage, related_name="predictions", on_delete=models.DO_NOTHING
    )
