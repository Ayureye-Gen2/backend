from django.db import models


class XRayImage(models.Model):
    """
    One image can have multiple predictions.
    """

    name = models.CharField(max_length=255)
    img_file = models.ImageField(upload_to="images/")
    upload_date = models.DateTimeField(auto_now_add=True)


class XRayPrediction(models.Model):
    """
    One prediction needs to be associated with only one infection.
    There are no bounds on the co-ordinates
    """

    infection_type = models.CharField(max_length=255)  # using string for now
    bounding_box_coordinates = models.JSONField(
        max_length=1000
    )  # todo: need to check the max length
    image = models.ForeignKey(XRayImage, on_delete=models.DO_NOTHING)
