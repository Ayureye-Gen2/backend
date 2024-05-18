from pathlib import Path

import numpy as np

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.files import File
from django.core.files.storage import FileSystemStorage
from django.db.models import OuterRef, Subquery
from django.shortcuts import render

from keras.models import Model, load_model
from keras.utils import img_to_array, load_img

from rest_framework import status
from rest_framework.authentication import (SessionAuthentication,
                                           TokenAuthentication)
from rest_framework.decorators import (api_view, authentication_classes,
                                       permission_classes)
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.utils.serializer_helpers import ReturnDict, ReturnList

from .models import XRayImage, XRayPrediction
from .serializers import XRayImageSerializer, XRayPredictionSerializer

from accounts.permissions import PatientPermission, DoctorPermission
from accounts.decorators import authenticated_user

# constants
IMG_HEIGHT = 512
IMG_WIDTH = 512

BASE_DIR: Path = settings.BASE_DIR
MODELS_PATH: Path = BASE_DIR.joinpath("models")

MODEL_NAME = "classification_model"
MODEL: Model | None = None  # load_model(MODELS_PATH.joinpath(MODEL_NAME))


@api_view(["GET"])
@authentication_classes([SessionAuthentication, TokenAuthentication])
@permission_classes([DoctorPermission])
def get_images(request):
    """
    Returns the saved images

    Only doctors are allowed to view the saved images
    """
    images = XRayImage.objects.all()
    x_ray_image_serializer = XRayImageSerializer(images, many=True)

    return Response(x_ray_image_serializer.data)


@api_view(["GET"])
@authentication_classes([SessionAuthentication, TokenAuthentication])
@permission_classes([DoctorPermission])
def get_predictions_and_images(request):
    """
    Returns the saved predictions along with the associated images.
    Only doctors are allowed to view the predictions for every images

    todo: make it only accessible by doctors
    """
    images = XRayPrediction.objects.order_by("image")
    x_ray_prediction_serializer = XRayPredictionSerializer(images, many=True)

    return Response(x_ray_prediction_serializer.data)


def upload_image(img_file: File) -> None | tuple[int, str, str, str, str]:
    """
    Saves the given image in the server.

    Returns
    -------
    tuple of (image id, image name, image file path, image url, upload date) if the upload succeded \n
    None if upload failed
    """

    fs = FileSystemStorage(  # using this folder as the images are stored inside /media/images
        location=settings.MEDIA_ROOT.joinpath("images"),
        base_url=settings.MEDIA_URL + "images/",
    )

    # creating a data to save to database
    request_data = {"name": img_file.name, "img_file": img_file}

    x_ray_image_serializer = XRayImageSerializer(data=request_data)

    if x_ray_image_serializer.is_valid(raise_exception=True):
        x_ray_image_serializer.save()

        img_id = x_ray_image_serializer.data["id"]
        file_path = x_ray_image_serializer.data[
            "name"
        ]  # because of relative paths and FileSystemStorage's basepath, only the image identifier is required
        upload_date = x_ray_image_serializer.data["upload_date"]

        url = fs.url(file_path)

        return img_id, img_file.name, fs.path(file_path), url, upload_date
    else:
        return None


@api_view(["POST"])
@authentication_classes([SessionAuthentication, TokenAuthentication])
@permission_classes([DoctorPermission | PatientPermission]) # both patient and doctor and save images
def save_image(request):
    """
    The request must contain a image file

    Both the doctors and patients are allowed to save a image
    """
    upload_return = upload_image(request.FILES["image"])

    if upload_return is None:
        return Response(
            data={"msg": "Upload Failed"}, status=status.HTTP_400_BAD_REQUEST
        )  # TODO: properly select satus code

    return Response(
        {
            "id": upload_return[0],
            "name": upload_return[1],
            "img_file": upload_return[3],
            "upload_date": upload_return[4],
        }
    )


def predict(model: Model, img: np.ndarray):
    """
    Run inference on the image and return the sanitized predictions

    one possibility might be.
    {
        {
            "infection": "infection1"
            "co-ordinates": ["(x1, y1)", "(x2, y2)", ...]
        },
        {
            "infection": "infection2"
            "co-ordinates": ["(x3, y3)", "(x4, y4)", ...]
        }

    }

    todo: implement this

    """
    predictions = model.predict(img)

    # sanitize predictions

    return "Nothing", "Nothing", 0


def run_inference_on_image(file_path, image_height=IMG_HEIGHT, image_width=IMG_WIDTH):

    original = load_img(file_path, target_size=(image_width, image_height))
    numpy_image = img_to_array(original)

    if MODEL is not None:
        label, remarks, confidence = predict(MODEL, numpy_image)

        # save the predictions corresponding to the image

    else:
        label, remarks, confidence = (
            "None",
            "Couldn't find the model to run inference",
            0,
        )

    return label, remarks, confidence


@api_view(["POST"])
@authentication_classes([SessionAuthentication, TokenAuthentication])
@permission_classes([IsAuthenticated])
def run_inference(request):
    """
    todo: integrate predictions with the predictions model

    Any logged in users can run inference
    """

    (image_id, image_name, file_path, url, _) = upload_image(request.FILES["image"])
    label, remarks, confidence = run_inference_on_image(file_path)
    response_data = {
        "label": label,
        "remarks": remarks,
        "confidence": confidence,
    }
    return Response(response_data, status=status.HTTP_202_ACCEPTED)
