from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.core.files import File
from django.conf import settings

from keras.models import load_model, Model
from keras.utils import img_to_array, load_img

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import renderer_classes

from .models import XRayImage, XRayPrediction
from .serializers import XRayImageSerializer, XRayPredictionSerializer
from .renderers import JPEGRenderer, PNGRenderer

from pathlib import Path

import numpy as np

# constants
IMG_HEIGHT = 512
IMG_WIDTH = 512

BASE_DIR: Path = settings.BASE_DIR
MODELS_PATH: Path = BASE_DIR.joinpath("models")

MODEL_NAME = "classification_model"
MODEL: Model | None = None  # load_model(MODELS_PATH.joinpath(MODEL_NAME))


@api_view(["GET"])
@renderer_classes([PNGRenderer])
def get_images(request):
    """
    Returns the saved images

    todo: make it return images which it doesn't do now
    """
    images = XRayImage.objects.all()
    x_ray_image_serializer = XRayImageSerializer(images, many=True)

    return Response(x_ray_image_serializer.data, content_type="image/png")


def upload_image(request):
    """
    Saves the given image in the server and returns the image name, image file path and the url on the server
    """
    f = request.FILES["image"]
    fs = FileSystemStorage()
    file_path = fs.save(f.name, f)
    url = fs.url(file_path)

    return f.name, fs.path(file_path), url


@api_view(["POST"])
def save_image(request):
    """
    The request must contain a image file

    """
    image_name, file_path, url = upload_image(request)

    # creating a data to save
    request_data = {"name": image_name, "img_field": File(open(file_path, "r"))}

    x_ray_image_serializer = XRayImageSerializer(data=request_data)

    if x_ray_image_serializer.is_valid():
        x_ray_image_serializer.save()
    return Response(x_ray_image_serializer.data)


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
def run_inference(request):
    """
    todo: integrate predictions with the predictions model
    """

    (image_name, file_path, url) = upload_image(request=request)
    label, remarks, confidence = run_inference_on_image(file_path)
    response_data = {
        "label": label,
        "remarks": remarks,
        "confidence": confidence,
    }
    return Response(response_data, status=status.HTTP_202_ACCEPTED)
