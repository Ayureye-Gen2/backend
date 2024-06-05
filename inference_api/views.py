from pathlib import Path

import numpy as np

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.files import File
from django.core.files.storage import FileSystemStorage
from django.db.models import OuterRef, Subquery
from django.shortcuts import render

from rest_framework import status
from rest_framework.authentication import SessionAuthentication, TokenAuthentication
from rest_framework.decorators import (
    api_view,
    authentication_classes,
    permission_classes,
)
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.utils.serializer_helpers import ReturnDict, ReturnList

from accounts.serializers import UserSerializer
from .models import XRayImage, XRayPrediction, PredictedXRayImage, UserProfile
from .serializers import (
    XRayImageSerializer,
    XRayPredictionSerializer,
    XRayPredictedImageSerializer,
)

from accounts.permissions import PatientPermission, DoctorPermission
from accounts.decorators import authenticated_user

from enum import Enum
import cv2

from .detect import initialize_model, get_boxes_and_filled_image
import os

initialize_model()
from .detect import class_names


class ModelType(Enum):
    NONE = 0
    DETECTION = 1
    SEGMENTATION = 2


IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.2


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
def get_patients(request):
    """
    All of the patients
    """
    patients = UserProfile.objects.filter(user_type="Pt")

    user_serializer = UserSerializer(patients, many=True)

    return Response(user_serializer.data)


@api_view(["GET"])
@authentication_classes([SessionAuthentication, TokenAuthentication])
@permission_classes([DoctorPermission])
def get_doctor_specific_patients(request):
    """
    Patients for a particular doctor
    """
    doctor = request.user
    patients = UserProfile.objects.filter(
        id__in=Subquery(
            PredictedXRayImage.objects.filter(diagnosing_doctor=doctor.pk).values(
                "patient"
            )
        )
    )

    user_serializer = UserSerializer(patients, many=True)

    return Response(user_serializer.data)


@api_view(["GET"])
@authentication_classes([SessionAuthentication, TokenAuthentication])
@permission_classes([DoctorPermission | PatientPermission])
def get_images_for_a_patient(request):
    """
    Images of a patient.

    If doctor is requesting this, a patient id must be provided
    """
    if request.user.user_type == "Pt":
        patient = request.user
        patient_pk = patient.pk
    else:
        patient_pk = request.query_params["patient_id"]
    
    images = XRayImage.objects.filter(patient=patient_pk)

    x_ray_image_serializer = XRayImageSerializer(images, many=True)

    return Response(x_ray_image_serializer.data)

@api_view(["POST", "GET"])
@authentication_classes([SessionAuthentication, TokenAuthentication])
@permission_classes([DoctorPermission | PatientPermission])
def get_predictions_for_a_patient(request):
    """
    Predictions of a patient. (NOt Required)

    If doctor is requesting this, a patient id must be provided
    """
    if request.user.user_type == "Pt":
        patient = request.user
        patient_pk = patient.pk
    else:
        if request.method == 'GET':
            patient_pk = request.query_params["patient_id"]
        elif request.method == 'POST':
            patient_pk = request.data["patient_id"]
    
    images = PredictedXRayImage.objects.filter(patient=patient_pk)

    x_ray_image_serializer = XRayPredictedImageSerializer(images, many=True)

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
    images = XRayPrediction.objects.order_by("original_image")
    x_ray_prediction_serializer = XRayPredictionSerializer(images, many=True)

    return_data = x_ray_prediction_serializer.data

    for data in return_data:
        data["original_image_file"] = XRayImage.objects.filter(pk=data["original_image"]).values("img_file").get()["img_file"]
        data["predicted_image_file"] = PredictedXRayImage.objects.filter(pk=data["predicted_image"]).values("img_file").get()["img_file"]

    return Response(return_data)


def upload_image(
    img_file: File, patient_id: int, type: str = "image", doctor_id: int | None = None
) -> None | tuple[int, str, str, str, str]:
    """
    Saves the given image in the server.

    For saving image, the actual file is required
    for saving prediction, the uri is required

    Returns
    -------
    tuple of (image id, image name, image file path, image url, upload date) if the upload succeded \n
    None if upload failed
    """

    fs = FileSystemStorage(  # using this folder as the images are stored inside /media/images
        location=settings.MEDIA_ROOT.joinpath("images"),
        base_url=settings.MEDIA_URL + "images/",
    )

    if type == "image":
        request_data = {"name": img_file.name, "img_file": img_file, "patient": patient_id}
        x_ray_image_serializer = XRayImageSerializer(data=request_data)
    elif type == "prediction":
        request_data = {"name": img_file.name, "img_file": img_file, "patient": patient_id, "diagnosing_doctor": doctor_id}
        x_ray_image_serializer = XRayPredictedImageSerializer(data=request_data)
    else:
        raise NotImplementedError  # for debugging

    data_valid = x_ray_image_serializer.is_valid()

    if data_valid:
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


def upload_prediction(
    prediction: dict,
    original_image_id: int,
    image_name: str,
    predicted_image: np.ndarray,
    patient_id: int,
    doctor_id: int
) -> tuple[bool, str]:
    filename = f"predicted_{image_name}"
    success = cv2.imwrite(filename, predicted_image)

    if success:
        with open(filename, "rb") as img_file:
            img_id, file_name, path, url, upload_date = upload_image(
                File(img_file),patient_id, "prediction", doctor_id
            )

        if os.path.exists(
            filename
        ):  # since upload_image creates new file, deleting the old one
            os.remove(filename)

        for i, cls in enumerate(prediction["infection_types"]):

            request_data = {
                "prediction_type": prediction["prediction_type"],
                "infection_type": cls,
                "confidence": prediction["confidence"][i],
                "bounding_box_coordinates": prediction["bounding_box_coordinates"][i],
                "original_image": original_image_id,
                "predicted_image": img_id,
            }

            x_ray_prediction_serializer = XRayPredictionSerializer(data=request_data)
            if x_ray_prediction_serializer.is_valid(raise_exception=True):
                x_ray_prediction_serializer.save()

        return True, url

    return False, ""


@api_view(["POST"])
@authentication_classes([SessionAuthentication, TokenAuthentication])
@permission_classes(
    [DoctorPermission | PatientPermission]
)  # both patient and doctor and save images
def save_image(request):
    """
    The request must contain a image file as well a the primary key of the associated patient

    Both the doctors and patients are allowed to save a image
    """
    upload_return = upload_image(request.FILES["image"], request.data["patient_id"])

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


def run_inference_on_image(
    file_path, model_type: ModelType = ModelType.DETECTION
) -> tuple[bool, dict | list[tuple[bool, dict, np.ndarray]]]:
    """
    Only detection case is implemented for now

    Returns
    --------
    The bool is the result of whole inference, it is true if inference ran successfully false otherwise. \n
    The dictionary contains the error message if inference couldn't be run. \n

    If inference ran successfuly, the list contains a tuples whose first value indicates where there was any detection and
    the dictionary has the values of the prediction if there was any.\n
    The plotted image is returned with the annotations if there was any predictions else, the same image is returned

    """

    if model_type == ModelType.DETECTION:
        success, predictions = get_boxes_and_filled_image(
            file_path, IMAGE_SIZE, CONFIDENCE_THRESHOLD
        )

        if success:

            data: list[tuple[bool, dict]] = []
            for bbox, plotted_image in predictions:
                cls = bbox.cls.detach().cpu().numpy()
                cls_names = [class_names[int(i)] for i in cls]
                confs = bbox.conf.detach().cpu().numpy()

                xyxys = bbox.xyxy.detach().cpu().numpy()

                plotted_image = cv2.resize(plotted_image, (IMAGE_SIZE, IMAGE_SIZE))

                if not len(cls):
                    data.append((0, {}))
                else:
                    data.append(
                        (
                            True,
                            {
                                "prediction_type": "DET",
                                "infection_types": cls_names,
                                "confidence": confs,
                                "bounding_box_coordinates": [
                                    {  # since this is denormalized values, we can cast it as int
                                        "x_min": int(xyxy[0]),
                                        "y_min": int(xyxy[1]),
                                        "x_max": int(xyxy[2]),
                                        "y_max": int(xyxy[3]),
                                    }
                                    for xyxy in xyxys
                                ],
                            },
                            plotted_image,
                        )
                    )

            return (True, data)

        return (False, {"msg": "Failed to run inference."})

    return (False, {"msg": "Only detection case is implemented."})


@api_view(["POST"])
@authentication_classes([SessionAuthentication, TokenAuthentication])
@permission_classes([IsAuthenticated])
def run_inference(request):
    """
    Any logged in users can run inference

    This version is for running inference on a new image given as a FILE
    """

    patient_id = None
    doctor_id = None
    if request.user.user_type == "Pt":
        patient_id = request.user.pk
    
    if request.user.user_type == "Dr":
        doctor_id = request.user.pk

    if patient_id is None:
        patient_id = request.data.get("patient_id", None)
    
    if doctor_id is None:
        doctor_id = request.data.get("doctor_id", None)

    if patient_id is None or doctor_id is None:
        return Response({"detected": False, "msg": "Required parameters 'patient_id' and 'doctor_id'"})

    (image_id, image_name, file_path, url, _) = upload_image(request.FILES["image"], patient_id=patient_id)
    success, data = run_inference_on_image(file_path.replace(" ", "_")) # the spaces are replaces by _

    if not success:
        return Response(data, status=status.HTTP_400_BAD_REQUEST)

    if not data[0] or not data[0][0]:  # there is only one detection
        return Response({"detected": False, "msg": "No Detections", "img_file": url})

    if success:
        pred, image = data[0][1], data[0][2]

        upload_success, url = upload_prediction(pred, image_id, image_name, image, patient_id, doctor_id)

        detections = []

        if not isinstance(data, list):
            return Response({"detected": False, "msg": "Got invalid predictions"})

        for cls_available, prediction_data, _ in data:
            if cls_available:
                detections.append(prediction_data)

        if not upload_success:
            return Response({"detected": True, "msg": "Couldn't Upload Prediction", "detections": detections})

        return Response({"detected": True, "detections": detections, "img_file": url})


@api_view(["POST"])
@authentication_classes([SessionAuthentication, TokenAuthentication])
@permission_classes([DoctorPermission])
def run_inference_on_already_uploaded_image(request):
    """

   Only doctors can perform this

   This version is for running inference on already uploaded image given a primary key for the uploaded image
   or the uri
    """

    patient_id = None
    doctor_id = None
    if request.user.user_type == "Pt":
        patient_id = request.user.pk
    
    if request.user.user_type == "Dr":
        doctor_id = request.user.pk

    if patient_id is None:
        patient_id = request.data.get("patient_id", None)
    
    if doctor_id is None:
        doctor_id = request.data.get("doctor_id", None)

    if patient_id is None or doctor_id is None:
        return Response({"detected": False, "msg": "Required parameters 'patient_id' and 'doctor_id'"}, status=status.HTTP_400_BAD_REQUEST)

    original_image_id = None

    original_image_id = request.data.get("original_image_id", None)

    if original_image_id is None:
        return Response({"detected": False, "msg": "Required parameters 'original_image_id' ."}, status=status.HTTP_400_BAD_REQUEST)

    query_ret = XRayImage.objects.filter(pk=original_image_id).values("id", "name", "img_file").get() # there should be only one
    image_id, image_name, original_image_url = query_ret["id"], query_ret["name"], query_ret["img_file"]

    fs = FileSystemStorage(  # using this folder as the images are stored inside /media/images
        location=settings.MEDIA_ROOT.joinpath("images"),
        base_url=settings.MEDIA_URL + "images/",
    )

    file_path = fs.path(image_name)
    success, data = run_inference_on_image(file_path.replace(" ", "_")) # the spaces are replaces by _

    if not success:
        return Response({"detected": False, "detections": data}, status=status.HTTP_400_BAD_REQUEST)

    if not data[0] or not data[0][0]:  # there is only one detection
        return Response({"detected": False, "msg": "No Detections", "img_file": url})

    if success:
        pred, image = data[0][1], data[0][2]

        upload_success, url = upload_prediction(pred, image_id, image_name, image, patient_id, doctor_id)

        detections = []

        if not isinstance(data, list):
            return Response({"detected": False, "msg": "Got invalid predictions"})

        for cls_available, prediction_data, _ in data:
            if cls_available:
                detections.append(prediction_data)

        if not upload_success:
            return Response({"detected": True, "msg": "Couldn't Upload Prediction", "detections": detections})

        return Response({"detected": True, "detections": detections, "img_file": url})
