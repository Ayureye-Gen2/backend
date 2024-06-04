from PIL import Image
from ultralytics import YOLO

from pathlib import Path
import numpy as np
import typing

from django.conf import settings

import cv2

if __name__ != "__main__":
    MODELS_DIR = Path(settings.STATICFILES_DIRS[0]).joinpath("models")

DEFAULT_MODEL_NAME = "yolov9c-trained-epoch30.pt"
DEFAULT_IMAGE_SIZE = 640
DEFAULT_CONFIDENCES_THRESHOLD = 0.25


model: YOLO | None = None
class_names: dict[int, str] | None = None


def initialize_model(model_name: str = DEFAULT_MODEL_NAME):
    global model, class_names
    model = YOLO(MODELS_DIR.joinpath(model_name), task="detect")
    class_names = model.names


def get_boxes_and_filled_image(
    imgs: np.ndarray | str,
    image_size=DEFAULT_IMAGE_SIZE,
    conf_threshold=DEFAULT_CONFIDENCES_THRESHOLD,
) -> tuple[bool, list[tuple[typing.Any, np.ndarray]]]:
    """
    imgs: a image or a batch of images

    Return
    -------
    Bool to indicate whether the prediction ran successfully. (False if there was any error)
    The associated bounding boxes(if any) and the plotted image (bgr)
    """

    if model is not None:
        results = model.predict(
            imgs, task="detect", imgsz=image_size, conf=conf_threshold
        )

        return (True, [(result.boxes, result.plot()) for result in results])
    else:
        return (False, [])


def display_image_and_wait(
    imgs: list[np.ndarray], window_name: str = "Plotted Image [ Press q to quit]"
):
    while True:
        for i, img in enumerate(imgs):
            _window_name = f" [{i}] {window_name}"

            cv2.imshow(_window_name, mat=img)

            k = cv2.waitKey(1)

            if k == ord("q"):
                cv2.destroyAllWindows()
                return

            if k == ord("n"):
                cv2.destroyWindow(_window_name)
                imgs.pop(i)
                break


if __name__ == "__main__":  # for testing
    MODELS_DIR = Path(__file__).parents[1].joinpath("static", "models")
    IMAGES_DIR = Path(__file__).parents[1].joinpath("static", "images")
    TEST_IMAGES_NAMES = [
        "05ba77b1cf87cd4597d988581f269b44.png",
        "005d70155f949c7785671800f2c8e1ca.png",
        "0021df30f3fddef551eb3df4354b1d06.png",
        "000434271f63a053c4128a0ba6352c7f.png",
        "bus.png",
    ]

    test_images = [IMAGES_DIR.joinpath(image_name) for image_name in TEST_IMAGES_NAMES]

    initialize_model()

    _image_size = 640
    _display_image = False

    print(class_names)

    success, predictions = get_boxes_and_filled_image(
        test_images,
        image_size=_image_size,
    )

    if success:
        plotted_images = []
        for i, (bbox, plotted_image) in enumerate(predictions):
            image_index = i
            cls = bbox.cls.numpy()
            conf = bbox.conf.numpy()

            if len(cls):
                xyxy = bbox.xyxy.numpy()[0]
            else:
                xyxy = []

            print(f"image index: {image_index}")
            print(f"class: {cls}")
            print(f"class_name: {class_names[cls[0]] if len(cls) else 'None'}")
            print(f"confidence: {conf}")
            print(f"xyxy: {xyxy}")

            plotted_images.append(cv2.resize(plotted_image, (_image_size, _image_size)))

        if _display_image:
            display_image_and_wait(plotted_images)
    else:
        print(
            "There was an error in running prediction, either the model wasn't initialized or the passed in parameters are invalid."
        )
