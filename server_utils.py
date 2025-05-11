from PIL import Image
import numpy as np
# from explain_models.yolo_cam import show_cam_on_image
# from explain_models.yolo_cam import EigenCAM

from explain_models.yolo_cam.utils.image import show_cam_on_image
from explain_models.yolo_cam.eigen_cam import EigenCAM
import io
import base64


def resize_image(image_pil, target_size=(640, 480)):
    """
    Изменяет размер изображения.
    :param image: Изображение в формате numpy array.
    :param target_size: Целевой размер (ширина, высота).
    :return: Изображение с измененным размером.
    """
    resized_image = image_pil.resize(target_size)
    return resized_image  # .flatten().tolist()


def make_classification(model, image) -> list:
    result = model(image)

    return result[0].probs.top1 + 1


def make_explain(model, image):
    model.cpu()
    image = image.copy()
    rgb_img = image.copy()
    image = np.float32(image) / 255
    # target_layers = [model.model.model[-2]]
    target_layers = target_layers = [
        model.model.model[-2],
        model.model.model[-3],
        model.model.model[-4],
    ]
    cam = EigenCAM(model, target_layers, task="cls")
    grayscale_cam = cam(rgb_img)[0, :, :]
    cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=True)
    return cam_image


def decode_image(image):
    image = Image.fromarray(image)
    buffered_image = io.BytesIO()
    image.save(buffered_image, format="JPEG")
    decoded_imaged = base64.b64encode(buffered_image.getvalue()).decode("utf-8")
    return decoded_imaged
