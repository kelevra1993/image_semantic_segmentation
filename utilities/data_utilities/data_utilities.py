import cv2
import numpy as np


def preprocess(image, image_size, keep_ratio: bool = True):
    """Todo document function and add function signature type hints"""

    if keep_ratio:
        image_height, image_width, _ = image.shape

        maximum_size = max([image_height, image_width])
        processed_image = np.zeros((maximum_size, maximum_size, 3), dtype=image.dtype)

        start_x = int((maximum_size - image_width) / 2)
        start_y = int((maximum_size - image_height) / 2)

        processed_image[
            start_y: start_y + image_height, start_x: start_x + image_width, :] = image[
            :image_height, :image_width, :]
        processed_image = cv2.resize(processed_image, (image_size, image_size))
    else:
        processed_image = cv2.resize(image, (image_size, image_size))

    return processed_image
