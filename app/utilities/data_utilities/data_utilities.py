import cv2
import numpy as np


def preprocess(image: np.ndarray, image_size: int, keep_ratio: bool = True) -> np.ndarray:
    """
    Preprocesses an image by resizing it to a specified square size, optionally maintaining its aspect ratio.

    If `keep_ratio` is True, the image is padded with zeros (black) to make it square before resizing,
    preventing distortion. If False, the image is simply stretched or squashed to the target size.

    Args:
        image (np.ndarray): The input image array, typically of shape (Height, Width, Channels).
        image_size (int): The target width and height for the output square image.
        keep_ratio (bool): If True, pads the image to maintain its original aspect ratio before resizing.
        Defaults to True.

    Returns:
        np.ndarray: The preprocessed and resized image of shape (image_size, image_size, Channels).
    """
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
