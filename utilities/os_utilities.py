import json
import os.path

from random import shuffle
from typing import Dict, Any


def read_json(json_path: str) -> Dict[str, Any]:
    """
    Reads a JSON file from the specified path and returns its contents as a dictionary.

    This utility function is used across the project to load configuration or annotation data,
    such as the VGG Image Annotator output.

    Args:
        json_path (str): The absolute or relative path to the JSON file to read.

    Returns:
        Dict[str, Any]: The parsed data from the JSON file.
    """
    data = None

    with open(json_path) as f:
        data = json.load(f)

    return data


def make_directory(directory_path: str):
    os.makedirs(directory_path, exist_ok=True)


def get_images(path, basename=False, sort=False, mix=False, coherence=False):
    if coherence:
        if basename:
            images = [file for file in os.listdir(path) if
                      file.endswith(('.jpg', '.png', '.jpeg', '.tiff')) and os.stat(
                          os.path.join(path, file)).st_size != 0]
        else:
            images = [os.path.join(path, file) for file in os.listdir(path) if
                      file.endswith(('.jpg', '.png', '.jpeg', '.tiff')) and os.stat(
                          os.path.join(path, file)).st_size != 0]
    else:
        if basename:
            images = [file for file in os.listdir(path) if
                      file.endswith(('.jpg', '.png', '.jpeg', '.tiff'))]
        else:
            images = [os.path.join(path, file) for file in os.listdir(path) if
                      file.endswith(('.jpg', '.png', '.jpeg', '.tiff'))]
    if mix:
        shuffle(images)

    if sort:
        images = sorted(images)

    return images
