import json
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
