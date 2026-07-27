import os
import cv2
import numpy as np
from typing import Dict, List, Literal
from shapely.geometry import Polygon

from utilities.os_utilities import read_json


class AnnotationsConverter:
    """
    A VGG Image Annotation converter in order to get masks for training a neural network.
    """

    def __init__(self, image_path: str, annotation_path: str, label_dictionary: Dict) -> None:
        """
        Initializes the converter by loading the image and annotations, and immediately generating the mask.

        Args:
            image_path (str): Path to the source image file.
            annotation_path (str): Path to the VGG Image Annotator JSON file.
            label_dictionary (Dict): A mapping of string labels to integer pixel values for the mask.
        """
        self.image_path = image_path
        self.image = cv2.imread(self.image_path)

        self.annotation_path = annotation_path
        self.annotation_data = read_json(json_path=self.annotation_path)
        self.image_annotations = self.convert_image_annotations()

        self.label_dictionary = label_dictionary
        self.neural_network_mask = self.get_mask()

    def convert_image_annotations(self) -> List[Dict]:
        """
        Extracts and converts the raw VGG Image Annotator (VIA) regions into a standardized format.

        This method acts as the initial parsing step for the JSON annotations. It searches the raw
        JSON data for entries matching the specified image file name, extracts the polygon coordinate 
        points, and associates them with their corresponding class labels (e.g., 'membrane', 'bacteria'). 
        The resulting structured data is sorted by polygon area in descending order and is then used
        downstream for creating the segmentation masks.

        Returns:
            List[Dict]: A list of dictionaries, where each dictionary represents a labeled region 
                containing a 'label' string, a 'points' list of (x, y) coordinate tuples, and its 'area'.
        """
        annotation_regions = None

        for key, labelling_information in self.annotation_data.items():
            if labelling_information["filename"] == os.path.basename(self.image_path):
                # Grab the image annotations
                annotation_regions = labelling_information["regions"]

        if not annotation_regions:
            print(f"Unfortunately we could not find any labelled regions for {self.image_path}")
            exit()

        converted_image_annotations = []
        for index, annotation_region in enumerate(annotation_regions):
            shape_attributes = annotation_region["shape_attributes"]
            points = list(zip(shape_attributes["all_points_x"], shape_attributes["all_points_y"]))

            # Compute the area of each polygon since this will determine how they are processed
            converted_image_annotations.append({
                "label": annotation_region["region_attributes"]["class"],
                "points": points,
                "area": Polygon(points).area})

        # Order polygons by their area
        converted_image_annotations = sorted(converted_image_annotations,
                                             key=lambda x: x["area"],
                                             reverse=True)

        return converted_image_annotations

    def get_mask(self) -> np.ndarray:
        """
        Generates a categorical 2D image mask from the parsed polygon annotations.

        This function maps the region annotations to an array of pixel values, acting as the primary
        data preparation step before feeding images into the segmentation neural network. Because the 
        annotations are ordered by area descending, larger structures (like membranes) are drawn first, 
        so smaller enclosed structures (like bacteria) are drawn on top, preventing them from being 
        overwritten.

        Returns:
            np.ndarray: A 2D numpy array of shape (height, width) representing the mask.
        """

        # Set up the mask
        image_height, image_width, _ = self.image.shape
        mask = np.zeros((image_height, image_width), dtype=np.uint8)

        for index, image_annotation in enumerate(self.image_annotations):
            # Get label as well as label class value
            # example label=bacteria , label_value=2
            label = image_annotation["label"]
            label_value = self.label_dictionary[label]

            # Fetch the polygon points [(x1, y1), (x2, y2), (x3, y3)...] and structure it for opencv
            points = image_annotation["points"]
            polygon_points = np.array(points).reshape((-1, 1, 2))

            # Fill the mask based on label dictionary set by user
            cv2.fillPoly(img=mask, pts=[polygon_points], color=label_value)

        return mask

    def show_mask(self, multiplier: int = 30, window_name: str = "Image Mask") -> None:
        """
        Displays the generated neural network mask in a local GUI window for visual inspection.

        Args:
            multiplier (int, optional): A scaling factor applied to the mask pixel values to enhance 
                visibility on screen. Defaults to 30.
            window_name (str, optional): The title of the OpenCV window. Defaults to "Image Mask".
        """
        cv2.imshow(window_name, self.neural_network_mask * multiplier)
        cv2.waitKey(0)

    def save_mask(self, output_file_path: str, add_interpretable_version: bool = False, multiplier: int = 30) -> None:
        """
        Saves the generated neural network mask to disk in PNG format for downstream model training and interpretation.

        Args:
            output_file_path (str): The absolute or relative file path where the model-ready 
                mask should be saved.
            add_interpretable_version (bool, optional): If True, a second scaled version of the mask 
                is saved alongside the original to facilitate human review. Defaults to False.
            multiplier (int, optional): The scaling factor applied to pixel values for the 
                human-interpretable version. Defaults to 30.
        """
        cv2.imwrite(filename=output_file_path, img=self.neural_network_mask, params=[cv2.IMWRITE_PNG_COMPRESSION, 0])

        if add_interpretable_version:
            # human interpretable_version
            output_file_name = f"human_interpretable_{os.path.basename(output_file_path)}"
            human_interpretable_mask = self.neural_network_mask * multiplier
            human_interpretable_output_file_path = os.path.join(os.path.dirname(output_file_path), output_file_name)
            cv2.imwrite(filename=human_interpretable_output_file_path,
                        img=human_interpretable_mask, params=[cv2.IMWRITE_PNG_COMPRESSION, 0])
