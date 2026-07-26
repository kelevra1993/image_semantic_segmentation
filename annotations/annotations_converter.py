import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Literal
from shapely.geometry import Polygon

from utilities.os_utilities import read_json


class AnnotationsConverter():
    def __init__(self, image_path: str, annotation_path: str, label_dictionary: Dict,
                 method: Literal["naive", "ambiguous"] = "naive") -> None:
        """
        Initializes the converter by loading the image and annotations, and immediately generating the mask.

        Args:
            image_path (str): Path to the source image file.
            annotation_path (str): Path to the VGG Image Annotator JSON file.
            label_dictionary (Dict): A mapping of string labels to integer pixel values for the mask.
            method (Literal["naive", "ambiguous"], optional): The strategy for resolving overlapping 
                polygons. Defaults to "naive".
        """
        self.image_path = image_path
        self.image = cv2.imread(self.image_path)

        self.annotation_path = annotation_path
        self.annotation_data = read_json(json_path=self.annotation_path)
        self.image_annotations = self.convert_image_annotations()

        self.label_dictionary = label_dictionary
        self.neural_network_mask = self.get_mask(method=method)

    def convert_image_annotations(self) -> List[Dict]:
        """
        Extracts and converts the raw VGG Image Annotator (VIA) regions into a standardized format.

        This method acts as the initial parsing step for the JSON annotations. It searches the raw
        JSON data for entries matching the specified image file name, extracts the polygon coordinate 
        points, and associates them with their corresponding class labels (e.g., 'membrane', 'bacteria'). 
        The resulting structured data is then used downstream for creating the segmentation masks.

        Returns:
            List[Dict]: A list of dictionaries, where each dictionary represents a labeled region 
                containing a 'label' string and a 'points' list of (x, y) coordinate tuples.
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
            converted_image_annotations.append({
                "label": annotation_region["region_attributes"]["class"],
                "points": list(zip(shape_attributes["all_points_x"], shape_attributes["all_points_y"]))})

        return converted_image_annotations

    def prioritize_largest_membrane(self) -> List[Dict]:
        """
        Reorders the image annotations to ensure the largest membrane is processed first.

        This method resolves ambiguity in the provided annotations where a large membrane spans 
        approximately the whole image, and contains bacteria regions that themselves have inner 
        membrane regions. By identifying and placing the largest membrane first in the list, it 
        ensures that subsequent bacteria and inner regions are drawn correctly on top of it.

        Returns:
            List[Dict]: A reordered list of annotation dictionaries, with the largest membrane 
                annotation at the beginning (if any membranes are present).
        """
        membrane_areas = []

        found_membrane = False

        for region_information in self.image_annotations:
            if region_information["label"] == "membrane":
                membrane_areas.append(Polygon(region_information["points"]).area)
                found_membrane = True
            else:
                membrane_areas.append(0)

        # Just re-order data while keeping the largest membrane as the first element
        if found_membrane:
            largest_membrane_index = np.argmax(membrane_areas)
            re_ordered_image_annotations = [self.image_annotations[largest_membrane_index]]

            for index, image_annotation in enumerate(self.image_annotations):
                # Ignore largest image annotation since it has already be considered at the beginning
                if index == largest_membrane_index:
                    continue
                re_ordered_image_annotations.append(image_annotation)

            return re_ordered_image_annotations
        else:
            return self.image_annotations

    def get_mask(self, method: Literal["naive", "ambiguous"]) -> np.ndarray:
        """
        Generates a categorical 2D image mask from the parsed polygon annotations.

        This function maps the region annotations to an array of pixel values, acting as the primary
        data preparation step before feeding images into the segmentation neural network. Based on the 
        selected `method`, it handles overlapping polygons differently. The 'ambiguous' method prioritizes
        the largest membrane to be drawn first, so smaller enclosed structures (like bacteria) are drawn 
        on top, preventing them from being overwritten.

        Args:
            method (Literal["naive", "ambiguous"]): The strategy used to handle overlapping annotations. 
                'naive' draws them in the order they appear, whereas 'ambiguous' re-orders them to draw 
                the largest membrane first.

        Returns:
            np.ndarray: A 2D numpy array of shape (height, width) representing the mask.
        """
        # Set up the mask
        image_height, image_width, _ = self.image.shape
        mask = np.zeros((image_height, image_width), dtype=np.uint8)

        # Get the largest membrane and set it at the begining if the method is set to ambiguous, if not
        # just use the labels as they are ordered
        image_annotatations = self.prioritize_largest_membrane() if method == "ambiguous" else self.image_annotations

        for index, image_annotatation in enumerate(image_annotatations):
            # Get label as well as label class value
            # example label=bacteria , label_value=2
            label = image_annotatation["label"]
            label_value = self.label_dictionary[label]

            # Fetch the polygon points [(x1, y1), (x2, y2), (x3, y3)...] and structure it for opencv
            points = image_annotatation["points"]
            polygon_points = np.array(points).reshape((-1, 1, 2))

            # Fill the mask based on label dictionary set by user
            cv2.fillPoly(img=mask, pts=[polygon_points], color=label_value)

        return mask

    def show_mask(self, multiplier: int = 30, window_name="Image Mask"):
        """
        Displays the generated neural network mask in a local GUI window for visual inspection.

        Args:
            multiplier (int, optional): A scaling factor applied to the mask pixel values to enhance 
                visibility on screen. Defaults to 30.
            window_name (str, optional): The title of the OpenCV window. Defaults to "Image Mask".
        """
        cv2.imshow(window_name, self.neural_network_mask * multiplier)
        cv2.waitKey(0)

    def save_mask(self, output_file_path: str, add_interpretable_version: bool = False, multiplier=30):
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
