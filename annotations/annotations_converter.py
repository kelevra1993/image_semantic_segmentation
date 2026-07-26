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

        self.image_path = image_path
        self.image = cv2.imread(self.image_path)

        self.annotation_path = annotation_path
        self.annotation_data = read_json(json_path=self.annotation_path)
        self.image_annotations = self.convert_image_annotations()

        self.label_dictionary = label_dictionary
        self.neural_network_mask = self.get_mask(method=method)

    def convert_image_annotations(self):
        annotation_regions = None

        # TODO : Check with Spore.bio -> I assume that the first key is the project name
        #  , multiple different labellers or multiple different resolutions of an image given the key name but no sure.
        #  we will iterate through it and check that we have the same file name before adding labelled regions
        for key, labelling_information in self.annotation_data.items():
            if labelling_information["filename"] == os.path.basename(self.image_path):
                # Grab the image annotations
                annotation_regions = labelling_information["regions"]

        # TODO Inform that we could not get the labels
        # todo to be re-reviewed
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

    def prioritize_largest_membrane(self):
        # Trying to deal with ambiguity of the the annotation that was provided
        # There is a membrane that spans approximately the whole image
        # Inside of it there are bacteria regions that also have inner regions
        # that are membranes followed by other bacteria regions
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

    def get_mask(self, method):
        # todo definition of method for ambiguous

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

    def show_mask(self, multiplier=50, window_name="Image Mask"):
        """Displays the generated neural network mask in a window."""
        cv2.imshow(window_name, self.neural_network_mask * multiplier)
        cv2.waitKey(0)

    def save_mask(self, output_file_path: str):
        """Saves the generated neural network mask to the specified file path."""
        cv2.imwrite(filename=output_file_path, img=self.neural_network_mask, params=[cv2.IMWRITE_PNG_COMPRESSION, 0])
