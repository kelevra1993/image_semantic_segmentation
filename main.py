import os
from imaplib import Literal

from annotations.annotations_converter import AnnotationsConverter

# Set Up Paths
current_working_directory = os.getcwd()
dataset_directory = os.path.join(current_working_directory, "technical_information", "dataset")

# Get Bacteria Image + it's annotations
bacteria_path = os.path.join(dataset_directory, "bacteria.png")
annotation_path = os.path.join(dataset_directory, "annotations.json")

# Setup Desired Label Dictionary {class_name: class_index_in_mask}
label_dictionary = {"membrane": 0, "bacteria": 1, "unsure": 2}

# Based on some ambiguity in the annotation interpretation, i chose to implement two interpretations.
# Naive interpretation -> Set up regions in the order they were labeled
# Ambiguous interpretation -> Set largest membrane annotation as first labeled region then add the remaining regions
# Other : I might have overthought this, but i couldn't find a better interpretation of what was expected
for interpretation in ["ambiguous", "naive"]:
    annotations = AnnotationsConverter(image_path=bacteria_path,
                                       annotation_path=annotation_path,
                                       label_dictionary=label_dictionary,
                                       method=interpretation)

    # Define where to save the masks
    mask_path = os.path.join(dataset_directory, f"{interpretation}_mask_annotations.png")
    annotations.save_mask(output_file_path=mask_path)

    # Display Mask For Visualisation
    # For viewing purposes we scale the values of the mask by the multiplier
    annotations.show_mask(window_name=f"{interpretation} Mask", multiplier=50)
