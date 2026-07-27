import os

from annotations.annotations_converter import AnnotationsConverter

# Set Up Paths
current_working_directory = os.getcwd()
dataset_directory = os.path.join(current_working_directory, "technical_information", "dataset")

# Get Bacteria Image + it's annotations
bacteria_path = os.path.join(dataset_directory, "bacteria.png")
annotation_path = os.path.join(dataset_directory, "annotations.json")

# Setup Desired Label Dictionary {class_name: class_index_in_mask}
label_dictionary = {"background": 0, "membrane": 1, "bacteria": 2, "unsure": 3}

# Convert annotations to masks
# First order polygons by area, which should yield order of priority.
annotations = AnnotationsConverter(image_path=bacteria_path,
                                   annotation_path=annotation_path,
                                   label_dictionary=label_dictionary)

# Define where to save the masks
mask_path = os.path.join(dataset_directory, "mask_annotations.png")

# Save the mask (by default just one channel with pixel intensities as classes)
# We also save an image mask that is easily interpretable by a human
annotations.save_mask(output_file_path=mask_path, add_interpretable_version=True)

# Display Mask For Visualisation
# For viewing purposes we scale the values of the mask by the multiplier
annotations.show_mask(window_name=f"Mask", multiplier=70)
