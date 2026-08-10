import cv2
import os
import torch
from typing import Dict, Tuple
from app.loss.focal_loss import FocalLoss
from scripts.loss.focal_loss_functions import (create_input_data,
                                               visualize_focal_loss,
                                               compute_class_probabilities,
                                               print_class_probabilities,
                                               create_focal_loss_dataframe)
from app.utilities.os_utilities import print_yellow
from app.utilities.tensor_utilities import print_tensor_status

# Suppress Qt C++ warnings (like QFontDatabase missing fonts)
os.environ["QT_LOGGING_RULES"] = "*=false"


def debug_focal_loss(parameter_dictionary: Dict[str, Dict[str, float]],
                     image_size: Tuple[int, int],
                     background_logit: float,
                     inactive_logit: float,
                     number_classes: int,
                     batch_size: int,
                     window_size: int,
                     spacing_x: int,
                     spacing_y: int,
                     length_separator: int) -> None:
    """
    Creates a 5-class synthetic testing scenario to visually verify the focal loss implementation.

    This function utilizes the modular data generators from focal_loss_functions.py to create
    a full 5-channel image containing a circle, square, pentagon, and ellipse in each quadrant.
    It passes the generated tensors to the FocalLoss module to review behavior under different
    logit configurations.
    
    Args:
        parameter_dictionary (Dict[str, Dict[str, float]]): The dictionary of class logits.
        image_size (Tuple[int, int]): The total (height, width) of the image.
        background_logit (float): The default prediction logit for the background class.
        inactive_logit (float): Logit value for pixels not belonging to a class.
        number_classes (int): Total number of output classes.
        batch_size (int): The batch size of the generated tensors.
        window_size (int): The display width and height for each OpenCV window.
        spacing_x (int): Horizontal spacing between windows in pixels.
        spacing_y (int): Vertical spacing between windows in pixels.
        length_separator (int): The number of characters to use for the visual separator.
    """
    # Generate Data containing our object divided into two regions.
    ground_truth_tensor, prediction_tensor, positions = create_input_data(image_size=image_size,
                                                                          background_logit=background_logit,
                                                                          logit_dictionary=parameter_dictionary,
                                                                          inactive_logit=inactive_logit,
                                                                          number_of_classes=number_classes,
                                                                          batch_size=batch_size)

    # Initialize FocalLoss
    device = torch.device('cpu')

    focal_loss = FocalLoss(alpha=[1.0,
                                  parameter_dictionary["circle"]["alpha"],
                                  parameter_dictionary["square"]["alpha"],
                                  parameter_dictionary["pentagon"]["alpha"],
                                  parameter_dictionary["ellipse"]["alpha"]],
                           gamma=0.50, device=device, dtype=torch.float32)

    # Compute Loss
    loss, focal_loss_image = focal_loss(prediction_tensor, ground_truth_tensor)

    # Globally min-max normalize the image
    focal_min, focal_max = focal_loss_image[0].min(), focal_loss_image[0].max()
    focal_loss_class_image = (focal_loss_image[0] - focal_min) / (focal_max - focal_min + 1e-8)

    # Compute predictions softmaxes
    probabilities = compute_class_probabilities(parameter_dictionary=parameter_dictionary,
                                                background_logit=background_logit,
                                                inactive_logit=inactive_logit)

    # Generate and print dataframe
    focal_loss_dataframe = create_focal_loss_dataframe(probabilities=probabilities,
                                                       focal_loss_image=focal_loss_image,
                                                       positions=positions)

    print(focal_loss_dataframe.to_string())

    # Get visualization Of Focal Loss
    visualize_focal_loss(parameter_dictionary=parameter_dictionary,
                         window_size=window_size, spacing_x=spacing_x, spacing_y=spacing_y,
                         focal_loss_class_image=focal_loss_class_image,
                         ground_truth_tensor=ground_truth_tensor, prediction_tensor=prediction_tensor)


if __name__ == "__main__":
    # Definition of logits for left and right part of our masked objects.
    test_parameter_dictionary = {
        "circle": {"left_logit": 2.8, "right_logit": 1.0, "alpha": 1.0},
        "square": {"left_logit": 2.8, "right_logit": 1.0, "alpha": 1.0},
        "pentagon": {"left_logit": 1.0, "right_logit": 5.0, "alpha": 1.0},
        "ellipse": {"left_logit": 2.8, "right_logit": 1.0, "alpha": 1.0}}

    # Definition of visualizers
    test_image_size = (400, 400)
    test_background_logit = 0.0
    test_inactive_logit = -20.0
    test_number_classes = 5
    test_batch_size = 1
    test_window_size = 400
    test_spacing_x = 80
    test_spacing_y = 60
    test_length_separator = 50

    # Run focal loss debugger and visualizer
    debug_focal_loss(parameter_dictionary=test_parameter_dictionary,
                     image_size=test_image_size,
                     background_logit=test_background_logit, inactive_logit=test_inactive_logit,
                     number_classes=test_number_classes, batch_size=test_batch_size,
                     window_size=test_window_size, spacing_x=test_spacing_x, spacing_y=test_spacing_y,
                     length_separator=test_length_separator)
