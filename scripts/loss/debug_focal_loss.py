import cv2
import os
import torch
from typing import Dict, Tuple, List
from app.loss.focal_loss import FocalLoss
from scripts.loss.focal_loss_functions import (create_input_data,
                                               visualize_focal_loss,
                                               compute_class_probabilities,
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
                     alpha: List[float],
                     gamma: float,
                     save_visualization: bool = True) -> None:
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
        alpha (List[float]): List of alpha weighting factors for each class.
        gamma (float): Focusing parameter for the focal loss.
        save_visualization (bool): If True, saves the visualization image to disk. Defaults to True.
    """
    # Generate Data containing our object divided into two regions.
    ground_truth_tensor, prediction_tensor, object_information = create_input_data(
        image_size=image_size,
        background_logit=background_logit,
        logit_dictionary=parameter_dictionary,
        inactive_logit=inactive_logit,
        number_of_classes=number_classes,
        batch_size=batch_size)

    # Initialize FocalLoss
    device = torch.device('cpu')

    # Create Focal Loss Object And Compute The Loss
    focal_loss = FocalLoss(alpha=alpha, gamma=gamma, device=device, dtype=torch.float32)
    loss, focal_loss_image = focal_loss(prediction_tensor,
                                        ground_truth_tensor)

    # Globally min-max normalize the image
    focal_min, focal_max = focal_loss_image[0].min(), focal_loss_image[0].max()
    focal_loss_class_image = (focal_loss_image[0] - focal_min) / (focal_max - focal_min + 1e-8)

    # Compute predictions softmaxes
    probabilities = compute_class_probabilities(parameter_dictionary=parameter_dictionary,
                                                background_logit=background_logit,
                                                inactive_logit=inactive_logit)

    # Generate and print dataframe
    focal_loss_dataframe = create_focal_loss_dataframe(parameter_dictionary=parameter_dictionary,
                                                       probabilities=probabilities,
                                                       focal_loss_image=focal_loss_image,
                                                       object_information=object_information,
                                                       gamma=gamma)

    print(focal_loss_dataframe.to_string(justify='center'))

    # Get visualization Of Focal Loss
    visualize_focal_loss(parameter_dictionary=parameter_dictionary,
                         window_size=window_size, spacing_x=spacing_x, spacing_y=spacing_y,
                         focal_loss_class_image=focal_loss_class_image,
                         ground_truth_tensor=ground_truth_tensor, prediction_tensor=prediction_tensor,
                         object_information=object_information,
                         save_visualization=save_visualization)


if __name__ == "__main__":
    # Definition of logits for left and right part of our masked objects.
    test_parameter_dictionary = {
        "circle": {"left_logit": 2.8, "right_logit": 1.0, "alpha": 1.0},
        "square": {"left_logit": 2.8, "right_logit": 1.0, "alpha": 1.0},
        "pentagon": {"left_logit": 1.0, "right_logit": 5.0, "alpha": 1.0},
        "ellipse": {"left_logit": 2.8, "right_logit": 1.0, "alpha": 1.0}}

    # Definition of visualizers
    test_window_size = 200
    test_image_size = (test_window_size, test_window_size)
    test_background_logit = 0.8
    test_inactive_logit = -10.0
    test_number_classes = 5
    test_batch_size = 1
    test_spacing_x = 80
    test_spacing_y = 60
    test_alpha = [1.0] + [test_parameter_dictionary[key]["alpha"] for key in test_parameter_dictionary]
    test_gamma = 0.5

    # Run focal loss debugger and visualizer
    debug_focal_loss(parameter_dictionary=test_parameter_dictionary,
                     image_size=test_image_size,
                     background_logit=test_background_logit, inactive_logit=test_inactive_logit,
                     number_classes=test_number_classes, batch_size=test_batch_size,
                     window_size=test_window_size, spacing_x=test_spacing_x, spacing_y=test_spacing_y,
                     alpha=test_alpha, gamma=test_gamma)
