import cv2
import os
import torch
from app.loss.focal_loss import FocalLoss
from scripts.loss.focal_loss_functions import create_input_data, visualize_focal_loss
from app.utilities.os_utilities import print_yellow
from app.utilities.tensor_utilities import print_tensor_status

# Suppress Qt C++ warnings (like QFontDatabase missing fonts)
os.environ["QT_LOGGING_RULES"] = "*=false"


def debug_focal_loss() -> None:
    """
    Creates a 5-class synthetic testing scenario to visually verify the focal loss implementation.

    This function utilizes the modular data generators from focal_loss_functions.py to create
    a full 5-channel image containing a circle, square, pentagon, and ellipse in each quadrant.
    It passes the generated tensors to the FocalLoss module to review behavior under different
    logit configurations.
    """
    # 1. Define Logits Dictionary
    parameter_dictionary = {
        "circle": {"left_logit": 2.8, "right_logit": 1.0, "alpha": 1.0},
        "square": {"left_logit": 2.8, "right_logit": 1.0, "alpha": 1.0},
        "pentagon": {"left_logit": 1.0, "right_logit": 5.0, "alpha": 1.0},
        "ellipse": {"left_logit": 2.8, "right_logit": 1.0, "alpha": 1.0}}

    # 2. Generate Data
    image_size = (400, 400)
    ground_truth_tensor, prediction_tensor = create_input_data(
        image_size=image_size, background_logit=0.0, logit_dictionary=parameter_dictionary)

    # 3. Initialize FocalLoss
    device = torch.device('cpu')

    focal_loss = FocalLoss(alpha=[1.0,
                                  parameter_dictionary["circle"]["alpha"],
                                  parameter_dictionary["square"]["alpha"],
                                  parameter_dictionary["pentagon"]["alpha"],
                                  parameter_dictionary["ellipse"]["alpha"]],
                           gamma=0.50, device=device, dtype=torch.float32)

    # 4. Compute Loss
    loss, focal_loss_image = focal_loss(prediction_tensor, ground_truth_tensor)

    # Globally min-max normalize the image
    focal_min, focal_max = focal_loss_image[0].min(), focal_loss_image[0].max()
    focal_loss_class_image = (focal_loss_image[0] - focal_min) / (focal_max - focal_min + 1e-8)

    # Compute predictions softmaxes
    print(50*"#")
    print_yellow("----Class Probabilities----")
    for predicted_class, predicted_information in parameter_dictionary.items():
        softmax = torch.softmax(
            torch.tensor(data=[predicted_information["left_logit"], predicted_information["right_logit"]]),dim=0)
        print(f" Softmax For {predicted_class.upper()} :: {softmax.numpy().round(3)}")
    print(50 * "#")


    # Define grid properties (using scaled down windows to fit on a normal screen)
    window_size = 400
    spacing_x = 80
    spacing_y = 60  # larger y spacing to account for OS window title bars

    visualize_focal_loss(
        parameter_dictionary=parameter_dictionary,
        window_size=window_size,
        spacing_x=spacing_x,
        spacing_y=spacing_y,
        focal_loss_class_image=focal_loss_class_image,
        ground_truth_tensor=ground_truth_tensor,
        prediction_tensor=prediction_tensor
    )

if __name__ == "__main__":
    debug_focal_loss()
