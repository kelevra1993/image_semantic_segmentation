import cv2
import torch
from ultrasound_segmentation.loss.focal_loss import FocalLoss
from ultrasound_segmentation.loss.focal_loss_functions import create_input_data
from utilities.tensor_utilities import print_tensor_status


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
        "circle": {"left_logit": 1.2, "right_logit": 1.0, "alpha": 1.0},
        "square": {"left_logit": 1.2, "right_logit": 1.0, "alpha": 1.0},
        "pentagon": {"left_logit": 1.2, "right_logit": 1.0, "alpha": 1.0},
        "ellipse": {"left_logit": 1.2, "right_logit": 1.0, "alpha": 1.0},
    }

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
                           gamma=5.0, device=device, dtype=torch.float32)

    # 4. Compute Loss
    loss, focal_loss_image = focal_loss(prediction_tensor, ground_truth_tensor)

    # Globally min-max normalize the image
    focal_min, focal_max = focal_loss_image[0].min(), focal_loss_image[0].max()
    focal_loss_class_image = (focal_loss_image[0] - focal_min) / (focal_max - focal_min + 1e-8)

    print_tensor_status(prediction_tensor, "prediction_tensor")

    for index, object_class in enumerate(parameter_dictionary, start=1):
        index_model_predictions = torch.sigmoid(prediction_tensor[0, index])

        # Define grid properties (using scaled down windows to fit on a normal screen)
        window_size = 400
        spacing_x = 100
        spacing_y = 100  # larger y spacing to account for OS window title bars

        # Calculate X and Y positions
        col_0_x = spacing_x
        col_1_x = col_0_x + window_size + spacing_x
        col_2_x = col_1_x + window_size + spacing_x
        row_y = spacing_y + (index - 1) * (window_size + spacing_y)

        # 1. Model Prediction Window
        prediction_name = f"Model Prediction - {object_class}"
        cv2.namedWindow(prediction_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(prediction_name, window_size, window_size)
        cv2.imshow(prediction_name, index_model_predictions.detach().cpu().numpy())

        cv2.moveWindow(prediction_name, col_1_x, row_y)

        # 2. Ground Truth Window
        ground_truth_name = f"Ground Truth - {object_class}"
        cv2.namedWindow(ground_truth_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(ground_truth_name, window_size, window_size)
        cv2.imshow(ground_truth_name, ground_truth_tensor[0, index].detach().cpu().numpy())
        cv2.moveWindow(ground_truth_name, col_0_x, row_y)

        # 3. Focal Loss Window
        focal_loss_name = f"Focal Loss - {object_class}"
        cv2.namedWindow(focal_loss_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(focal_loss_name, window_size, window_size)
        cv2.imshow(focal_loss_name, focal_loss_class_image.detach().cpu().numpy())
        cv2.moveWindow(focal_loss_name, col_2_x, row_y)

        # print_tensor_status(index_model_predictions, name="index_model_predictions")
    cv2.waitKey(0)


if __name__ == "__main__":
    debug_focal_loss()
