import cv2
import torch
from ultrasound_segmentation.loss.focal_loss import FocalLoss
from ultrasound_segmentation.loss.focal_loss_functions import create_input_data
from utilities.tensor_utilities import print_tensor_status


def debug_focal_loss() -> None:
    """
    Creates a 5-class synthetic testing scenario to visually verify the focal loss implementation.

    This function utilizes the modular data generators from focal_loss_functions.py to create
    a full 5-channel image containing a circle, square, triangle, and ellipse in each quadrant.
    It passes the generated tensors to the FocalLoss module to review behavior under different
    logit configurations.
    """
    # 1. Define Logits Dictionary
    logit_dictionary = {
        "circle": {"left_logit": 5.0, "right_logit": 1.0},
        "square": {"left_logit": 3.0, "right_logit": -1.0},
        "triangle": {"left_logit": 2.0, "right_logit": 0.0},
        "ellipse": {"left_logit": 4.0, "right_logit": -2.0},
    }

    # 2. Generate Data
    image_size = (400, 400)
    ground_truth_tensor, prediction_tensor = create_input_data(
        image_size=image_size, background_logit=0.0, logit_dictionary=logit_dictionary)

    # 3. Initialize FocalLoss
    device = torch.device('cpu')
    number_of_classes = 5
    focal_loss = FocalLoss(alpha=[1.0] * number_of_classes, gamma=1.0, device=device, dtype=torch.float32)

    # 4. Compute Loss
    print(f"Feeding 5-channel inputs to FocalLoss...")
    loss, focal_loss_image = focal_loss(prediction_tensor, ground_truth_tensor)

    print(f"Computed Focal Loss: {loss.item()}")

    focal_loss_class_image = torch.nn.functional.normalize(focal_loss_image[0])
    print_tensor_status(prediction_tensor, "prediction_tensor")

    for index, object_class in enumerate(logit_dictionary, start=1):
        index_model_predictions = torch.sigmoid(prediction_tensor[0, index])

        # probabilities_image = torch.nn.functional.normalize(torch.softmax(prediction_tensor, dim=-3)[0,index])
        prediction_name = f"Model Prediction - {object_class}"
        cv2.imshow(prediction_name, index_model_predictions.detach().cpu().numpy())
        cv2.moveWindow(prediction_name,x=10,y=index*10)

        ground_truth_name = f"Ground Truth - {object_class}"
        cv2.imshow(ground_truth_name, ground_truth_tensor[0, index].detach().cpu().numpy())
        cv2.moveWindow(ground_truth_name,x=image_size[0]*2,y=image_size[0]*index)

        focal_loss_name = f"Focal Loss - {object_class}"
        cv2.imshow(focal_loss_name, focal_loss_class_image.detach().cpu().numpy())
        cv2.moveWindow(focal_loss_name,x=image_size[0]*4,y=image_size[0]*index)

        # print_tensor_status(index_model_predictions, name="index_model_predictions")
    cv2.waitKey(0)


if __name__ == "__main__":
    debug_focal_loss()
