import torch
from ultrasound_segmentation.loss.focal_loss import FocalLoss
from ultrasound_segmentation.loss.focal_loss_functions import create_input_data

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
        "background_logit": 0.0,
        "circle": {"left_logit": 5.0, "right_logit": 1.0},
        "square": {"left_logit": 3.0, "right_logit": -1.0},
        "triangle": {"left_logit": 2.0, "right_logit": 0.0},
        "ellipse": {"left_logit": 4.0, "right_logit": -2.0},
    }

    # 2. Generate Data
    image_size = (400, 400)
    ground_truth_tensor, prediction_tensor = create_input_data(
        image_size=image_size, 
        logit_dictionary=logit_dictionary
    )

    # 3. Initialize FocalLoss
    device = torch.device('cpu')
    number_of_classes = 5
    focal_loss = FocalLoss(
        alpha=[1.0] * number_of_classes, 
        gamma=1.0, 
        device=device, 
        dtype=torch.float32
    )

    # 4. Compute Loss
    print(f"Feeding 5-channel inputs to FocalLoss...")
    loss = focal_loss(prediction_tensor, ground_truth_tensor)
    
    print(f"Computed Focal Loss: {loss.item()}")

if __name__ == "__main__":
    debug_focal_loss()
