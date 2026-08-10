from typing import List, Tuple

import torch
from torch import nn


class FocalLoss(nn.Module):
    """
    Computes the Focal Loss between the predicted logits and the one-hot encoded ground truth masks.
    
    This loss function is designed to address severe class imbalance by down-weighting the loss 
    assigned to well-classified examples, thereby focusing the training on a sparse set of hard examples.
    """

    def __init__(self, alpha: List[float], gamma: float, device: torch.device, dtype: torch.dtype):
        """
        Initializes the FocalLoss class.
        
        Args:
            alpha (List[float]): A list of weighting factors for each class to address class imbalance.
            gamma (float): The focusing parameter to smoothly adjust the rate at which easy examples are down-weighted.
            device (torch.device): The device (e.g., CPU or GPU) on which the tensors will be allocated.
            dtype (torch.dtype): The expected data type for the internal tensors.
            
        Returns:
            None
        """
        super().__init__()

        self.device = device
        self.dtype = dtype

        # Resize alpha to the proper shape for broadcasting
        # (1, number_classes, 1, 1) to match (batch, number_classes, height, width)
        self.alpha = torch.tensor(alpha, device=self.device, dtype=self.dtype)
        self.alpha = self.alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

        self.gamma = torch.tensor(gamma, device=self.device, dtype=self.dtype)

    def forward(self, model_predictions: torch.Tensor,
                ground_truths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the focal loss.
        
        Args:
            model_predictions (torch.Tensor): The raw logit predictions from the model. 
                Expected shape is (batch, number_classes, height, width).
            ground_truths (torch.Tensor): The one-hot encoded ground truth masks.
                Expected shape is (batch, number_classes, height, width).
                
        Returns:
            tuple: A tuple containing the scalar mean focal loss and the focal loss image tensor.
        """

        ground_truths = ground_truths.to(device=self.device)
        # print_tensor_status(model_predictions, "Model Predictions")
        # print_tensor_status(ground_truths, "Ground Truths")

        # Shape of model_predictions => (batch, number_classes, height, width)
        # Shape of ground_truths => (batch, number_classes, height, width) as one hot encoded on number_classes
        # Compute the softmax for the input, done on the number_classes,
        # Shape should still be (batch, number_classes, height, width)
        model_output_probabilities = torch.softmax(model_predictions, dim=-3)

        # Extraction of probabilities of foreground
        # In order to do that we need to multiply our softmax by our ground truth matrix.
        # This will only keep the prediction for the class that we would like to keep (our pt)
        # By summing up along all number_classes dimensions we should end up with a tensor that only contains pt's
        # Shape (batch, height, width)
        model_output_probabilities = torch.sum(model_output_probabilities * ground_truths, dim=-3)

        # First get the alpha matrix (alpha being class dependent)
        # Multiply by ground truths and then sum over the classes to get : Shape (batch, height, width)
        alpha_tensor = torch.sum(self.alpha * ground_truths, dim=-3)

        # First term : alpha * (1-pt)^gamma
        first_term = alpha_tensor * torch.pow(
            input=torch.tensor(1.0, device=self.device, dtype=self.dtype) - model_output_probabilities,
            exponent=self.gamma)

        # Second term : log(pt)
        # Clamp probabilities since log(0.0) drops to -infinity
        second_term = torch.log(torch.clamp(model_output_probabilities, min=1e-10))

        # Numerator = - alpha * (1-pt)^gamma * log(pt)
        focal_loss_numerator = - first_term * second_term

        # Denominator normalization by number of positive pixel per class
        # Shape (batch, number_classes, height, width)
        # Minimum of 1 to avoid division by zero for a class that does not exist in an image.
        # and sum up along the classes dimension to also get : Shape (batch, height, width)
        positive_pixels_per_class = torch.sum(ground_truths, dim=[-1, -2], keepdim=True)
        focal_loss_denominator = torch.clamp(torch.sum(input=positive_pixels_per_class * ground_truths, dim=-3), min=1)

        # Get the final normalized focal loss (and average it across batches)
        focal_loss_image = focal_loss_numerator / focal_loss_denominator
        batched_focal_loss = torch.sum(focal_loss_image, dim=[-1, -2])
        focal_loss = torch.mean(batched_focal_loss)

        return focal_loss, focal_loss_image
