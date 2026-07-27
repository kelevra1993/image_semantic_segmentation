from math import gamma

from typing import List

import torch
from torch import nn
import cv2
from utilities.tensor_utilities import print_tensor_status, print_tensor_list


class FocalLoss(nn.Module):
    """

    """

    def __init__(self, alpha: List[float], gamma: float, device: torch.device, dtype: torch.dtype):
        """"""
        super().__init__()

        self.device = device
        self.dtype = dtype

        # Resize alpha to the proper shape for broadcasting
        # (1, number_classes, 1, 1) to match (batch, number_classes, height, width)
        self.alpha = torch.tensor(alpha, device=self.device, dtype=self.dtype)
        self.alpha = self.alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

        self.gamma = torch.tensor(gamma, device=self.device, dtype=self.dtype)

    def forward(self, model_predictions: torch.Tensor, ground_truths: torch.Tensor) -> torch.Tensor:
        """"""

        ground_truths = ground_truths.to(device=self.device)
        # print_tensor_status(model_predictions, "Model Predictions")
        # print_tensor_status(ground_truths, "Ground Truths")

        # Shape of model_predictions => (batch, number_classes, height, width)
        # Shape of ground_truths => (batch, number_classes, height, width) as one hot encoded on number_classes
        print_tensor_status(model_predictions)

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

        # Get the final normalized focal loss (and average it accross batches)
        batched_focal_loss = torch.sum(focal_loss_numerator / focal_loss_denominator, dim=[-1, -2])
        focal_loss = torch.mean(batched_focal_loss)

        focal_loss_image = focal_loss_numerator/ focal_loss_denominator
        for index in range(self.alpha.shape[1]):
            focal_loss_class_image = torch.nn.functional.normalize(focal_loss_image[0])
            index_model_predictions = torch.sigmoid(model_predictions[0, index])

            probabilities_image = torch.nn.functional.normalize(torch.softmax(model_predictions, dim=-3)[0,index])
            cv2.imshow(f"Model - Prediction", index_model_predictions.detach().cpu().numpy())
            cv2.imshow(f"Ground Truth", ground_truths[0, index].detach().cpu().numpy())
            cv2.imshow(f"Focal Loss", focal_loss_class_image.detach().cpu().numpy())
            print_tensor_status(index_model_predictions,name="index_model_predictions")
            cv2.waitKey(0)
        exit()

