from typing import List

import torch
from torch import nn


class WeightedCrossEntropyLoss(nn.Module):
    """
    Computes the Weighted Cross Entropy loss between the predicted logits and the one-hot ground truth masks.
    
    This class wraps PyTorch's `nn.CrossEntropyLoss` and applies a class-specific weight to handle 
    imbalances between different classes in the segmentation task. Softmax is applied over the channel dimension.
    """
    
    def __init__(self, weights: List[float], device: torch.device, dtype: torch.dtype) -> None:
        """
        Initializes the WeightedCrossEntropyLoss class by preparing the class-specific weights tensor.
        
        Args:
            weights (List[float]): A list of weighting factors for each class.
            device (torch.device): The device (e.g., CPU or GPU) on which the tensors will be allocated.
            dtype (torch.dtype): The expected data type for the internal tensors.
            
        Returns:
            None
        """
        super().__init__()
        
        self.device = device
        self.dtype = dtype
        
        # Create weight tensor. For nn.CrossEntropyLoss, weight should be a 1D tensor of size C.
        weight_tensor = torch.tensor(weights, device=self.device, dtype=self.dtype)
        
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        
    def forward(self, model_predictions: torch.Tensor, ground_truths: torch.Tensor) -> torch.Tensor:
        """
        Calculates the weighted cross entropy loss between the model predictions and the target masks
        to provide a gradient for updating the model weights during the backpropagation step of the training loop.
        
        Args:
            model_predictions (torch.Tensor): The raw logit predictions from the model.
                Expected shape is (batch, number_classes, height, width).
            ground_truths (torch.Tensor): The one-hot encoded ground truth masks.
                Expected shape is (batch, number_classes, height, width).
                
        Returns:
            torch.Tensor: The computed scalar loss value.
        """
        return self.criterion(input=model_predictions, target=ground_truths)
