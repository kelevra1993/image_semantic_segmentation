import torch
import torch.nn as nn

class BCELoss(nn.Module):
    """
    Computes the Binary Cross Entropy loss between the predicted logits and the ground truth masks.
    
    This class wraps PyTorch's `nn.BCEWithLogitsLoss` which combines a Sigmoid layer and the BCELoss
    in one single class, offering better numerical stability than using a plain Sigmoid followed
    by a BCELoss.
    """
    
    def __init__(self) -> None:
        """
        Initializes the BCELoss class.
        
        Args:
            None
            
        Returns:
            None
        """
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, predictions: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """
        Calculates the binary cross entropy loss.
        
        Args:
            predictions (torch.Tensor): The raw logit predictions from the model.
            ground_truth (torch.Tensor): The binary ground truth masks.
                
        Returns:
            torch.Tensor: The computed scalar loss value.
        """
        return self.criterion(predictions, ground_truth)
