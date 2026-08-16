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
        
    def forward(self, model_predictions: torch.Tensor, ground_truths: torch.Tensor) -> torch.Tensor:
        """
        Calculates the binary cross entropy loss.
        
        Args:
            model_predictions (torch.Tensor): The raw logit predictions from the model.
            ground_truths (torch.Tensor): The binary ground truth masks.
                
        Returns:
            torch.Tensor: The computed scalar loss value.
        """
        return self.criterion(model_predictions, ground_truths)
