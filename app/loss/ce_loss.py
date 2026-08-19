import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    """
    Computes the Cross Entropy loss between the predicted logits and the one-hot ground truth masks.
    
    This class wraps PyTorch's `nn.CrossEntropyLoss` which combines a LogSoftmax layer and the NLLLoss
    in one single class, applying Softmax over the channel dimension to enforce mutually exclusive classes.
    """
    
    def __init__(self) -> None:
        """
        Initializes the CrossEntropyLoss class.
        
        Args:
            None
            
        Returns:
            None
        """
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, model_predictions: torch.Tensor, ground_truths: torch.Tensor) -> torch.Tensor:
        """
        Calculates the cross entropy loss between the model predictions and the target masks
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
