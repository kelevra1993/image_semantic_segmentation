import torch
import torch.nn.functional as F


def generate_segmentation_error_map(ground_truth: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """
    Creates an error map highlighting True Positives (Green), False Positives (Red),
    and False Negatives (Yellow).

    Args:
        ground_truth (torch.Tensor): The ground truth binary mask of shape (B, 1, H, W).
        prediction (torch.Tensor): The predicted binary mask of shape (B, 1, H, W).

    Returns:
        torch.Tensor: An RGB tensor of shape (B, 3, H, W) containing the error map.
    """
    true_positive = ((ground_truth == 1) & (prediction == 1)).squeeze(1)
    false_positive = ((prediction == 1) & (ground_truth == 0)).squeeze(1)
    false_negative = ((ground_truth == 1) & (prediction == 0)).squeeze(1)

    batch_size, _, height, width = ground_truth.shape
    error_map = torch.zeros((batch_size, 3, height, width), device=ground_truth.device, dtype=ground_truth.dtype)

    # True Positives: Green (0, 1, 0)
    error_map[:, 1, :, :][true_positive] = 1.0

    # False Positives: Red (1, 0, 0)
    error_map[:, 0, :, :][false_positive] = 1.0

    # False Negatives: Yellow (1, 1, 0)
    error_map[:, 0, :, :][false_negative] = 1.0
    error_map[:, 1, :, :][false_negative] = 1.0

    return error_map


def generate_boundary_overlay(original_image: torch.Tensor, mask: torch.Tensor, color: list[float],
                              alpha: float = 0.3) -> torch.Tensor:
    """
    Overlays the binary mask onto the original image with transparency and adds a bold boundary contour.

    Args:
        original_image (torch.Tensor): The grayscale input image (B, 1, H, W).
        mask (torch.Tensor): The binary mask (B, 1, H, W).
        color (list[float]): The RGB color for the overlay and boundary.
        alpha (float): Transparency level for the mask fill (0.0 = invisible, 1.0 = solid).

    Returns:
        torch.Tensor: An RGB tensor (B, 3, H, W) with the mask and bold boundary overlaid.
    """
    # Create bold boundary (dilation - erosion with kernel size 5 for a thicker border)
    dilated_mask = F.max_pool2d(mask, kernel_size=5, stride=1, padding=2)
    eroded_mask = -F.max_pool2d(-mask, kernel_size=5, stride=1, padding=2)
    boundary = (dilated_mask - eroded_mask) > 0
    boundary = boundary.squeeze(1)  # Shape: (B, H, W)

    mask_squeezed = mask.squeeze(1) > 0  # Shape: (B, H, W)

    # Ensure the original image is 3-channel RGB for coloring
    if original_image.shape[1] == 1:
        overlay = original_image.repeat(1, 3, 1, 1)
    else:
        overlay = original_image.clone()

    for channel_index in range(3):
        channel_data = overlay[:, channel_index, :, :]

        # 1. Apply transparent mask fill
        blended = (1.0 - alpha) * channel_data + alpha * color[channel_index]
        channel_data = torch.where(mask_squeezed, blended, channel_data)

        # 2. Apply solid bold boundary
        solid_boundary = torch.full_like(channel_data, color[channel_index])
        channel_data = torch.where(boundary, solid_boundary, channel_data)

        overlay[:, channel_index, :, :] = channel_data

    return overlay


def create_evaluation_row_for_channel(original_image: torch.Tensor, ground_truth: torch.Tensor,
                                      prediction: torch.Tensor) -> list[torch.Tensor]:
    """
    Creates a list of standardized evaluation visualization tensors for a single class channel.
    This fulfills the requirement of displaying the ground truth, prediction, the TP/FP/FN error map,
    and the original image with the predicted boundary overlaid.

    Args:
        original_image (torch.Tensor): The grayscale input image (B, 1, H, W).
        ground_truth (torch.Tensor): The ground truth binary mask (B, 1, H, W).
        prediction (torch.Tensor): The predicted binary mask (B, 1, H, W).

    Returns:
        list[torch.Tensor]: A list of RGB tensors containing [Original Image, Ground Truth, Prediction, Error Map, Boundary Overlay].
    """
    # Convert single channel masks to 3-channel grayscale for concatenation compatibility
    original_image_color = original_image.repeat(1, 3, 1, 1) if original_image.shape[1] == 1 else original_image
    ground_truth_color = ground_truth.repeat(1, 3, 1, 1)
    prediction_color = prediction.repeat(1, 3, 1, 1)

    error_map_color = generate_segmentation_error_map(ground_truth=ground_truth, prediction=prediction)
    boundary_overlay_color = generate_boundary_overlay(original_image=original_image, mask=prediction,
                                                       color=[0.0, 0.0, 1.0], alpha=0.3)  # Blue bold overlay

    return [original_image_color, ground_truth_color, prediction_color, error_map_color, boundary_overlay_color]
