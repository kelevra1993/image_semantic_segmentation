import torch
import cv2
import numpy as np
from typing import Tuple, Dict


def create_circle_data(
        quadrant_width: int,
        quadrant_height: int,
        left_logit: float,
        right_logit: float,
        inactive_logit: float = -10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates ground truth and prediction logits for a circle.
    
    Args:
        quadrant_width (int): The width of the image quadrant.
        quadrant_height (int): The height of the image quadrant.
        left_logit (float): The prediction logit for the left half of the circle.
        right_logit (float): The prediction logit for the right half of the circle.
        inactive_logit (float): The default logit for pixels outside the circle.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: The ground truth mask and prediction logits mask.
    """
    ground_truth_numpy_array = np.zeros((quadrant_height, quadrant_width), dtype=np.uint8)
    circle_center = (quadrant_width // 2, quadrant_height // 2)
    circle_radius = min(quadrant_width, quadrant_height) // 4

    cv2.circle(img=ground_truth_numpy_array, center=circle_center, radius=circle_radius, color=1, thickness=-1)

    prediction_logits = np.full((quadrant_height, quadrant_width), inactive_logit, dtype=np.float32)
    left_circle_mask = (ground_truth_numpy_array == 1) & (np.arange(quadrant_width) < circle_center[0])
    prediction_logits[left_circle_mask] = left_logit

    right_circle_mask = (ground_truth_numpy_array == 1) & (np.arange(quadrant_width) >= circle_center[0])
    prediction_logits[right_circle_mask] = right_logit

    return ground_truth_numpy_array, prediction_logits


def create_square_data(
        quadrant_width: int,
        quadrant_height: int,
        left_logit: float,
        right_logit: float,
        inactive_logit: float = -10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates ground truth and prediction logits for a square.
    
    Args:
        quadrant_width (int): The width of the image quadrant.
        quadrant_height (int): The height of the image quadrant.
        left_logit (float): The prediction logit for the left half of the square.
        right_logit (float): The prediction logit for the right half of the square.
        inactive_logit (float): The default logit for pixels outside the square.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: The ground truth mask and prediction logits mask.
    """
    ground_truth_numpy_array = np.zeros((quadrant_height, quadrant_width), dtype=np.uint8)
    square_center_x, square_center_y = quadrant_width // 2, quadrant_height // 2
    square_half_size = min(quadrant_width, quadrant_height) // 4

    top_left_point = (square_center_x - square_half_size, square_center_y - square_half_size)
    bottom_right_point = (square_center_x + square_half_size, square_center_y + square_half_size)

    cv2.rectangle(img=ground_truth_numpy_array, pt1=top_left_point, pt2=bottom_right_point, color=1, thickness=-1)

    prediction_logits = np.full((quadrant_height, quadrant_width), inactive_logit, dtype=np.float32)
    left_square_mask = (ground_truth_numpy_array == 1) & (np.arange(quadrant_width) < square_center_x)
    prediction_logits[left_square_mask] = left_logit

    right_square_mask = (ground_truth_numpy_array == 1) & (np.arange(quadrant_width) >= square_center_x)
    prediction_logits[right_square_mask] = right_logit

    return ground_truth_numpy_array, prediction_logits


def create_triangle_data(
        quadrant_width: int,
        quadrant_height: int,
        left_logit: float,
        right_logit: float,
        inactive_logit: float = -10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates ground truth and prediction logits for a triangle.
    
    Args:
        quadrant_width (int): The width of the image quadrant.
        quadrant_height (int): The height of the image quadrant.
        left_logit (float): The prediction logit for the left half of the triangle.
        right_logit (float): The prediction logit for the right half of the triangle.
        inactive_logit (float): The default logit for pixels outside the triangle.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: The ground truth mask and prediction logits mask.
    """
    ground_truth_numpy_array = np.zeros((quadrant_height, quadrant_width), dtype=np.uint8)
    triangle_center_x, triangle_center_y = quadrant_width // 2, quadrant_height // 2
    triangle_offset = min(quadrant_width, quadrant_height) // 4

    # Pointing upwards
    point_1 = [triangle_center_x, triangle_center_y - triangle_offset]
    point_2 = [triangle_center_x - triangle_offset, triangle_center_y + triangle_offset]
    point_3 = [triangle_center_x + triangle_offset, triangle_center_y + triangle_offset]

    triangle_points = np.array([point_1, point_2, point_3], np.int32)
    triangle_points = triangle_points.reshape((-1, 1, 2))

    cv2.fillPoly(img=ground_truth_numpy_array, pts=[triangle_points], color=1)

    prediction_logits = np.full((quadrant_height, quadrant_width), inactive_logit, dtype=np.float32)
    left_triangle_mask = (ground_truth_numpy_array == 1) & (np.arange(quadrant_width) < triangle_center_x)
    prediction_logits[left_triangle_mask] = left_logit

    right_triangle_mask = (ground_truth_numpy_array == 1) & (np.arange(quadrant_width) >= triangle_center_x)
    prediction_logits[right_triangle_mask] = right_logit

    return ground_truth_numpy_array, prediction_logits


def create_ellipse_data(
        quadrant_width: int,
        quadrant_height: int,
        left_logit: float,
        right_logit: float,
        inactive_logit: float = -10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates ground truth and prediction logits for an ellipse.
    
    Args:
        quadrant_width (int): The width of the image quadrant.
        quadrant_height (int): The height of the image quadrant.
        left_logit (float): The prediction logit for the left half of the ellipse.
        right_logit (float): The prediction logit for the right half of the ellipse.
        inactive_logit (float): The default logit for pixels outside the ellipse.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: The ground truth mask and prediction logits mask.
    """
    ground_truth_numpy_array = np.zeros((quadrant_height, quadrant_width), dtype=np.uint8)
    ellipse_center = (quadrant_width // 2, quadrant_height // 2)
    axes_length = (min(quadrant_width, quadrant_height) // 3, min(quadrant_width, quadrant_height) // 6)

    cv2.ellipse(img=ground_truth_numpy_array, center=ellipse_center, axes=axes_length, angle=0, startAngle=0,
                endAngle=360, color=1, thickness=-1)

    prediction_logits = np.full((quadrant_height, quadrant_width), inactive_logit, dtype=np.float32)
    left_ellipse_mask = (ground_truth_numpy_array == 1) & (np.arange(quadrant_width) < ellipse_center[0])
    prediction_logits[left_ellipse_mask] = left_logit

    right_ellipse_mask = (ground_truth_numpy_array == 1) & (np.arange(quadrant_width) >= ellipse_center[0])
    prediction_logits[right_ellipse_mask] = right_logit

    return ground_truth_numpy_array, prediction_logits


def create_input_data(image_size: Tuple[int, int],
                      background_logit: float,
                      logit_dictionary: Dict[str, Dict[str, float]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a 5-channel ground truth and prediction tensor for focal loss debugging.
    
    The channels correspond to:
    0: Background
    1: Circle (Top-Left quadrant)
    2: Square (Top-Right quadrant)
    3: Triangle (Bottom-Left quadrant)
    4: Ellipse (Bottom-Right quadrant)
    
    Args:
        image_size (Tuple[int, int]): The total (height, width) of the image.
        background_logit (float): The default prediction logit for the background class.
        logit_dictionary (Dict[str, Dict[str, float]]): A dictionary specifying the prediction logits 
            for each foreground class. Must include nested dicts for 'circle', 'square', 'triangle', 
            and 'ellipse' with 'left_logit' and 'right_logit' specified.
            
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The ground truth tensor and the prediction tensor, 
            both of shape (1, 5, height, width).
    """

    image_height, image_width = image_size
    quadrant_height, quadrant_width = image_height // 2, image_width // 2
    number_of_classes = 5
    batch_size = 1
    inactive_logit = -10.0

    ground_truth_tensor = torch.zeros((batch_size, number_of_classes, image_height, image_width), dtype=torch.float32)
    prediction_tensor = torch.full((batch_size, number_of_classes, image_height, image_width), inactive_logit,
                                   dtype=torch.float32)

    # 0. Background
    prediction_tensor[0, 0, :, :] = background_logit

    # 1. Circle (Top-Left)
    circle_left = logit_dictionary["circle"]["left_logit"]
    circle_right = logit_dictionary["circle"]["right_logit"]
    circle_ground_truth, circle_prediction = create_circle_data(quadrant_width, quadrant_height, circle_left,
                                                                circle_right, inactive_logit)

    ground_truth_tensor[0, 1, 0:quadrant_height, 0:quadrant_width] = torch.from_numpy(circle_ground_truth)
    prediction_tensor[0, 1, 0:quadrant_height, 0:quadrant_width] = torch.from_numpy(circle_prediction)

    # 2. Square (Top-Right)
    square_left = logit_dictionary["square"]["left_logit"]
    square_right = logit_dictionary["square"]["right_logit"]
    square_ground_truth, square_prediction = create_square_data(quadrant_width, quadrant_height, square_left,
                                                                square_right, inactive_logit)
    ground_truth_tensor[0, 2, 0:quadrant_height, quadrant_width:image_width] = torch.from_numpy(square_ground_truth)
    prediction_tensor[0, 2, 0:quadrant_height, quadrant_width:image_width] = torch.from_numpy(square_prediction)

    # 3. Triangle (Bottom-Left)
    triangle_left = logit_dictionary["triangle"]["left_logit"]
    triangle_right = logit_dictionary["triangle"]["right_logit"]
    triangle_ground_truth, triangle_prediction = create_triangle_data(quadrant_width, quadrant_height, triangle_left,
                                                                      triangle_right, inactive_logit)
    ground_truth_tensor[0, 3, quadrant_height:image_height, 0:quadrant_width] = torch.from_numpy(triangle_ground_truth)
    prediction_tensor[0, 3, quadrant_height:image_height, 0:quadrant_width] = torch.from_numpy(triangle_prediction)

    # 4. Ellipse (Bottom-Right)
    ellipse_left = logit_dictionary["ellipse"]["left_logit"]
    ellipse_right = logit_dictionary["ellipse"]["right_logit"]
    ellipse_ground_truth, ellipse_prediction = create_ellipse_data(quadrant_width, quadrant_height, ellipse_left,
                                                                   ellipse_right, inactive_logit)
    ground_truth_tensor[0, 4, quadrant_height:image_height, quadrant_width:image_width] = torch.from_numpy(
        ellipse_ground_truth)
    prediction_tensor[0, 4, quadrant_height:image_height, quadrant_width:image_width] = torch.from_numpy(
        ellipse_prediction)

    # Background Ground Truth (Where all other classes are 0)
    foreground_sum = torch.sum(ground_truth_tensor[0, 1:, :, :], dim=0)
    ground_truth_tensor[0, 0, :, :] = 1.0 - foreground_sum

    return ground_truth_tensor, prediction_tensor
