import torch
import cv2
import numpy as np
import math
from typing import Tuple, Dict


def create_circle_data(quadrant_width: int, quadrant_height: int, left_logit: float, right_logit: float,
                       inactive_logit: float = -10.0) -> Tuple[np.ndarray, np.ndarray]:
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


def create_square_data(quadrant_width: int, quadrant_height: int, left_logit: float,
                       right_logit: float, inactive_logit: float = -10.0) -> Tuple[np.ndarray, np.ndarray]:
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


def create_pentagon_data(quadrant_width: int, quadrant_height: int, left_logit: float,
                         right_logit: float, inactive_logit: float = -10.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates ground truth and prediction logits for a pentagon.
    
    Args:
        quadrant_width (int): The width of the image quadrant.
        quadrant_height (int): The height of the image quadrant.
        left_logit (float): The prediction logit for the left half of the pentagon.
        right_logit (float): The prediction logit for the right half of the pentagon.
        inactive_logit (float): The default logit for pixels outside the pentagon.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: The ground truth mask and prediction logits mask.
    """
    ground_truth_numpy_array = np.zeros((quadrant_height, quadrant_width), dtype=np.uint8)
    pentagon_center_x, pentagon_center_y = quadrant_width // 2, quadrant_height // 2
    pentagon_radius = min(quadrant_width, quadrant_height) // 5

    pentagon_points = []
    for i in range(5):
        angle_rad = math.radians(i * 72 - 90)
        x = pentagon_center_x + int(pentagon_radius * math.cos(angle_rad))
        y = pentagon_center_y + int(pentagon_radius * math.sin(angle_rad))
        pentagon_points.append([x, y])

    pentagon_points = np.array(pentagon_points, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img=ground_truth_numpy_array, pts=[pentagon_points], color=1)

    prediction_logits = np.full((quadrant_height, quadrant_width), inactive_logit, dtype=np.float32)
    left_pentagon_mask = (ground_truth_numpy_array == 1) & (np.arange(quadrant_width) < pentagon_center_x)
    prediction_logits[left_pentagon_mask] = left_logit

    right_pentagon_mask = (ground_truth_numpy_array == 1) & (np.arange(quadrant_width) >= pentagon_center_x)
    prediction_logits[right_pentagon_mask] = right_logit

    return ground_truth_numpy_array, prediction_logits


def create_ellipse_data(quadrant_width: int, quadrant_height: int, left_logit: float, right_logit: float,
                        inactive_logit: float = -10.0) -> Tuple[np.ndarray, np.ndarray]:
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


def create_input_data(image_size: Tuple[int, int], background_logit: float,
                      logit_dictionary: Dict[str, Dict[str, float]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a 5-channel ground truth and prediction tensor for focal loss debugging.
    
    The channels correspond to:
    0: Background
    1: Circle (Top-Left quadrant)
    2: Square (Top-Right quadrant)
    3: Pentagon (Bottom-Left quadrant)
    4: Ellipse (Bottom-Right quadrant)
    
    Args:
        image_size (Tuple[int, int]): The total (height, width) of the image.
        background_logit (float): The default prediction logit for the background class.
        logit_dictionary (Dict[str, Dict[str, float]]): A dictionary specifying the prediction logits 
            for each foreground class. Must include nested dicts for 'circle', 'square', 'pentagon', 
            and 'ellipse' with 'left_logit' and 'right_logit' specified.
            
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The ground truth tensor and the prediction tensor, 
            both of shape (1, 5, height, width).
    """

    image_height, image_width = image_size
    quadrant_height, quadrant_width = image_height // 2, image_width // 2
    number_of_classes = 5
    batch_size = 1
    inactive_logit = -20.0

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
    print_non_zero_pixels(circle_ground_truth, "circle")

    ground_truth_tensor[0, 1, 0:quadrant_height, 0:quadrant_width] = torch.from_numpy(circle_ground_truth)
    prediction_tensor[0, 1, 0:quadrant_height, 0:quadrant_width] = torch.from_numpy(circle_prediction)

    # 2. Square (Top-Right)
    square_left = logit_dictionary["square"]["left_logit"]
    square_right = logit_dictionary["square"]["right_logit"]
    square_ground_truth, square_prediction = create_square_data(quadrant_width, quadrant_height, square_left,
                                                                square_right, inactive_logit)
    print_non_zero_pixels(square_ground_truth, "square")
    ground_truth_tensor[0, 2, 0:quadrant_height, quadrant_width:image_width] = torch.from_numpy(square_ground_truth)
    prediction_tensor[0, 2, 0:quadrant_height, quadrant_width:image_width] = torch.from_numpy(square_prediction)

    # 3. Pentagon (Bottom-Left)
    pentagon_left = logit_dictionary["pentagon"]["left_logit"]
    pentagon_right = logit_dictionary["pentagon"]["right_logit"]
    pentagon_ground_truth, pentagon_prediction = create_pentagon_data(quadrant_width, quadrant_height, pentagon_left,
                                                                      pentagon_right, inactive_logit)
    print_non_zero_pixels(pentagon_ground_truth, "pentagon")
    ground_truth_tensor[0, 3, quadrant_height:image_height, 0:quadrant_width] = torch.from_numpy(pentagon_ground_truth)
    prediction_tensor[0, 3, quadrant_height:image_height, 0:quadrant_width] = torch.from_numpy(pentagon_prediction)

    # 4. Ellipse (Bottom-Right)
    ellipse_left = logit_dictionary["ellipse"]["left_logit"]
    ellipse_right = logit_dictionary["ellipse"]["right_logit"]
    ellipse_ground_truth, ellipse_prediction = create_ellipse_data(quadrant_width, quadrant_height, ellipse_left,
                                                                   ellipse_right, inactive_logit)
    print_non_zero_pixels(ellipse_ground_truth, "ellipse")
    ground_truth_tensor[0, 4, quadrant_height:image_height, quadrant_width:image_width] = torch.from_numpy(
        ellipse_ground_truth)
    prediction_tensor[0, 4, quadrant_height:image_height, quadrant_width:image_width] = torch.from_numpy(
        ellipse_prediction)

    # Background Ground Truth (Where all other classes are 0)
    foreground_sum = torch.sum(ground_truth_tensor[0, 1:, :, :], dim=0)
    ground_truth_tensor[0, 0, :, :] = 1.0 - foreground_sum
    print_non_zero_pixels(ground_truth_tensor[0, 0, :, :].numpy(), "background")

    return ground_truth_tensor, prediction_tensor


def print_non_zero_pixels(numpy_array: np.ndarray, name: str) -> None:
    """
    Counts and prints the number of non-zero pixels in a given numpy array.
    
    Args:
        numpy_array (np.ndarray): The array or mask to be evaluated.
        name (str): The descriptive name of the object or class being evaluated.
    """
    non_zero_count = np.count_nonzero(numpy_array)
    print(f"We Have {non_zero_count} Non Zero Pixels for {name}")
