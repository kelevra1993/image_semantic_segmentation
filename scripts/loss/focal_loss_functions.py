import torch
import cv2
import numpy as np
import math
import pandas as pd
from typing import Tuple, Dict, Any
from app.utilities.os_utilities import print_yellow

# Suppress scientific notation in Pandas output
pd.set_option('display.float_format', lambda x: f'{x:.5f}')


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
    pentagon_radius = min(quadrant_width, quadrant_height) // 4

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
                      logit_dictionary: Dict[str, Dict[str, float]], inactive_logit: float = -20.0,
                      number_of_classes: int = 5, batch_size: int = 1,
                      verbose: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Dict[str, Any]]]:
    """
    todo add verbose
    Generates synthetic ground truth masks and corresponding prediction logits for focal loss verification.
    
    This function creates a controlled 5-channel tensor environment (background + 4 geometric shapes) 
    that serves as the fundamental testing data for the downstream focal loss module. By systematically 
    splitting each object into a 'left' and 'right' logit region, it allows the training pipeline to 
    simulate varying degrees of class imbalance and prediction confidence. It also computes the exact 
    pixel coordinates for these regions, which are subsequently used by debugging dashboards to extract 
    and monitor specific loss metrics.
    
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
        inactive_logit (float): Logit value for pixels not belonging to a class. Defaults to -20.0.
        number_of_classes (int): Total number of output classes. Defaults to 5.
        batch_size (int): The batch size of the generated tensors. Defaults to 1.
            
    Returns:
        Tuple[torch.Tensor, torch.Tensor, Dict[str, Dict[str, Tuple[int, int]]]]: A tuple containing:
            - The ground truth tensor of shape (batch_size, number_of_classes, height, width).
            - The prediction tensor of shape (batch_size, number_of_classes, height, width).
            - A dictionary of spatial sampling positions and non-zero pixels for each object.
    """

    image_height, image_width = image_size
    quadrant_height, quadrant_width = image_height // 2, image_width // 2

    mid_quadrant_height = quadrant_height // 2
    one_third_quadrant_width = quadrant_width // 3
    two_thirds_quadrant_width = 2 * quadrant_width // 3

    object_information = {
        "circle": {
            "positions": {"left": (mid_quadrant_height, one_third_quadrant_width),
                          "right": (mid_quadrant_height, two_thirds_quadrant_width)}},
        "square": {
            "positions": {"left": (mid_quadrant_height, quadrant_width + one_third_quadrant_width),
                          "right": (mid_quadrant_height, quadrant_width + two_thirds_quadrant_width)}},
        "pentagon": {
            "positions": {"left": (quadrant_height + mid_quadrant_height, one_third_quadrant_width),
                          "right": (quadrant_height + mid_quadrant_height, two_thirds_quadrant_width)}},
        "ellipse": {
            "positions": {"left": (quadrant_height + mid_quadrant_height, quadrant_width + one_third_quadrant_width),
                          "right": (quadrant_height + mid_quadrant_height, quadrant_width + two_thirds_quadrant_width)}}
    }

    ground_truth_tensor = torch.zeros((batch_size, number_of_classes, image_height, image_width), dtype=torch.float32)
    prediction_tensor = torch.full(size=(batch_size, number_of_classes, image_height, image_width),
                                   fill_value=inactive_logit,
                                   dtype=torch.float32)

    # 0. Background
    prediction_tensor[0, 0, :, :] = background_logit

    # 1. Circle (Top-Left)
    circle_left = logit_dictionary["circle"]["left_logit"]
    circle_right = logit_dictionary["circle"]["right_logit"]
    circle_ground_truth, circle_prediction = create_circle_data(quadrant_width,
                                                                quadrant_height,
                                                                circle_left,
                                                                circle_right,
                                                                inactive_logit)
    object_information["circle"]["non_zero_pixels"] = compute_non_zero_pixels(numpy_array=circle_ground_truth,
                                                                              name="circle", verbose=verbose)

    ground_truth_tensor[0, 1, 0:quadrant_height, 0:quadrant_width] = torch.from_numpy(circle_ground_truth)
    prediction_tensor[0, 1, 0:quadrant_height, 0:quadrant_width] = torch.from_numpy(circle_prediction)

    # 2. Square (Top-Right)
    square_left = logit_dictionary["square"]["left_logit"]
    square_right = logit_dictionary["square"]["right_logit"]
    square_ground_truth, square_prediction = create_square_data(quadrant_width,
                                                                quadrant_height,
                                                                square_left,
                                                                square_right,
                                                                inactive_logit)
    object_information["square"]["non_zero_pixels"] = compute_non_zero_pixels(numpy_array=square_ground_truth,
                                                                              name="square", verbose=verbose)
    ground_truth_tensor[0, 2, 0:quadrant_height, quadrant_width:image_width] = torch.from_numpy(square_ground_truth)
    prediction_tensor[0, 2, 0:quadrant_height, quadrant_width:image_width] = torch.from_numpy(square_prediction)

    # 3. Pentagon (Bottom-Left)
    pentagon_left = logit_dictionary["pentagon"]["left_logit"]
    pentagon_right = logit_dictionary["pentagon"]["right_logit"]
    pentagon_ground_truth, pentagon_prediction = create_pentagon_data(quadrant_width,
                                                                      quadrant_height,
                                                                      pentagon_left,
                                                                      pentagon_right,
                                                                      inactive_logit)
    object_information["pentagon"]["non_zero_pixels"] = compute_non_zero_pixels(numpy_array=pentagon_ground_truth,
                                                                                name="pentagon", verbose=verbose)
    ground_truth_tensor[0, 3, quadrant_height:image_height, 0:quadrant_width] = torch.from_numpy(pentagon_ground_truth)
    prediction_tensor[0, 3, quadrant_height:image_height, 0:quadrant_width] = torch.from_numpy(pentagon_prediction)

    # 4. Ellipse (Bottom-Right)
    ellipse_left = logit_dictionary["ellipse"]["left_logit"]
    ellipse_right = logit_dictionary["ellipse"]["right_logit"]
    ellipse_ground_truth, ellipse_prediction = create_ellipse_data(quadrant_width,
                                                                   quadrant_height,
                                                                   ellipse_left,
                                                                   ellipse_right,
                                                                   inactive_logit)
    object_information["ellipse"]["non_zero_pixels"] = compute_non_zero_pixels(numpy_array=ellipse_ground_truth,
                                                                               name="ellipse", verbose=verbose)
    ground_truth_tensor[0, 4, quadrant_height:image_height, quadrant_width:image_width] = torch.from_numpy(
        ellipse_ground_truth)
    prediction_tensor[0, 4, quadrant_height:image_height, quadrant_width:image_width] = torch.from_numpy(
        ellipse_prediction)

    # Background Ground Truth (Where all other classes are 0)
    foreground_sum = torch.sum(ground_truth_tensor[0, 1:, :, :], dim=0)
    ground_truth_tensor[0, 0, :, :] = 1.0 - foreground_sum
    compute_non_zero_pixels(numpy_array=ground_truth_tensor[0, 0, :, :].numpy(), name="background", verbose=verbose)

    return ground_truth_tensor, prediction_tensor, object_information


def compute_non_zero_pixels(numpy_array: np.ndarray, name: str, verbose: bool = False) -> int:
    """
    Computes the exact number of non-zero pixels representing foreground objects within the synthetic masks.
    This serves as the normalization factor for the downstream focal loss calculation,
     ensuring proper loss scaling across variably sized geometries.
    
    Args:
        numpy_array (np.ndarray): The array or mask to be evaluated.
        name (str): The descriptive name of the object or class being evaluated.
        verbose (bool): If True, prints the count.
        
    Returns:
        int: The number of non-zero pixels.
    """
    non_zero_count = int(np.count_nonzero(numpy_array))
    if verbose:
        print(f"We Have {non_zero_count} Non Zero Pixels for {name}")
    return non_zero_count


def visualize_focal_loss(parameter_dictionary: Dict[str, Dict[str, float]], window_size: int, spacing_x: int,
                         spacing_y: int, focal_loss_class_image: torch.Tensor, ground_truth_tensor: torch.Tensor,
                         prediction_tensor: torch.Tensor,
                         object_information: Dict[str, Dict[str, Any]]) -> None:
    """
    Visualizes the focal loss components using OpenCV windows.

    This function isolates the visualization logic, creating a grid of OpenCV windows
    to display the model prediction, ground truth mask, and focal loss image for each
    class, scaled appropriately for display. It also plots green and red points on the
    model prediction image to indicate the sampling locations for the left and right sides.

    Args:
        parameter_dictionary (Dict[str, Dict[str, float]]): The dictionary of class logits.
        window_size (int): The display width and height for each window.
        spacing_x (int): Horizontal spacing between windows in pixels.
        spacing_y (int): Vertical spacing between windows in pixels.
        focal_loss_class_image (torch.Tensor): The normalized focal loss image tensor.
        ground_truth_tensor (torch.Tensor): The ground truth label tensor.
        prediction_tensor (torch.Tensor): The raw prediction logit tensor.
        object_information (Dict[str, Dict[str, Any]]): Pixel coordinates and metadata to sample the focal loss.
    """
    for index, object_class in enumerate(parameter_dictionary, start=1):
        index_model_predictions = torch.sigmoid(prediction_tensor[0, index])
        prediction_image = cv2.cvtColor(index_model_predictions.detach().cpu().numpy(), cv2.COLOR_GRAY2BGR)

        left_position = object_information[object_class]["positions"]["left"]
        right_position = object_information[object_class]["positions"]["right"]

        # Draw green point for left and red point for right (BGR format)
        cv2.circle(img=prediction_image, center=(left_position[1], left_position[0]), radius=3, color=(0.0, 1.0, 0.0),
                   thickness=-1)
        cv2.circle(img=prediction_image, center=(right_position[1], right_position[0]), radius=3, color=(0.0, 0.0, 1.0),
                   thickness=-1)

        # Calculate X and Y positions
        column_0_x = spacing_x
        column_1_x = column_0_x + window_size + spacing_x
        column_2_x = column_1_x + window_size + spacing_x
        row_y = spacing_y + (index - 1) * (window_size + spacing_y)

        # 1. Model Prediction Window
        prediction_name = f"Model Prediction - {object_class}"
        cv2.namedWindow(prediction_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(prediction_name, window_size, window_size)
        cv2.imshow(prediction_name, prediction_image)
        cv2.moveWindow(prediction_name, column_1_x, row_y)

        # 2. Ground Truth Window
        ground_truth_name = f"Ground Truth - {object_class}"
        cv2.namedWindow(ground_truth_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(ground_truth_name, window_size, window_size)
        cv2.imshow(ground_truth_name, ground_truth_tensor[0, index].detach().cpu().numpy())
        cv2.moveWindow(ground_truth_name, column_0_x, row_y)

        # 3. Focal Loss Window
        focal_loss_name = f"Focal Loss - {object_class}"
        cv2.namedWindow(focal_loss_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(focal_loss_name, window_size, window_size)
        cv2.imshow(focal_loss_name, focal_loss_class_image.detach().cpu().numpy())
        cv2.moveWindow(focal_loss_name, column_2_x, row_y)

    cv2.waitKey(0)


def compute_class_probabilities(parameter_dictionary: Dict[str, Dict[str, float]], background_logit: float,
                                inactive_logit: float = -20.0) -> Dict[str, Dict[str, float]]:
    """
    Computes the softmax probability for the left and right regions of each foreground class.

    This function calculates the exact probability of each class by considering the active 
    logit for the foreground class, the background logit, and the inactive logits for the 
    other classes. This gives an accurate representation of what the model predicts at 
    those specific pixel locations.

    Args:
        parameter_dictionary (Dict[str, Dict[str, float]]): The dictionary of class logits.
        background_logit (float): The logit value assigned to the background class.
        inactive_logit (float): The logit value assigned to inactive classes. Defaults to -20.0.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary containing the left and right probabilities 
            for each class.
    """
    probabilities = {}
    number_of_foreground_classes = len(parameter_dictionary)

    for object_class, info in parameter_dictionary.items():
        # Initialize logits with background and inactive foregrounds
        left_logits = [background_logit] + [inactive_logit] * number_of_foreground_classes
        right_logits = [background_logit] + [inactive_logit] * number_of_foreground_classes

        # Find index of this class (1-based because 0 is background)
        class_index = list(parameter_dictionary.keys()).index(object_class) + 1

        left_logits[class_index] = info["left_logit"]
        right_logits[class_index] = info["right_logit"]

        left_softmax = torch.softmax(torch.tensor(data=left_logits), dim=0)
        right_softmax = torch.softmax(torch.tensor(data=right_logits), dim=0)

        probabilities[object_class] = {
            "left_probability": left_softmax[class_index].item(),
            "right_probability": right_softmax[class_index].item()}

    return probabilities


def create_focal_loss_dataframe(parameter_dictionary: Dict[str, Dict[str, float]],
                                probabilities: Dict[str, Dict[str, float]],
                                focal_loss_image: torch.Tensor,
                                object_information: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Constructs a Pandas DataFrame aggregating regional probabilities alongside standard and focal losses.

    This function extracts loss metrics from the computed focal loss map and standard cross-entropy 
    formulation to provide a structured tabular format. This tabular data is consumed by downstream 
    visualization services and debugging dashboards to monitor class-imbalance mitigations 
    and verify that the focal loss correctly down-weights well-classified examples.
    
    Args:
        parameter_dictionary (Dict[str, Dict[str, float]]): The dictionary of class logits and alpha values.
        probabilities (Dict[str, Dict[str, float]]): The dictionary of class probabilities.
        focal_loss_image (torch.Tensor): The computed focal loss tensor.
        object_information (Dict[str, Dict[str, Any]]): Pixel coordinates and metadata to sample the focal loss.
            
    Returns:
        pd.DataFrame: A DataFrame containing the required loss metrics.
    """
    data_list = []

    for object_class, class_probabilities in probabilities.items():
        softmax_left = class_probabilities["left_probability"]
        softmax_right = class_probabilities["right_probability"]

        loss_left = -math.log(max(softmax_left, 1e-10))
        loss_right = -math.log(max(softmax_right, 1e-10))

        left_position = object_information[object_class]["positions"]["left"]
        right_position = object_information[object_class]["positions"]["right"]
        non_zero_pixels = object_information[object_class]["non_zero_pixels"]

        focal_loss_left = focal_loss_image[0, left_position[0], left_position[1]].item()
        focal_loss_right = focal_loss_image[0, right_position[0], right_position[1]].item()

        alpha_value = parameter_dictionary[object_class]["alpha"]

        data_list.append({"object_class": object_class,
                          "softmax_left": softmax_left,
                          "softmax_right": softmax_right,
                          "loss_left": loss_left,
                          "loss_right": loss_right,
                          "alpha": alpha_value,
                          "normalisation_factor": non_zero_pixels,
                          "focal_loss_left": focal_loss_left,
                          "focal_loss_right": focal_loss_right})

    focal_loss_dataframe = pd.DataFrame(data=data_list)

    return focal_loss_dataframe
