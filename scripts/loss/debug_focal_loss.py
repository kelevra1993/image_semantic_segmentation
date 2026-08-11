import cv2
import os

import pandas as pd
import torch
from typing import Dict, Tuple, List
from app.loss.focal_loss import FocalLoss
from scripts.loss.focal_loss_functions import (create_input_data,
                                               export_focal_loss_visualization,
                                               compute_class_probabilities,
                                               create_focal_loss_dataframe,
                                               save_experimental_data)
from app.utilities.os_utilities import print_yellow
from app.utilities.tensor_utilities import print_tensor_status

# Suppress Qt C++ warnings (like QFontDatabase missing fonts)
os.environ["QT_LOGGING_RULES"] = "*=false"


def debug_focal_loss(parameter_dictionary: Dict[str, Dict[str, float]],
                     image_size: Tuple[int, int],
                     background_logit: float,
                     inactive_logit: float,
                     number_classes: int,
                     batch_size: int,
                     alpha: List[float],
                     gamma: float,
                     save_experiment: bool = True,
                     experiment_folder: str = "") -> pd.DataFrame:
    """
    Creates a 5-class synthetic testing scenario to visually verify the focal loss implementation.

    This function utilizes the modular data generators from focal_loss_functions.py to create
    a full 5-channel image containing a circle, square, pentagon, and ellipse in each quadrant.
    It passes the generated tensors to the FocalLoss module to review behavior under different
    logit configurations.
    
    Args:
        parameter_dictionary (Dict[str, Dict[str, float]]): The dictionary of class logits.
        image_size (Tuple[int, int]): The total (height, width) of the image.
        background_logit (float): The default prediction logit for the background class.
        inactive_logit (float): Logit value for pixels not belonging to a class.
        number_classes (int): Total number of output classes.
        batch_size (int): The batch size of the generated tensors.
        alpha (List[float]): List of alpha weighting factors for each class.
        gamma (float): Focusing parameter for the focal loss.
        save_experiment (bool): If True, saves the visualization image, parameters, and dataframe to disk. Defaults to True.
        experiment_folder (str): The folder path where the outputs should be saved.
        
    Returns:
        pd.DataFrame: The generated focal loss metrics dataframe.
    """
    # Generate Data containing our object divided into two regions.
    ground_truth_tensor, prediction_tensor, object_information = create_input_data(
        image_size=image_size,
        background_logit=background_logit,
        logit_dictionary=parameter_dictionary,
        inactive_logit=inactive_logit,
        number_of_classes=number_classes,
        batch_size=batch_size)

    # Initialize FocalLoss
    device = torch.device('cpu')

    # Create Focal Loss Object And Compute The Loss
    focal_loss = FocalLoss(alpha=alpha, gamma=gamma, device=device, dtype=torch.float32)
    loss, focal_loss_image = focal_loss(prediction_tensor,
                                        ground_truth_tensor)

    # Globally min-max normalize the image
    focal_min, focal_max = focal_loss_image[0].min(), focal_loss_image[0].max()
    focal_loss_class_image = (focal_loss_image[0] - focal_min) / (focal_max - focal_min + 1e-8)

    # Compute predictions softmaxes
    probabilities = compute_class_probabilities(parameter_dictionary=parameter_dictionary,
                                                background_logit=background_logit,
                                                inactive_logit=inactive_logit)

    # Generate and print dataframe
    focal_loss_dataframe = create_focal_loss_dataframe(parameter_dictionary=parameter_dictionary,
                                                       probabilities=probabilities,
                                                       focal_loss_image=focal_loss_image,
                                                       object_information=object_information,
                                                       gamma=gamma)

    # Get visualization Of Focal Loss
    export_focal_loss_visualization(parameter_dictionary=parameter_dictionary,
                                    focal_loss_class_image=focal_loss_class_image,
                                    ground_truth_tensor=ground_truth_tensor,
                                    prediction_tensor=prediction_tensor,
                                    object_information=object_information,
                                    save_visualization=save_experiment,
                                    experiment_folder=experiment_folder)

    if save_experiment:
        # Save the parameter_dictionary as well as the dataframe
        save_experimental_data(parameter_dictionary=parameter_dictionary,
                               focal_loss_dataframe=focal_loss_dataframe,
                               experiment_folder=experiment_folder)

    return focal_loss_dataframe


if __name__ == "__main__":
    # Constant Values Across All Experiments That Never Change
    test_image_size = (200, 200)
    test_number_classes = 5
    test_batch_size = 1

    # Additional constant values accross experiments but might change
    test_background_logit = 0.8
    test_inactive_logit = -10.0

    # Experiment parameters
    # Definition of logits for left and right part of our masked objects as well as alphas and gammas
    test_parameter_dictionary = {
        "circle": {"left_logit": 2.8, "right_logit": 1.0, "alpha": 1.0},
        "square": {"left_logit": 2.8, "right_logit": 1.0, "alpha": 1.0},
        "pentagon": {"left_logit": 1.0, "right_logit": 5.0, "alpha": 1.0},
        "ellipse": {"left_logit": 2.8, "right_logit": 1.0, "alpha": 1.0}}

    experimental_parameters = {
        "1": {"alpha": {"circle": 1.0, "square": 1.0, "pentagon": 1.0, "ellipse": 1.0}, "gamma": 0.0},
        "2": {"alpha": {"circle": 1.0, "square": 2.0, "pentagon": 1.0, "ellipse": 1.0}, "gamma": 0.0},
        "3": {"alpha": {"circle": 1.0, "square": 1.0, "pentagon": 1.0, "ellipse": 1.0}, "gamma": 0.5},
        "4": {"alpha": {"circle": 1.0, "square": 1.0, "pentagon": 1.0, "ellipse": 1.0}, "gamma": 0.75},
        "5": {"alpha": {"circle": 1.0, "square": 1.0, "pentagon": 1.0, "ellipse": 1.0}, "gamma": 1.0},
        "6": {"alpha": {"circle": 1.0, "square": 1.0, "pentagon": 1.0, "ellipse": 1.0}, "gamma": 2.0},
        "7": {"alpha": {"circle": 1.0, "square": 1.0, "pentagon": 1.0, "ellipse": 1.0}, "gamma": 4.0},
    }
    # Dataframe that will contain all the different experiments that will be ran.
    experiments_dataframe = pd.DataFrame()
    main_experimental_folder = "experiments"

    for experiment_name, experiment_information in experimental_parameters.items():
        # Update test_parameter_dictionary with alphas for this experiment
        for shape, alpha_value in experiment_information["alpha"].items():
            test_parameter_dictionary[shape]["alpha"] = alpha_value

        # get alphas (background will always be 1.0)
        test_alpha = [1.0] + [test_parameter_dictionary[key]["alpha"] for key in test_parameter_dictionary]
        test_gamma = experiment_information["gamma"]

        # Run focal loss debugger and visualizer
        experiment_dataframe = debug_focal_loss(parameter_dictionary=test_parameter_dictionary,
                                                image_size=test_image_size,
                                                background_logit=test_background_logit,
                                                inactive_logit=test_inactive_logit,
                                                number_classes=test_number_classes,
                                                batch_size=test_batch_size,
                                                alpha=test_alpha, gamma=test_gamma,
                                                save_experiment=True,
                                                experiment_folder=os.path.join(main_experimental_folder,
                                                                               experiment_name))

        # Add the experiment identifier as the first column
        experiment_dataframe.insert(loc=0, column="experiment", value=experiment_name)
        experiments_dataframe = pd.concat(objs=[experiments_dataframe, experiment_dataframe], ignore_index=True)

    print_yellow("All experiments finished.")
    print(experiments_dataframe.to_string(justify='center'))

    # Save all experiments into one pandas dataframe.
    experiments_csv_path = os.path.join(main_experimental_folder, "focal_loss_experiments.csv")
    experiments_dataframe.to_csv(path_or_buf=experiments_csv_path, index=False)
