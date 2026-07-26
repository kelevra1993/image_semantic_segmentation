import json
import os.path
import yaml
import torch
from pathlib import Path
from shutil import copyfile

from random import shuffle
from typing import Dict, Any, Tuple


def read_json(json_path: str) -> Dict[str, Any]:
    """
    Reads a JSON file from the specified path and returns its contents as a dictionary.

    This utility function is used across the project to load configuration or annotation data,
    such as the VGG Image Annotator output.

    Args:
        json_path (str): The absolute or relative path to the JSON file to read.

    Returns:
        Dict[str, Any]: The parsed data from the JSON file.
    """
    data = None

    with open(json_path) as f:
        data = json.load(f)

    return data


def save_json(data: Dict[str, Any], file_path: str | Path) -> None:
    """
    Saves a dictionary as a JSON file at the specified path.

    Args:
        data (Dict[str, Any]): The dictionary data to serialize.
        file_path (str | Path): The destination path for the JSON file.

    Returns:
        None
    """
    with open(file_path, 'w') as output_file:
        json.dump(data, output_file, indent=4)


def make_directory(directory_path: str | Path) -> None:
    """
    Creates a directory and any necessary parent directories.
    Does not raise an error if the directory already exists.

    Args:
        directory_path (str | Path): The path of the directory to create.

    Returns:
        None
    """
    os.makedirs(directory_path, exist_ok=True)


def get_images(path: str | Path, basename: bool = False, sort: bool = False, mix: bool = False,
               coherence: bool = False) -> list[str]:
    """
    Retrieves a list of image files from a specified directory.

    Args:
        path (str | Path): The directory path containing the images.
        basename (bool): If True, returns only the filenames instead of absolute paths. Defaults to False.
        sort (bool): If True, sorts the resulting list alphabetically. Defaults to False.
        mix (bool): If True, randomly shuffles the resulting list. Defaults to False.
        coherence (bool): If True, filters out files that are empty (0 bytes). Defaults to False.

    Returns:
        list[str]: A list of paths (or basenames) to the found image files.
    """
    if coherence:
        if basename:
            images = [file for file in os.listdir(path) if
                      file.endswith(('.jpg', '.png', '.jpeg', '.tiff')) and os.stat(
                          os.path.join(path, file)).st_size != 0]
        else:
            images = [os.path.join(path, file) for file in os.listdir(path) if
                      file.endswith(('.jpg', '.png', '.jpeg', '.tiff')) and os.stat(
                          os.path.join(path, file)).st_size != 0]
    else:
        if basename:
            images = [file for file in os.listdir(path) if
                      file.endswith(('.jpg', '.png', '.jpeg', '.tiff'))]
        else:
            images = [os.path.join(path, file) for file in os.listdir(path) if
                      file.endswith(('.jpg', '.png', '.jpeg', '.tiff'))]
    if mix:
        shuffle(images)

    if sort:
        images = sorted(images)

    return images


def print_blue(output: str, add_separators: bool = False) -> None:
    """
    Prints a string to the console in bold blue color.

    This utility is used throughout the project to highlight informational
    messages, status updates, and progress indicators during model training
    or data processing.

    Args:
        output (str): The string to be printed.
        add_separators (bool): If True, wraps the output with horizontal
            separators for better visibility.
    """
    if add_separators:
        length = max(len(line) for line in output.split("\n")) + 1
        print("\033[94m" + "\033[1m" + str(length * "-") + "\033[0m")
        print("\033[94m" + "\033[1m" + output + "\033[0m")
        print("\033[94m" + "\033[1m" + str(length * "-") + "\033[0m")
    else:
        print("\033[94m" + "\033[1m" + output + "\033[0m")


def print_green(output: str, add_separators: bool = False) -> None:
    """
    Prints a string to the console in bold green color.

    This utility is typically used to indicate successful operations, such
    as completed training iterations, saved model weights, or successful
    data extraction.

    Args:
        output (str): The string to be printed.
        add_separators (bool): If True, wraps the output with horizontal
            separators for better visibility.
    """
    if add_separators:
        length = max(len(line) for line in output.split("\n")) + 1
        print("\033[32m" + "\033[1m" + str(length * "-") + "\033[0m")
        print("\033[32m" + "\033[1m" + output + "\033[0m")
        print("\033[32m" + "\033[1m" + str(length * "-") + "\033[0m")
    else:
        print("\033[32m" + "\033[1m" + output + "\033[0m")


def print_yellow(output: str, add_separators: bool = False) -> None:
    """
    Prints a string to the console in bold yellow color.

    This utility is used for warnings or important notices that require
    user attention but are not necessarily critical failures (e.g., missing
    optional configuration fields).

    Args:
        output (str): The string to be printed.
        add_separators (bool): If True, wraps the output with horizontal
            separators for better visibility.
    """
    if add_separators:
        length = max(len(line) for line in output.split("\n")) + 1
        print("\033[93m" + "\033[1m" + str(length * "-") + "\033[0m")
        print("\033[93m" + "\033[1m" + output + "\033[0m")
        print("\033[93m" + "\033[1m" + str(length * "-") + "\033[0m")
    else:
        print("\033[93m" + "\033[1m" + output + "\033[0m")


def print_red(output: str, add_separators: bool = False) -> None:
    """
    Prints a string to the console in bold red color.

    This utility is reserved for error messages, critical failures, and
    exceptions that might halt the execution of the model or data pipeline.

    Args:
        output (str): The string to be printed.
        add_separators (bool): If True, wraps the output with horizontal
            separators for better visibility.
    """
    if add_separators:
        length = max(len(line) for line in output.split("\n")) + 1
        print("\033[91m" + "\033[1m" + str(length * "-") + "\033[0m")
        print("\033[91m" + "\033[1m" + output + "\033[0m")
        print("\033[91m" + "\033[1m" + str(length * "-") + "\033[0m")
    else:
        print("\033[91m" + "\033[1m" + output + "\033[0m")


def print_bold(output: str, add_separators: bool = False) -> None:
    """
    Prints a string to the console in bold font.

    This utility is used for general emphasis in console output, often for
    headers or key parameters in the experiment logs.

    Args:
        output (str): The string to be printed.
        add_separators (bool): If True, wraps the output with horizontal
            separators for better visibility.
    """
    if add_separators:
        length = max(len(line) for line in output.split("\n")) + 1
        print("\033[1m" + str(length * "-") + "\033[0m")
        print("\033[1m" + output + "\033[0m")
        print("\033[1m" + str(length * "-") + "\033[0m")
    else:
        print("\033[1m" + output + "\033[0m")


def print_dictionary(dictionary: Dict[str, Any], indent: int = 4) -> None:
    """
    Prints a dictionary to the console in a formatted JSON-like style.

    This utility is used to display configuration parameters, experiment
    summaries, or manifest data in a readable format during execution.

    Args:
        dictionary (Dict[str, Any]): The dictionary to be printed.
        indent (int): The number of spaces to use for indentation.
    """
    print(json.dumps(dictionary, indent=indent))


def load_configuration(configuration_path: str | Path) -> Dict[str, Any]:
    """
    Parses a YAML configuration file into a dictionary.

    Args:
        configuration_path (str | Path): The file path to the YAML configuration.

    Returns:
        Dict[str, Any]: The parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    path = Path(configuration_path)

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as file:
        try:
            configuration = yaml.safe_load(file)
            return configuration if configuration is not None else {}
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file at {path}:\n{e}")


def load_experiment_configuration(configuration_path: str | Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Loads a YAML configuration file, extracting the experiment-specific settings
    and leaving the remainder as model-specific settings.

    This function automatically casts certain keys to their correct Python types 
    (e.g., Paths, integers, floats, torch dtypes). It also creates the project 
    root directory and saves a backup of the configuration file inside it.

    Args:
        configuration_path (str | Path): The file path to the YAML configuration.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing two dictionaries:
            - experiment_configuration: Settings strictly related to training parameters.
            - model_configuration: Settings strictly related to the model architecture.

    Raises:
        KeyError: If the 'ExperimentConfiguration' key is missing from the YAML file.
    """
    configuration = load_configuration(configuration_path)
    if "ExperimentConfiguration" not in configuration:
        raise KeyError(f"Key 'ExperimentConfiguration' not found in {configuration_path}")

    experiment_configuration = configuration.pop("ExperimentConfiguration")
    model_configuration = configuration

    # Convert paths
    path_keys = ["data_folder", "dataset_folder"]

    for key in path_keys:
        if key in experiment_configuration:
            experiment_configuration[key] = Path(experiment_configuration[key])

    # Convert numerics
    integer_keys = ["information_dump", "weight_saving_iterations", "number_iterations"]
    for key in integer_keys:
        if key in experiment_configuration:
            experiment_configuration[key] = int(float(experiment_configuration[key]))

    if "learning_rate" in experiment_configuration:
        experiment_configuration["learning_rate"] = float(experiment_configuration["learning_rate"])

    # Convert dtype
    if "dtype" in experiment_configuration:
        dtype_map = {"float32": torch.float32, "float64": torch.float64}
        experiment_configuration["dtype"] = dtype_map.get(experiment_configuration["dtype"], torch.float32)

    # Set the project root and save a copy of the configuration file
    # Save a copy of the configuration file
    project_root = experiment_configuration["project_root"]
    project_root.mkdir(parents=True, exist_ok=True)
    copyfile(src=str(configuration_path), dst=str(project_root / "training_configuration.yaml"))

    return experiment_configuration, model_configuration
