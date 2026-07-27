import cv2
import torch
import random
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from torch.utils.data import Dataset, DataLoader
from utilities.os_utilities import read_json


class UltrasoundDataset(Dataset):
    """
    A PyTorch Dataset for loading Breast Ultrasound Images and their merged masks.
    """

    def __init__(self, json_path: str | Path, data_directory: str | Path, label_dictionary: Dict[str, int],
                 input_channels: int = 1, augment: bool = False) -> None:
        """
        Initializes the UltrasoundDataset.

        Args:
            json_path (str | Path): The path to the JSON file containing the dataset splits.
            data_directory (str | Path): The root directory where the images and masks are stored.
            input_channels (int, optional): The number of channels to load the image in (1 for grayscale, 3 for RGB).
             Defaults to 1.
            augment (bool, optional): Whether to apply data augmentation (left-right flipping). Defaults to False.
            
        Returns:
            None
        """
        self.data_directory = Path(data_directory)
        self.data_entries = read_json(str(json_path))
        self.input_channels = input_channels
        self.augment = augment
        self.label_dictionary = label_dictionary

    def __len__(self) -> int:
        """
        Returns the total number of entries in the dataset.

        Returns:
            int: The size of the dataset.
        """
        return len(self.data_entries)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves an image and its corresponding merged mask at the specified index.

        Args:
            index (int): The index of the dataset entry to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the image tensor and the mask tensor.
        """
        dataset_entry = self.data_entries[index]
        image_name = dataset_entry["image_name"]
        mask_names = dataset_entry["masks"]

        image_path = self.data_directory / image_name

        if self.input_channels == 1:
            image_array = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        else:
            image_array = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        if image_array is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")

        mask_list = [None] * len(self.label_dictionary)

        combined_original = None
        for mask_name in mask_names["original"]:
            mask_path = self.data_directory / mask_name
            mask_array = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if combined_original is None:
                combined_original = mask_array
            else:
                combined_original = np.maximum(combined_original, mask_array)
        if combined_original is None:
            combined_original = np.zeros(image_array.shape[:2], dtype=np.uint8)

        mask_list[self.label_dictionary["original"]] = combined_original

        for key in ["object_square", "object_b", "object_plus"]:
            mask_array = cv2.imread(str(self.data_directory / mask_names[key]), cv2.IMREAD_GRAYSCALE)
            mask_list[self.label_dictionary[key]] = mask_array

        combined_mask_array = np.stack(mask_list, axis=2)

        # Apply left-right flip augmentation if enabled
        if self.augment and random.random() > 0.5:
            image_array = cv2.flip(image_array, 1)
            combined_mask_array = cv2.flip(combined_mask_array, 1)

        if self.input_channels == 1:
            image_tensor = torch.from_numpy(image_array).float().unsqueeze(0) / 255.0
        else:
            # Transpose from (Height, Width, Channels) to (Channels, Height, Width)
            image_tensor = torch.from_numpy(image_array).float().permute(2, 0, 1) / 255.0

        # mask_tensor shape: (4, H, W)
        mask_tensor = torch.from_numpy(combined_mask_array).float().permute(2, 0, 1) / 255.0

        return image_tensor, mask_tensor


def get_dataloaders(preprocessed_directory: str | Path,
                    experiment_configuration: Dict[str, Any],
                    model_configuration: Dict[str, Any], batch_size: int = 8,
                    number_of_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    todo to be updated
    Creates DataLoaders for the training, validation, and testing splits.

    Args:
        preprocessed_directory (str | Path): The directory containing the preprocessed dataset and JSON splits.
        configuration (Dict[str, Any]): The loaded configuration dictionary, expected to contain the model parameters.
        batch_size (int, optional): The number of samples per batch. Defaults to 8.
        number_of_workers (int, optional): The number of subprocesses to use for data loading. Defaults to 4.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: A tuple containing the training, validation, and testing DataLoaders.
    """
    directory_path = Path(preprocessed_directory)

    # Retrieve input_channels from the configuration dictionary
    input_channels = model_configuration["input_channels"]
    label_dictionary = experiment_configuration["label_dictionary"]

    training_dataset = UltrasoundDataset(
        json_path=directory_path / "train_dataset.json",
        data_directory=directory_path,
        label_dictionary=label_dictionary,
        input_channels=input_channels,
        augment=True)

    validation_dataset = UltrasoundDataset(
        json_path=directory_path / "validation_dataset.json",
        data_directory=directory_path,
        label_dictionary=label_dictionary,
        input_channels=input_channels,
        augment=False)

    testing_dataset = UltrasoundDataset(
        json_path=directory_path / "test_dataset.json",
        data_directory=directory_path,
        label_dictionary=label_dictionary,
        input_channels=input_channels,
        augment=False)

    # DataLoader num_workers argument explicitly expects the 'num_workers' parameter name
    training_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=number_of_workers)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False,
                                   num_workers=number_of_workers)
    testing_loader = DataLoader(dataset=testing_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=number_of_workers)

    return training_loader, validation_loader, testing_loader
