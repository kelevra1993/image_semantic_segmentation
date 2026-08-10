import os
import cv2
import json
import random
import kagglehub
import numpy as np

from tqdm import tqdm
from pathlib import Path

from app.utilities.os_utilities import make_directory, get_images, save_json
from app.utilities.data_utilities.data_utilities import preprocess
from shutil import rmtree, move


class DatasetRetriever:
    """
    A utility class to download, clean, and preprocess the Breast Ultrasound Images Dataset.
    """

    def __init__(self, dataset_directory: str, image_size: int, keep_aspect_ratio: bool) -> None:
        """
        Initializes the DatasetRetriever with directories and preprocessing configuration.

        Args:
            dataset_directory (str): Path to the root dataset directory. The raw data will be stored in a 'raw' subdirectory.
            image_size (int): The target size for the images (both width and height) during preprocessing.
            keep_aspect_ratio (bool): Whether to maintain the original aspect ratio of images by padding them.
        """
        self.dataset_directory = dataset_directory
        self.raw_dataset_directory = os.path.join(dataset_directory, "raw")
        self.image_size = image_size
        self.keep_aspect_ratio = keep_aspect_ratio

    def download_dataset(self) -> None:
        """
        Downloads the breast ultrasound images dataset from Kaggle and renames the root folder.

        This method will automatically create the parent directory if it does not exist. It uses a 
        temporary directory for the download process. If the dataset has not been downloaded already 
        (checked via a '.complete.txt' file), it downloads the dataset, cleans up unnecessary folders, 
        and renames the extracted 'Dataset_BUSI_with_GT' directory to the specified raw dataset 
        directory. Finally, it creates a '.complete.txt' file to prevent redundant downloads.
        """

        make_directory(self.dataset_directory)

        complete_file = os.path.join(self.raw_dataset_directory, ".complete.txt")
        if not os.path.exists(complete_file):

            # Create temporary dataset directory
            temporary_dataset_directory = os.path.join(self.dataset_directory, "temporary")
            make_directory(temporary_dataset_directory)

            path = kagglehub.dataset_download("aryashah2k/breast-ultrasound-images-dataset",
                                              output_dir=temporary_dataset_directory, force_download=True)

            # Clean up and rename directories
            complete_directory = os.path.join(temporary_dataset_directory, ".complete")

            if os.path.exists(complete_directory):
                rmtree(complete_directory)

                # Rename to a clean, easily readable folder name
                os.rename(src=os.path.join(temporary_dataset_directory, "Dataset_BUSI_with_GT"),
                          dst=self.raw_dataset_directory)

                with open(complete_file, "w") as output_file:
                    output_file.write("download completed")

                rmtree(temporary_dataset_directory)

        print(f"Dataset Was Saved Under file://{self.dataset_directory}")

    def clean_dataset(self) -> None:
        """
        Cleans the dataset by removing unwanted classes and flattening the directory structure.

        This method removes the 'normal' class folder entirely, and then moves all files from the
        'benign' and 'malignant' folders directly into the raw dataset directory. The now-empty 
        'benign' and 'malignant' folders are subsequently deleted.
        """
        normal_directory = os.path.join(self.raw_dataset_directory, "normal")
        benign_directory = os.path.join(self.raw_dataset_directory, "benign")
        malignant_directory = os.path.join(self.raw_dataset_directory, "malignant")

        # Remove images that do not have any segmentations
        if os.path.exists(normal_directory):
            rmtree(normal_directory)

        # Move images to a single directory
        for data_directory in [benign_directory, malignant_directory]:
            if os.path.exists(data_directory):
                for file_name in os.listdir(data_directory):
                    source_path = os.path.join(data_directory, file_name)
                    if os.path.isfile(source_path):
                        move(source_path, self.raw_dataset_directory)
                os.rmdir(data_directory)

    def preprocess_dataset(self) -> None:
        """
        Preprocesses the raw dataset images and masks, and saves them to a structured directory.

        This method constructs a target directory name based on the image size and whether the aspect 
        ratio is kept (e.g., 'preprocessed_512_KAR'). It iterates through all images in the raw dataset 
        directory, resizes them (with or without padding to keep aspect ratio), and saves the resulting 
        files into the designated preprocessing folder. PNG lossless compression is used to ensure mask 
        integrity is preserved.
        """
        # Create preprocessed directory name
        directory_name = f"preprocessed_{self.image_size}"
        directory_name += "_KAR" if self.keep_aspect_ratio else ""

        target_directory = os.path.join(self.dataset_directory, directory_name)
        make_directory(target_directory)

        if not os.path.exists(self.raw_dataset_directory):
            print(f"Directory {self.raw_dataset_directory} does not exist.")
            return

        for file_path in tqdm(get_images(self.raw_dataset_directory), desc="Preprocessing Dataset"):
            # Open image and preprocess it and save it.
            image = cv2.imread(file_path)
            processed_image = preprocess(image=image,
                                         image_size=self.image_size,
                                         keep_ratio=self.keep_aspect_ratio)

            output_path = os.path.join(target_directory, os.path.basename(file_path))

            # We use lossless compression since we want to keep our mask
            cv2.imwrite(filename=output_path,
                        img=processed_image,
                        params=[cv2.IMWRITE_PNG_COMPRESSION, 0])

        print(f"Processed Dataset Was Saved Under file://{target_directory}")

    def generate_dummy_masks(self) -> None:
        """
        Generates and saves the dummy masks for multi-class segmentation, and burns them into the images.

        This method iterates over all preprocessed images and creates three distinct, non-overlapping
        dummy masks for each image: a square, a letter B, and a plus sign. These dummy shapes are
        used to train the network on multi-class segmentation. The shapes are randomly placed, and
        we ensure they do not intersect with each other or the original masks. We also generate the
        objects with larger dimensions and line thickness to ensure they are easily segmentable. 
        Crucially, we draw these shapes directly onto the input image in white so the neural network
        can detect them.

        Returns:
            None
        """
        directory_name = f"preprocessed_{self.image_size}"
        directory_name += "_KAR" if self.keep_aspect_ratio else ""
        target_directory = os.path.join(self.dataset_directory, directory_name)

        all_files = get_images(target_directory, basename=True)
        # Filter out masks, only keep raw images
        images = [file_name for file_name in all_files if '_mask' not in file_name and not any(
            suffix in file_name for suffix in ['_object_b', '_object_plus', '_object_square'])]

        for image_name in tqdm(images, desc="Generating Dummy Masks & Updating Images"):
            base_name = os.path.splitext(image_name)[0]

            # Load the original image to burn shapes onto it
            image_path = os.path.join(target_directory, image_name)
            image_array = cv2.imread(image_path)

            # Load original masks to avoid overlap
            original_masks = [file_name for file_name in all_files if file_name.startswith(base_name + '_mask')]

            combined_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            for mask_name in original_masks:
                mask_path = os.path.join(target_directory, mask_name)
                mask_array = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                combined_mask = np.maximum(combined_mask, mask_array)

            masks = [combined_mask]

            for shape_type, suffix in [("square", "_object_square"), ("B", "_object_b"), ("+", "_object_plus")]:
                dummy_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
                attempts = 0
                while attempts < 100:
                    attempts += 1
                    temporary_dummy_mask = np.zeros_like(dummy_mask)

                    if shape_type == "square":
                        size = random.randint(50, 100)  # Increased size significantly
                        x_coordinate = random.randint(0, self.image_size - size - 1)
                        y_coordinate = random.randint(0, self.image_size - size - 1)
                        cv2.rectangle(temporary_dummy_mask, (x_coordinate, y_coordinate),
                                      (x_coordinate + size, y_coordinate + size), 255, -1)
                    elif shape_type == "B":
                        x_coordinate = random.randint(50, self.image_size - 80)
                        y_coordinate = random.randint(80, self.image_size - 50)
                        cv2.putText(temporary_dummy_mask, "B", (x_coordinate, y_coordinate), cv2.FONT_HERSHEY_SIMPLEX,
                                    3.0, 255, 6)  # Increased font scale and thickness
                    elif shape_type == "+":
                        x_coordinate = random.randint(50, self.image_size - 80)
                        y_coordinate = random.randint(80, self.image_size - 50)
                        cv2.putText(temporary_dummy_mask, "+", (x_coordinate, y_coordinate), cv2.FONT_HERSHEY_SIMPLEX,
                                    3.0, 255, 6)  # Increased font scale and thickness

                    # Combine all generated masks to check intersections
                    combined_existing_masks = np.zeros_like(dummy_mask)
                    for existing_mask in masks:
                        combined_existing_masks = np.maximum(combined_existing_masks, existing_mask)

                    intersection = np.logical_and(temporary_dummy_mask > 0, combined_existing_masks > 0)
                    if not np.any(intersection):
                        masks.append(temporary_dummy_mask)
                        output_path = os.path.join(target_directory, f"{base_name}{suffix}.png")
                        cv2.imwrite(output_path, temporary_dummy_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])

                        # Burn the shape into the input image (setting shape pixels to white 255)
                        image_array[temporary_dummy_mask > 0] = 255
                        break
                else:
                    # Fallback to empty if a spot is not found
                    masks.append(dummy_mask)
                    output_path = os.path.join(target_directory, f"{base_name}{suffix}.png")
                    cv2.imwrite(output_path, dummy_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            # Save the modified input image back to disk
            cv2.imwrite(image_path, image_array, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    def create_train_validation_test_split(self, train_ratio: float = 0.7,
                                           validation_ratio: float = 0.2) -> None:
        """
        Creates training, validation, and testing splits from the preprocessed dataset and saves them as JSON files.

        Each entry in the resulting JSON files is structured as a dictionary containing the image name
        and a list of its corresponding masks.

        Args:
            train_ratio (float, optional): The proportion of the dataset to include in the training split. Defaults to 0.7.
            validation_ratio (float, optional): The proportion of the dataset to include in the validation split. Defaults to 0.2.

        Returns:
            None
        """
        directory_name = f"preprocessed_{self.image_size}"
        directory_name += "_KAR" if self.keep_aspect_ratio else ""
        target_directory = os.path.join(self.dataset_directory, directory_name)

        if not os.path.exists(target_directory):
            print(f"Preprocessed directory {target_directory} does not exist.")
            return

        all_files = get_images(target_directory, basename=True)

        # Identify all images (those without '_mask' in their name)
        images = [file_name for file_name in all_files if '_mask' not in file_name and not any(
            suffix in file_name for suffix in ['_object_b', '_object_plus', '_object_square'])]

        dataset_entries = []
        for image_name in images:
            base_name = os.path.splitext(image_name)[0]

            # Find all corresponding masks for the current image
            masks = {
                "original": [file_name for file_name in all_files if file_name.startswith(base_name + '_mask')],
                "object_square": f"{base_name}_object_square.png",
                "object_b": f"{base_name}_object_b.png",
                "object_plus": f"{base_name}_object_plus.png"}

            dataset_entries.append({"image_name": image_name, "masks": masks})

        # Shuffle dataset entries to ensure random distribution
        # For reproducibility
        random.seed(42)
        random.shuffle(dataset_entries)

        total_length = len(dataset_entries)
        training_end = int(total_length * train_ratio)
        validation_end = training_end + int(total_length * validation_ratio)

        training_split = dataset_entries[:training_end]
        validation_split = dataset_entries[training_end:validation_end]
        testing_split = dataset_entries[validation_end:]

        for data_split, file_path in [(training_split, os.path.join(target_directory, "train_dataset.json")),
                                      (validation_split, os.path.join(target_directory, "validation_dataset.json")),
                                      (testing_split, os.path.join(target_directory, "test_dataset.json"))]:
            save_json(data_split, file_path=file_path)
            print(f"We Saved {len(data_split)} Elements In file://{file_path}")


# Download data under "ultrasound_dataset" folder at the project root.
project_directory = Path(__file__).parents[3]
dataset_directory = project_directory / "ultrasound_dataset"

# Settings for the dataset :
# Here the image size is set to 512 which is how the data will be preprocessed and prepared for training
dataset_retriever = DatasetRetriever(dataset_directory=str(dataset_directory),
                                     image_size=512,
                                     keep_aspect_ratio=True)

# Download dataset for kaggle dataset hub
dataset_retriever.download_dataset()

# Remove un-necessary folder and preprocess data by resizing images
dataset_retriever.clean_dataset()
dataset_retriever.preprocess_dataset()

# Generate additional mask dummy data such as letter B, a plus sign and a square
# This is done so that we can have a dataset with multiple different labels per image.
dataset_retriever.generate_dummy_masks()

# Create training and validation split for our data.
dataset_retriever.create_train_validation_test_split()
