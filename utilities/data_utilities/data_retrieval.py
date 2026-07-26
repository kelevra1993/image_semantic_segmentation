import os
import cv2
import kagglehub
from tqdm import tqdm

from utilities.os_utilities import make_directory, get_images
from utilities.data_utilities.data_utilities import preprocess
from shutil import rmtree, move


class DatasetRetriever:
    """
    A utility class to download, clean, and preprocess the Breast Ultrasound Images Dataset.
    """

    def __init__(self, dataset_directory: str, image_size: int, keep_aspect_ratio: bool) -> None:
        """
        Initializes the DatasetRetriever with directories and preprocessing configuration.

        Args:
            dataset_directory (str): Path to dataset directory.
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

        This method will automatically create the parent directory if it does not exist, download
        the dataset, clean up unnecessary folders, and place the data in the specified raw 
        dataset directory.
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
        'benign' and 'malignant' folders directly into the root dataset directory. The now-empty 
        'benign' and 'malignant' folders are subsequently deleted.
        """
        normal_directory = os.path.join(self.raw_dataset_directory, "normal")
        benign_directory = os.path.join(self.raw_dataset_directory, "benign")
        malignant_directory = os.path.join(self.raw_dataset_directory, "malignant")

        if os.path.exists(normal_directory):
            rmtree(normal_directory)

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

        This method iterates through all images in the raw dataset directory, resizes them (with
        or without padding to keep aspect ratio), and saves the resulting files into a designated 
        preprocessing folder. PNG lossless compression is used to ensure mask integrity is preserved.
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


parent_directory = os.path.dirname(os.getcwd())
dataset_directory = os.path.join(parent_directory, "ultrasound_dataset")

dataset_retriever = DatasetRetriever(dataset_directory=dataset_directory,
                                     image_size=512,
                                     keep_aspect_ratio=True)

dataset_retriever.download_dataset()
dataset_retriever.clean_dataset()
dataset_retriever.preprocess_dataset()
