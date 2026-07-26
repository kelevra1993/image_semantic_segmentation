import os
import cv2
import kagglehub

from utilities.os_utilities import make_directory
from utilities.data_utilities.data_utilities import preprocess
from shutil import rmtree, move


class DatasetRetriever:

    def __init__(self, dataset_directory, preprocessed_dataset_directory, image_size, keep_aspect_ratio):
        self.dataset_directory = dataset_directory
        self.preprocessed_dataset_directory = preprocessed_dataset_directory
        self.image_size = image_size
        self.keep_aspect_ratio = keep_aspect_ratio

    def download_dataset(self):

        parent_directory = os.path.dirname(self.dataset_directory)
        make_directory(parent_directory)

        complete_file = os.path.join(parent_directory, ".complete.txt")
        if not os.path.exists(complete_file):
            path = kagglehub.dataset_download("aryashah2k/breast-ultrasound-images-dataset",
                                              output_dir=parent_directory, force_download=True)

        # Clean up and rename directories
        complete_directory = os.path.join(parent_directory, ".complete")
        if os.path.exists(complete_directory):
            rmtree(complete_directory)

            with open(complete_file, "w") as output_file:
                output_file.write("download completed")

            os.rename(os.path.join(parent_directory, "Dataset_BUSI_with_GT"), self.dataset_directory)

        print(f"Dataset Was Saved Under file://{self.dataset_directory}")

    def clean_dataset(self):
        normal_directory = os.path.join(self.dataset_directory, "normal")
        benign_directory = os.path.join(self.dataset_directory, "benign")
        malignant_directory = os.path.join(self.dataset_directory, "malignant")

        if os.path.exists(normal_directory):
            rmtree(normal_directory)

        for data_directory in [benign_directory, malignant_directory]:
            if os.path.exists(data_directory):
                for file_name in os.listdir(data_directory):
                    source_path = os.path.join(data_directory, file_name)
                    if os.path.isfile(source_path):
                        move(source_path, self.dataset_directory)
                os.rmdir(data_directory)

    def preprocess_dataset(self):

        # Create preprocessed directory name
        directory_name = f"preprocessed_{self.image_size}_KAR" if self.keep_aspect_ratio else f"preprocessed_{self.image_size}"
        target_directory = os.path.join(self.preprocessed_dataset_directory, directory_name)
        make_directory(target_directory)

        if not os.path.exists(self.dataset_directory):
            print(f"Directory {self.dataset_directory} does not exist.")
            return

        for file_name in os.listdir(self.dataset_directory):

            file_path = os.path.join(self.dataset_directory, file_name)

            if os.path.isfile(file_path):
                # Open image and preprocess it and save it.
                image = cv2.imread(file_path)
                processed_image = preprocess(image=image,
                                             image_size=self.image_size,
                                             keep_ratio=self.keep_aspect_ratio)

                output_path = os.path.join(target_directory, file_name)

                # We use lossless compression since we want to keep our mask
                cv2.imwrite(filename=output_path, img=processed_image, params=[cv2.IMWRITE_PNG_COMPRESSION, 0])


parent_directory = os.path.dirname(os.getcwd())
dataset_directory = os.path.join(parent_directory, "ultrasound_dataset")

raw_dataset_directory = os.path.join(dataset_directory, "raw")
processed_dataset_directory = os.path.join(dataset_directory, "processed")

dataset_retriever = DatasetRetriever(dataset_directory=raw_dataset_directory,
                                     preprocessed_dataset_directory=processed_dataset_directory,
                                     image_size=512,
                                     keep_aspect_ratio=True)

dataset_retriever.download_dataset()
dataset_retriever.clean_dataset()
dataset_retriever.preprocess_dataset()
