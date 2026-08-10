import torchvision
from app.loss.bce_loss import BCELoss
import os
import csv
import time
import numpy as np

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from app.loss.focal_loss import FocalLoss
from utilities.os_utilities import load_configuration, print_red, print_green, print_blue, print_yellow
from utilities.tensor_utilities import get_device, print_tensor_status, print_tensor_list, print_tensor_shape
from app.model.model import UnetModel
from utilities.data_utilities.dataloader import get_dataloaders
from torch.utils.data.dataloader import _BaseDataLoaderIter

from torch.utils.data import DataLoader


class Trainer:
    """
    The Trainer class orchestrates the entire training lifecycle of the U-Net model.
    It manages data loading, model initialization, optimization, validation, checkpointing,
    and logging to TensorBoard.
    """

    def __init__(self, experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
        """
        Initializes the Trainer with all necessary components for a training run.

        Args:
            experiment_configuration (Dict[str, Any]): Dictionary containing the core training parameters.
            model_configuration (Dict[str, Any]): Dictionary containing the model architecture parameters.
        """
        self.experiment_configuration = experiment_configuration
        self.model_configuration = model_configuration["UnetConfiguration"]

        self.device = get_device()
        self.dtype = self.experiment_configuration["dtype"]

        print(f"For Training We Will Be Using device: {self.device}")

        # Basic training parameters (strictly enforced)
        self.project_root = self.experiment_configuration["project_root"]
        self.training_iterations = self.experiment_configuration["number_iterations"]
        self.weight_saving_iterations = self.experiment_configuration["weight_saving_iterations"]
        self.information_dump = self.experiment_configuration["information_dump"]
        self.learning_rate = self.experiment_configuration["learning_rate"]

        self.batch_size = self.experiment_configuration["batch_size"]
        self.compute_validation_iteration = self.experiment_configuration["compute_validation_iteration"]
        self.resume_training = self.experiment_configuration["resume_training"]

        # Setup paths and tensorboard
        self.tensorboard_directory, self.weights_directory = self.setup_training_paths()
        self.training_writer, self.validation_writer = self.setup_tensorboard_writers()

        # Setting up dataloaders
        self.dataset_folder = self.experiment_configuration["dataset_folder"]
        self.train_dataloader, self.validation_dataloader, self.test_dataloader = self.get_trainer_data_loaders()

        # Initialize Model (U-Net) and Optimizer
        self.model = UnetModel(unet_configuration=self.model_configuration)
        self.model.to(device=self.device, dtype=self.dtype)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # TODO Will be modified
        # we will compute all losses but we will only optimise some
        self.criterion = BCELoss()
        self.focal_loss = FocalLoss(alpha=self.experiment_configuration["alpha"],
                                    gamma=self.experiment_configuration["gamma"],
                                    device=self.device,
                                    dtype=self.dtype)

        # Restoration logic
        self.start_iteration = 1
        if self.resume_training:
            self.start_iteration = self.restore_last_model()

        # Metric tracking
        self.tracked_metrics_mapping = {"total_loss": "Total BCE Loss"}
        for label_name, index in self.experiment_configuration["label_dictionary"].items():
            self.tracked_metrics_mapping[f"iou_{label_name}"] = f"IoU {label_name.capitalize()}"

    def setup_training_paths(self) -> tuple[Path, Path]:
        """
        Sets up the directory structure and persistent files for training outputs.

        This method acts as the initialization step for experiment storage, ensuring that
        the `Tensorboard` directory, the `Weights` directory, and the `metrics_evolution.csv`
        file are properly initialized within the global `project_root` before the training
        loop commences.

        Returns:
            tuple[Path, Path, Path]: A tuple containing:
                - tensorboard_directory (Path): Path to the TensorBoard logs folder.
                - weights_directory (Path): Path to the saved model weights folder.
        """
        tensorboard_directory = self.project_root / "Tensorboard"
        weights_directory = self.project_root / "Weights"

        # Create directories
        tensorboard_directory.mkdir(exist_ok=True, parents=True)
        weights_directory.mkdir(exist_ok=True, parents=True)

        return tensorboard_directory, weights_directory

    def setup_tensorboard_writers(self) -> Tuple[SummaryWriter, Optional[SummaryWriter]]:
        """
        Initializes TensorBoard SummaryWriters for logging training and validation metrics.

        Returns:
            Tuple[SummaryWriter, Optional[SummaryWriter]]: A tuple containing the training writer
            and the validation writer (if validation is enabled).
        """
        training_writer = SummaryWriter(log_dir=str(self.tensorboard_directory / "Train"))
        if self.compute_validation_iteration:
            validation_writer = SummaryWriter(log_dir=str(self.tensorboard_directory / "Validation"))
        else:
            validation_writer = None

        return training_writer, validation_writer

    def get_trainer_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Initializes and returns the PyTorch DataLoaders for the training, validation, and test phases.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: A tuple containing the DataLoaders for
                the training, validation, and testing splits, respectively.
        """
        print_blue("Initializing U-Net DataLoaders...", add_separators=True)

        train_dataloader, validation_dataloader, test_dataloader = get_dataloaders(
            preprocessed_directory=self.dataset_folder,
            experiment_configuration=self.experiment_configuration,
            model_configuration=self.model_configuration,
            batch_size=self.batch_size, number_of_workers=4)

        return train_dataloader, validation_dataloader, test_dataloader

    @staticmethod
    def get_next_batch(dataloader_iterator: _BaseDataLoaderIter, dataloader: DataLoader) -> Tuple[
        torch.Tensor, torch.Tensor, _BaseDataLoaderIter]:
        """
        Retrieves the next batch of images and masks from the dataloader iterator.
        
        If the iterator is exhausted, it re-initializes it from the dataloader.
        If a FileNotFoundError is encountered during data loading, it continues attempting
        to fetch the next available batch, up to a maximum of 10 consecutive failures.
        
        Args:
            dataloader_iterator (_BaseDataLoaderIter): The current iterator for the dataloader.
            dataloader (DataLoader): The original dataloader object to reset the iterator.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, iter]: A tuple containing the batch of images, 
                the batch of masks, and the potentially refreshed iterator.
                
        Raises:
            SystemExit: If more than 10 consecutive FileNotFoundError occur.
        """
        missing_files_count = 0
        while True:
            try:
                images, masks = next(dataloader_iterator)
                break
            except StopIteration:
                dataloader_iterator = iter(dataloader)
            except FileNotFoundError:
                missing_files_count += 1
                if missing_files_count > 10:
                    print_red("Critical Error: Over 10 consecutive missing files encountered. Exiting.",
                              add_separators=True)
                    exit(1)
                continue

        return images, masks, dataloader_iterator

    def run_benchmarking_loop(self, benchmarking_iterations: int = 1e5) -> None:
        """
        Todo document function
        """

        # Get dataloader
        training_dataloader_iterator = iter(self.train_dataloader)
        validation_dataloader_iterator = iter(self.validation_dataloader)

        for training_iteration in tqdm(range(int(benchmarking_iterations)), desc="Benchmarking Loop"):

            # Get next training elements
            training_images, training_masks, training_dataloader_iterator = self.get_next_batch(
                dataloader_iterator=training_dataloader_iterator, dataloader=self.train_dataloader)

            self.model.train()
            self.optimizer.zero_grad()

            # Forward pass
            training_loss, model_outputs = self.run_model_iteration(batch_images=training_images, batch_masks=training_masks,
                                                        writer=self.training_writer, iteration=training_iteration,
                                                        tracker_dictionary=None)

            # TODO Still in progress
            # Testing of the focal loss
            self.focal_loss.forward(model_predictions=model_outputs, ground_truths=training_masks)

            # Backward and Step
            training_loss.backward()
            self.optimizer.step()

            # Validation phase
            if self.compute_validation_iteration:
                # Get next validation elements
                validation_images, validation_masks, validation_dataloader_iterator = self.get_next_batch(
                    dataloader_iterator=validation_dataloader_iterator, dataloader=self.validation_dataloader)

                self.model.eval()
                with torch.no_grad():
                    _, _ = self.run_model_iteration(batch_images=validation_images, batch_masks=validation_masks,
                                                    writer=self.validation_writer, iteration=training_iteration,
                                                    tracker_dictionary=None)
                    exit()

    def run_training_loop(self) -> None:
        """
        Executes the main training loop for the U-Net model.
        """
        # Get dataloader
        training_dataloader_iterator = iter(self.train_dataloader)
        validation_dataloader_iterator = iter(self.validation_dataloader)

        # Get Metric Trackers
        # Currently just BCE Loss
        training_trackers = self.get_metric_trackers()
        validation_trackers = self.get_metric_trackers() if self.compute_validation_iteration else None

        # Get current training iteration
        training_iteration = self.start_iteration

        try:
            for training_iteration in range(self.start_iteration, self.training_iterations + self.start_iteration, 1):
                if training_iteration % self.weight_saving_iterations == 0:
                    self.save_model(iteration=training_iteration)
                    self.run_test_evaluation(iteration=training_iteration)

                # Get next training elements
                training_images, training_masks, training_dataloader_iterator = self.get_next_batch(
                    dataloader_iterator=training_dataloader_iterator, dataloader=self.train_dataloader)

                self.model.train()
                self.optimizer.zero_grad()

                # Forward pass
                training_loss, _ = self.run_model_iteration(
                    batch_images=training_images,
                    batch_masks=training_masks,
                    writer=self.training_writer,
                    iteration=training_iteration,
                    tracker_dictionary=training_trackers)

                # Backward and Step
                training_loss.backward()
                self.optimizer.step()

                # Validation phase
                if self.compute_validation_iteration:
                    self.model.eval()
                    with torch.no_grad():
                        validation_images, validation_masks, validation_dataloader_iterator = self.get_next_batch(
                            dataloader_iterator=validation_dataloader_iterator, dataloader=self.validation_dataloader)

                        _, _ = self.run_model_iteration(
                            batch_images=validation_images,
                            batch_masks=validation_masks,
                            writer=self.validation_writer,
                            iteration=training_iteration,
                            tracker_dictionary=validation_trackers)

                # Console log dump
                if training_iteration % self.information_dump == 0:
                    training_trackers = self.console_log_update_tracker(
                        iterations=training_iteration,
                        training_tracker_dictionary=training_trackers,
                        validation_tracker_dictionary=validation_trackers)

                    if self.compute_validation_iteration:
                        validation_trackers = self.get_metric_trackers()

        except KeyboardInterrupt:
            print_red(f"\nTraining Interrupted by User at iteration {training_iteration}.", add_separators=True)
        finally:
            self.save_model(iteration=training_iteration)
            print_green(f"Model successfully saved at iteration {training_iteration}. Exiting Training.",
                        add_separators=True)
            self.training_writer.close()
            if self.compute_validation_iteration:
                self.validation_writer.close()

    def run_test_evaluation(self, iteration: int) -> None:
        """
        Performs a full evaluation on the test dataset and logs results to a file.

        Args:
            iteration (int): The current training iteration index.
        """
        print(f"Starting Full Test Evaluation at Iteration {iteration}...")
        self.model.eval()

        total_test_loss = 0.0
        number_batches = len(self.test_dataloader)

        with torch.no_grad():
            for test_images, test_masks in tqdm(self.test_dataloader, total=number_batches,
                                                desc=f"Test Evaluation Iteration {iteration}"):
                test_images = test_images.to(device=self.device, dtype=self.dtype)
                test_masks = test_masks.to(device=self.device, dtype=self.dtype)

                model_outputs = self.model(test_images)
                loss = self.criterion(predictions=model_outputs, ground_truth=test_masks)

                total_test_loss += loss.item()

        # Calculate mean
        mean_test_loss = total_test_loss / number_batches if number_batches > 0 else 0.0

        # Log to file
        evaluation_file = self.project_root / "test_evaluation_results.txt"
        with open(evaluation_file, "a") as f:
            f.write("+" + "-" * 50 + "\n")
            f.write(f"| Iteration: {iteration:<38} |\n")
            f.write("+" + "-" * 50 + "\n")
            f.write(f"| {'Total BCE Loss':<35} : {mean_test_loss:<8.4f} |\n")
            f.write("+" + "-" * 50 + "\n\n")

        print(f"Full Test Evaluation Completed. Results appended to {evaluation_file}")

        # Run sample predictions for visualization
        self.run_sample_predictions(iteration=iteration, number_samples=20)

    def run_sample_predictions(self, iteration: int, number_samples: int = 20) -> None:
        """
        Runs inference on a subset of the test dataset and saves the predicted
        segmentation masks as PNG images for visual inspection.

        Args:
            iteration (int): The current training iteration index.
            number_samples (int): The maximum number of test samples to process.
        """
        print(f"Running {number_samples} Sample Predictions for Iteration {iteration}...")

        output_directory = self.weights_directory / f"Iteration_{iteration}" / "test_sample_predictions"
        output_directory.mkdir(exist_ok=True, parents=True)

        self.model.eval()
        samples_processed = 0
        test_dataloader_iterator = iter(self.test_dataloader)

        with torch.no_grad():
            while samples_processed < number_samples:
                try:
                    test_images, test_masks = next(test_dataloader_iterator)
                except StopIteration:
                    break

                test_images = test_images.to(device=self.device, dtype=self.dtype)
                test_masks = test_masks.to(device=self.device, dtype=self.dtype)

                model_outputs = self.model(test_images)
                predicted_masks = (torch.sigmoid(model_outputs) > 0.5).to(self.dtype)

                batch_size_current = test_images.size(0)
                for i in range(batch_size_current):

                    if samples_processed >= number_samples:
                        break

                    img_to_add = test_images[i:i + 1]  # (1, C, H, W)
                    if img_to_add.shape[1] == 3:
                        img_to_add = img_to_add.mean(dim=1, keepdim=True)

                    components = [img_to_add]
                    for c in range(test_masks.shape[1]):
                        components.append(test_masks[i:i + 1, c:c + 1, :, :])
                    for c in range(predicted_masks.shape[1]):
                        components.append(predicted_masks[i:i + 1, c:c + 1, :, :])

                    comparison_grid = torch.cat(components, dim=0)

                    output_path = output_directory / f"sample_{samples_processed:04d}.png"
                    torchvision.utils.save_image(comparison_grid, output_path, nrow=5)

                    samples_processed += 1

        print_green(f"Successfully saved {samples_processed} sample predictions to {output_directory}")

    def intersection_over_union_per_class(self, predictions: torch.Tensor, targets: torch.Tensor,
                                          smooth: float = 1e-6) -> dict[str, torch.Tensor]:
        """
        Computes the Intersection over Union (IoU) for each class independently.

        Args:
            predictions (torch.Tensor): The model's raw logits output of shape (B, C, H, W).
            targets (torch.Tensor): The ground truth mask tensor of shape (B, C, H, W).
            smooth (float, optional): A small constant to avoid division by zero. Defaults to 1e-6.

        Returns:
            dict[str, torch.Tensor]: A dictionary mapping class IoU keys to their calculated tensor values.
        """
        predicted_masks = (torch.sigmoid(predictions) > 0.5).to(self.dtype)
        ious = {}

        for label_name, index in self.experiment_configuration["label_dictionary"].items():
            predicted_class_masks = predicted_masks[:, index, :, :]
            target_class_masks = targets[:, index, :, :]

            intersection = (predicted_class_masks * target_class_masks).sum(dim=(1, 2))
            union = predicted_class_masks.sum(dim=(1, 2)) + target_class_masks.sum(dim=(1, 2)) - intersection

            iou = (intersection + smooth) / (union + smooth)
            ious[f"iou_{label_name}"] = iou.mean()

        return ious

    def run_model_iteration(self, batch_images: torch.Tensor, batch_masks: torch.Tensor,
                            writer: SummaryWriter, iteration: int,
                            tracker_dictionary: dict | None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Executes a single forward pass of the U-Net, computes the BCE loss,
        and updates performance trackers.

        Args:
            batch_images (torch.Tensor): The input image tensor of shape (B, C, H, W).
            batch_masks (torch.Tensor): The ground truth mask tensor of shape (B, 1, H, W).
            writer (SummaryWriter): TensorBoard writer for logging.
            iteration (int): The current training iteration step.
            tracker_dictionary (Dict[str, Any] | None): Dictionary tracking rolling average metrics.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the total_loss and the model predictions.
        """
        # Move to target device/dtype
        batch_images = batch_images.to(device=self.device, dtype=self.dtype)
        batch_masks = batch_masks.to(device=self.device, dtype=self.dtype)

        # Forward pass
        model_outputs = self.model(batch_images)

        # Loss Calculation
        total_loss = self.criterion(model_outputs, batch_masks)

        metric_dictionary = {"total_loss": total_loss}
        iou_metrics = self.intersection_over_union_per_class(model_outputs, batch_masks)
        metric_dictionary.update(iou_metrics)

        # Update tracker dictionary (rolling average accumulation)
        if tracker_dictionary is not None:
            for metric_key, metric_value in metric_dictionary.items():
                tracker_dictionary[metric_key] += metric_value.item() / self.information_dump

        # Log data to tensorboard (per-iteration)
        self.log_metrics_to_tensorboard(writer=writer,
                                        iteration=iteration,
                                        metric_dictionary=metric_dictionary)

        return total_loss, model_outputs

    def save_model(self, iteration: int) -> None:
        """
        Persists the current model weights and optimizer state to disk.

        Args:
            iteration (int): The current training iteration, used for naming the weight file.
        """
        model_directory = self.weights_directory / f"Iteration_{iteration}"
        model_directory.mkdir(exist_ok=True, parents=True)

        checkpoint_path = model_directory / f"model_{iteration:06}.pt"
        print(f"Saving Checkpoint At : {checkpoint_path}...")

        torch.save({
            'iteration': iteration,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict()}, checkpoint_path)

        print("Checkpoint Successfully Saved.")

        # Update the full-checkpoint registry
        self.dump_in_checkpoint(iteration=iteration)

    def log_metrics_to_tensorboard(self, writer: SummaryWriter, iteration: int,
                                   metric_dictionary: Dict[str, torch.Tensor]):
        """
        Logs individual metric components and total loss to TensorBoard.

        Args:
            writer (SummaryWriter): The TensorBoard writer to use.
            iteration (int): The current iteration index.
            metric_dictionary (Dict[str, torch.Tensor]): Dictionary containing current iteration metrics.
        """
        for metric_key, display_name in self.tracked_metrics_mapping.items():
            writer.add_scalar(display_name, metric_dictionary[metric_key].item(), iteration)

    def extract_last_model_iteration(self) -> int:
        """
        Retrieves the iteration number of the most recently saved model checkpoint.

        This method reads the 'full-checkpoint' registry file located in the weights
        directory to determine the latest available checkpoint. This is crucial for
        seamlessly resuming training after an interruption without manual intervention.

        Returns:
            int: The iteration number of the last saved model, or 0 if no checkpoint exists.
        """
        last_iteration = 0

        full_checkpoint_logger = self.weights_directory / "full-checkpoint"

        if not full_checkpoint_logger.exists():
            return last_iteration

        with open(full_checkpoint_logger, 'r') as file:
            first_line = file.readline().strip()

        last_iteration_string = first_line.split('"')[1]
        last_iteration = int(last_iteration_string.split("_")[-1])

        return last_iteration

    def restore_last_model(self, index_iteration: Optional[int] = None) -> int:
        """
        Restores the model and optimizer states to a specific or the most recent checkpoint.

        If `index_iteration` is provided, it restores that exact checkpoint (useful for
        running isolated evaluation or inference on a specific model state). If not provided,
        it automatically finds and loads the latest checkpoint to resume training.

        Args:
            index_iteration (Optional[int]): The specific iteration to restore. If None,
                it resolves to the last saved iteration.

        Returns:
            int: The iteration number from which training should commence. If a model was
                loaded, it returns `loaded_iteration + 1`. If no model was found, it returns 1.
        """

        if not index_iteration:
            index_iteration = self.extract_last_model_iteration()

        # Despite trying to get the last model None was found
        if not index_iteration:
            print_green(
                "No Initiation Model Weights Will Be Used..."
                "\nWe generate A New Model That Will Be Trained From Scratch.",
                add_separators=True)
            return 1
        else:
            self.load_model(iteration=index_iteration)
            print_green(f"We Loaded A Model That Was Previously Saved At Iteration {index_iteration}",
                        add_separators=True)
            # Resume from the next iteration
            return index_iteration + 1

    def load_model(self, iteration: int) -> None:
        """
        Loads the model weights and optimizer state from a specified iteration checkpoint.

        This method reads the serialized `.pt` file from disk and maps the tensors to
        the currently active device (CPU, CUDA, or MPS).

        Args:
            iteration (int): The exact iteration number identifying the checkpoint to load.
        """

        model_path = self.weights_directory / f"Iteration_{iteration}" / f"model_{iteration:06}.pt"

        checkpoint = torch.load(model_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

    def dump_in_checkpoint(self, iteration: int) -> None:
        """
        Updates the checkpoint registry file to track the most recently saved model.

        The `full-checkpoint` file acts as a manifest, keeping a historical record of
        all saved checkpoints and explicitly marking the latest one. This ensures the
        resumption logic (`extract_last_model_iteration`) always knows where to start.

        Args:
            iteration (int): The iteration number of the newly saved checkpoint.
        """
        checkpoint_file = self.weights_directory / "full-checkpoint"

        # Saved the iteration in a checkpoint file
        try:
            with open(checkpoint_file, "r") as f:
                d = f.readlines()
        except FileNotFoundError:
            d = []

        with open(checkpoint_file, "w") as f:
            if len(d) == 0:
                d.append(f'model_checkpoint_path: "Iteration_{iteration}"\n')
                d.append(f'all_model_checkpoint_paths: "Iteration_{iteration}"\n')
            else:
                d[0] = f'model_checkpoint_path: "Iteration_{iteration}"\n'
                d.append(f'all_model_checkpoint_paths: "Iteration_{iteration}"\n')
            for line in d:
                f.write(line)

    def console_log_update_tracker(self, iterations: int,
                                   training_tracker_dictionary: Dict[str, Any],
                                   validation_tracker_dictionary: Optional[Dict[str, Any]] = None):
        """
        Prints the rolling average of metrics to the console.

        Args:
            iterations (int): Current global training iteration.
            training_tracker_dictionary (Dict[str, Any]): Rolling average trackers for training.
            validation_tracker_dictionary (Optional[Dict[str, Any]]): Rolling average trackers for validation.

        Returns:
            Dict[str, Any]: A fresh training tracker dictionary for the next interval.
        """
        print_length = 100
        print("-" * print_length)
        print(f"Iteration: {iterations}")

        for metric_key, display_name in self.tracked_metrics_mapping.items():
            train_value = training_tracker_dictionary[metric_key]
            message = f"Moving Average of Training {display_name:40} : {train_value:.4f}"
            print_blue(message)

            if validation_tracker_dictionary is not None:
                validation_value = validation_tracker_dictionary[metric_key]
                validation_message = f"Moving Average of Validation {display_name:38} : {validation_value:.4f}"
                print_yellow(validation_message)

        duration = time.time() - training_tracker_dictionary['start_time']
        print(f"These {self.information_dump} iterations took {duration:.2f} seconds")
        print("-" * print_length)

        return self.get_metric_trackers()

    def get_metric_trackers(self) -> Dict[str, Any]:
        """
        Initializes a dictionary to track rolling averages of metrics.

        Returns:
            Dict[str, Any]: A dictionary with metric keys set to 0.0 and a start_time.
        """
        tracker_dictionary = {"start_time": time.time()}
        for metric_key in self.tracked_metrics_mapping.keys():
            tracker_dictionary[metric_key] = 0.0

        return tracker_dictionary
