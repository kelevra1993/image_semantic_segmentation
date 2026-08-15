import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from app.utilities.os_utilities import load_experiment_configuration, print_yellow
from app.utilities.data_utilities.dataloader import get_dataloaders


def visualize_batch(image: np.ndarray, mask: np.ndarray) -> None:
    """
    Creates and saves a matplotlib visualization of an input image and its multi-channel masks.

    Args:
        image (np.ndarray): The input image array.
        mask (np.ndarray): The corresponding masks array.
    """
    # Create a matplotlib figure (1 for the image + 5 for the mask channels)
    fig, axes = plt.subplots(1, 6, figsize=(24, 5))

    # Plot Image
    if image.shape[0] == 1:
        axes[0].imshow(image[0], cmap='gray')
    else:
        axes[0].imshow(np.transpose(image, (1, 2, 0)))
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Titles matching our label dictionary
    titles = ["Background", "Tumor", "Mask Square", "Mask B", "Mask Plus"]

    # Plot each mask channel
    for i in range(5):
        axes[i + 1].imshow(mask[i], cmap='gray')
        axes[i + 1].set_title(titles[i])
        axes[i + 1].axis("off")

    plt.tight_layout()
    output_plot_path = Path(__file__).parent / "dataloader_debug_visualization.png"
    plt.savefig(output_plot_path)
    print(f"Visualization saved to file://{output_plot_path}")


def main() -> None:
    """
    Entry point for the dataloader visualization script.

    This function tests the data pipeline by loading the experiment configuration,
    initializing the training dataloader, and fetching the first batch of data. 
    It then creates a matplotlib plot displaying the input image alongside its
    corresponding multi-class segmentation masks, saving the output to disk 
    for visual verification of the dataset pipeline.
    """
    print("Loading Configurations...")
    app_folder = Path(__file__).parents[3] / "app"
    configuration_path = app_folder / "configuration" / "configuration.yaml"

    experiment_configuration, model_configuration = load_experiment_configuration(configuration_path)

    print("Initializing DataLoaders...")
    preprocessed_directory = experiment_configuration["dataset_folder"]

    train_loader, _, _ = get_dataloaders(
        preprocessed_directory=preprocessed_directory,
        experiment_configuration=experiment_configuration,
        model_configuration=model_configuration["UnetConfiguration"],
        batch_size=1,
        number_of_workers=0  # run on main thread for quick debug
    )

    print(f"Fetching First Batch From Training Set : (total batches: {len(train_loader)})...")
    images, masks = next(iter(train_loader))

    print(f"- Image tensor shape: {images.shape}")
    print(f"- Mask tensor shape: {masks.shape}")

    # Get the first image in the batch
    image = images[0].numpy()
    mask = masks[0].numpy()

    # Call the visualization function to plot the input image alongside all its corresponding multi-class masks
    visualize_batch(image=image, mask=mask)

    print_yellow("- Please review the saved image to verify the multi-class labels.")


if __name__ == '__main__':
    main()
