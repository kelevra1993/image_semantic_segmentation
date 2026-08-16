from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------
# Configuration Paths - Edit these directly as needed
# ---------------------------------------------------------
WEIGHTS_DIRECTORY_PATH = "/home/robert_kelevra/Projects/image_semantic_segmentation/experiments_folder/experiment_1/Weights"
DURATION_PER_FRAME_MS = 1000  # Speed of the GIF (milliseconds per frame)


def extract_training_progression_gifs() -> None:
    """
    Scans the weights directory for iteration folders, extracts sample prediction images,
    and compiles them into chronological GIFs. This creates a time-lapse of how the model
    learned to segment each specific sample over the training process.
    """
    weights_path = Path(WEIGHTS_DIRECTORY_PATH)
    output_path = Path(WEIGHTS_DIRECTORY_PATH) / "training_progression_gifs"

    if not weights_path.exists() or not weights_path.is_dir():
        print(f"Error: The weights directory '{WEIGHTS_DIRECTORY_PATH}'"
              f" does not exist. Please update WEIGHTS_DIRECTORY_PATH.")
        return

    # Create the root output folder for the GIFs
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Gifs Will Be Saved Under : file://{output_path}")

    # Find all Iteration folders
    iteration_folders = list(weights_path.glob("Iteration_*"))

    # Sort them numerically based on the iteration number to ensure chronological order in the GIF
    try:
        iteration_folders.sort(key=lambda folder: int(folder.name.split('_')[1]))
    except ValueError:
        print("Warning: Some folders did not match the 'Iteration_INTEGER' format and might not be sorted correctly.")

    if not iteration_folders:
        print("No iteration folders found. Exiting.")
        return

    # Dictionary to map sample_name -> list of chronological image paths
    sample_to_image_paths_mapping = {}

    for iteration_folder in iteration_folders:
        predictions_folder = iteration_folder / "test_sample_predictions"

        if not predictions_folder.exists() or not predictions_folder.is_dir():
            continue

        # Go through every sample image in this iteration's prediction folder
        for sample_image_path in predictions_folder.glob("sample_*.png"):
            sample_name = sample_image_path.stem  # For example: 'sample_0000'

            if sample_name not in sample_to_image_paths_mapping:
                sample_to_image_paths_mapping[sample_name] = []

            sample_to_image_paths_mapping[sample_name].append(sample_image_path)

    # Now generate a GIF for each tracked sample
    print(f"Compiling GIFs for {len(sample_to_image_paths_mapping)} unique samples...")

    for sample_name, chronological_image_paths in tqdm(sample_to_image_paths_mapping.items(),
                                                       desc="Creating Gif Samples For Evaluating Training :"):
        if not chronological_image_paths:
            continue

        # Load all frames into PIL Images
        frames = []
        for image_path in chronological_image_paths:
            try:
                frame = Image.open(image_path)
                frame.load()  # Force load into memory so we can safely close the file reference
                frames.append(frame)
            except Exception as read_error:
                print(f"Failed to load image {image_path}: {read_error}")

        if not frames:
            continue

        # Save as an animated GIF
        gif_destination_path = output_path / f"{sample_name}_progression.gif"

        frames[0].save(
            fp=str(gif_destination_path),
            format="GIF",
            append_images=frames[1:],
            save_all=True,
            duration=DURATION_PER_FRAME_MS,
            loop=0)  # 0 means loop infinitely

    print(f"Extraction Complete! All GIFs are located in: {output_path}")


if __name__ == "__main__":
    extract_training_progression_gifs()
