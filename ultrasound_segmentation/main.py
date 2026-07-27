import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())
from ultrasound_segmentation.trainer.trainer import Trainer
from utilities.os_utilities import load_experiment_configuration, print_yellow


def main() -> None:
    """
    Entry point for the ultrasound segmentation training process.

    This function loads the experiment and model configurations from a YAML file,
    initializes the Trainer class with these configurations, and starts the
    main training loop.
    """
    # Define configuration path
    experiment_configuration_path = Path(__file__).parent / "configuration" / f"configuration.yaml"

    # Load configuration
    experiment_configuration, model_configuration = load_experiment_configuration(experiment_configuration_path)

    # Initialize Trainer
    trainer = Trainer(experiment_configuration=experiment_configuration,
                      model_configuration=model_configuration)

    # Start training loop
    print(f"Starting Training For Experiment: {experiment_configuration['project_root']}...")
    trainer.run_training_loop()
    print("Training Example Completed.")


if __name__ == "__main__":
    main()
