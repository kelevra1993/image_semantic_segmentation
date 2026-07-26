from pathlib import Path
from ultrasound_segmentation.trainer.trainer import Trainer
from utilities.os_utilities import load_experiment_configuration, print_yellow
from utilities.tensor_utilities import get_device


def main():
    # Define configuration path
    experiment_configuration_path = Path(__file__).parent / "configurations" / f"configuration.yaml"

    # Load configuration
    experiment_configuration, model_configuration = load_experiment_configuration(experiment_configuration_path)


    # Initialize Trainer
    trainer = Trainer(
        project_root=experiment_configuration["project_root"],
        model_configuration=model_configuration,
        data_folder=experiment_configuration["data_folder"],
        number_iterations=experiment_configuration["number_iterations"],
        weight_saving_iterations=experiment_configuration["weight_saving_iterations"],
        learning_rate=experiment_configuration["learning_rate"],
        dtype=experiment_configuration["dtype"],
        information_dump=experiment_configuration["information_dump"],)

    # Start training loop
    print(f"Starting Training For Experiment: {experiment_configuration['experiment_name']}...")
    trainer.run_training_loop()
    print("Training Example Completed.")


if __name__ == "__main__":
    main()
