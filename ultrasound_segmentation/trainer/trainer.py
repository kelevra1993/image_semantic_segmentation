import os
import sys
import yaml
import torch

# Add the project root or ultrasound_segmentation directory to the path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from model.model import UnetModel

def debug_unet_forward_pass():
    config_path = os.path.join(parent_dir, 'configuration', 'configuration.yaml')
    
    print(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    unet_config = config.get('UnetConfiguration')
    if not unet_config:
        print("Error: UnetConfiguration not found in the config file.")
        return

    print("Initializing UnetModel with configuration...")
    model = UnetModel(unet_config)
    
    # Check the model
    print(model)
    
    # Create a small random image tensor
    # Shape: (Batch, Channels, Height, Width)
    batch_size = 2
    channels = unet_config.get('input_channels', 1)
    height, width = 64, 64  # Small spatial dimensions for a quick test
    
    print(f"Creating a random tensor of shape: ({batch_size}, {channels}, {height}, {width})")
    dummy_input = torch.randn(batch_size, channels, height, width)
    
    print("Running forward pass...")
    try:
        output = model(dummy_input)
        print(f"Forward pass successful! Output shape: {output.shape}")
        expected_output_channels = unet_config.get('output_channels', 1)
        print(f"Expected output shape: ({batch_size}, {expected_output_channels}, {height}, {width})")
    except Exception as e:
        print(f"Forward pass failed with error: {e}")

if __name__ == "__main__":
    debug_unet_forward_pass()
