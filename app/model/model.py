import torch
import torch.nn as nn
import torch.nn.functional as functional

from app.utilities.os_utilities import print_yellow, print_blue, print_red, print_green


class ChannelLayerNormalizer(nn.Module):
    """
    Applies Layer Normalization over the channel dimension for 4D image tensors.
    
    This class is required because PyTorch's native LayerNorm expects the 
    normalized dimension to be the last dimension. This wrapper permutes 
    the tensor so normalization is applied across the channel dimension correctly.
    """

    def __init__(self, channels: int):
        """
        Initializes the ChannelLayerNormalizer.
        
        Args:
            channels (int): The number of channels in the input tensor.
        """
        super().__init__()
        self.normalizer = nn.LayerNorm(channels)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for channel-wise layer normalization.
        
        Args:
            input_tensor (torch.Tensor): The input image tensor of shape (Batch, Channels, Height, Width).
            
        Returns:
            torch.Tensor: The normalized tensor of shape (Batch, Channels, Height, Width).
        """
        # Permute from [Batch, Channels, Height, Width] to [Batch, Height, Width, Channels]
        input_tensor = input_tensor.permute(0, 2, 3, 1)
        input_tensor = self.normalizer(input_tensor)

        # Permute back to [Batch, Channels, Height, Width]
        return input_tensor.permute(0, 3, 1, 2)


class ConvolutionBlock(nn.Module):
    """
    A block containing a sequential series of convolutional layers, 
    each followed by layer normalization and LeakyReLU activation.
    """

    def __init__(self, input_channels: int, output_channels: int, number_of_layers: int):
        """
        Initializes the ConvolutionBlock.
        
        Args:
            input_channels (int): The number of channels in the initial input to the block.
            output_channels (int):
             The number of output channels for all convolutions in the block.
            number_of_layers (int): The number of convolutional layers to sequentially apply.
        """
        super().__init__()
        layers = []
        for index in range(number_of_layers):
            current_input_channels = input_channels if index == 0 else output_channels
            # 3x3 convolution, stride of 1, and padding of 1 acts as 'same' padding
            layers.append(nn.Conv2d(current_input_channels, output_channels, kernel_size=3, stride=1, padding=1))
            layers.append(ChannelLayerNormalizer(output_channels))
            layers.append(nn.LeakyReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for the convolution block.
        
        Args:
            input_tensor (torch.Tensor): The input tensor.
            
        Returns:
            torch.Tensor: The tensor after passing through the convolutional layers.
        """
        return self.block(input_tensor)


class UnetModel(nn.Module):
    """
    A U-Net architecture model constructed from a configuration dictionary.
    
    The model parses the configuration to construct a downsampling encoder path
    and an identical upsampling decoder path with skip connections.
    """

    def __init__(self, unet_configuration: dict):
        """
        Initializes the UnetModel using the provided configuration.
        
        Args:
            unet_configuration (dict): A dictionary containing the model settings, 
                including 'input_channels', 'output_channels', and 'convolution_blocks'.
        """
        super().__init__()

        # Get input and output channels from the configuration file
        input_channels = unet_configuration['input_channels']
        output_channels = unet_configuration['output_channels']

        # Definition of downsampling encoder
        self.encoder_blocks = nn.ModuleList()
        self.pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)

        blocks_configuration = unet_configuration['convolution_blocks']

        current_input_channels = input_channels
        self.encoder_output_channels = []

        for key, convolution_block_information in blocks_configuration.items():
            # key : [number_of_layers, channels]
            # Example 1 : [1, 32]
            number_of_layers, channels = convolution_block_information
            self.encoder_blocks.append(ConvolutionBlock(
                input_channels=current_input_channels,
                output_channels=channels,
                number_of_layers=number_of_layers))
            self.encoder_output_channels.append(channels)
            current_input_channels = channels

        # Definition of upsampling decoder
        self.up_convolutions = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        # Run through keys in the reverse order
        keys = sorted([int(key) for key in blocks_configuration.keys()])
        reversed_keys = keys[::-1]

        for index in range(len(reversed_keys) - 1):
            # Same process as for downsampling we just go through the downsampling architecture in reverse
            input_channel_count = self.encoder_output_channels[-(index + 1)]
            output_channel_count = self.encoder_output_channels[-(index + 2)]

            self.up_convolutions.append(nn.ConvTranspose2d(input_channel_count,
                                                           output_channel_count,
                                                           kernel_size=2, stride=2))

            number_of_layers, _ = blocks_configuration[str(reversed_keys[index + 1])]

            # Here the x2 comes from the fact that we are concatenating channels
            self.decoder_blocks.append(ConvolutionBlock(output_channel_count * 2,
                                                        output_channel_count,
                                                        number_of_layers))

        # Final layer before predictions
        self.final_convolution = nn.Conv2d(self.encoder_output_channels[0],
                                           output_channels,
                                           kernel_size=1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for the U-Net model.
        
        Args:
            input_tensor (torch.Tensor): The input image tensor.
            
        Returns:
            torch.Tensor: The predicted output segmentation mask.
        """
        encoder_features = []
        # Downsampling
        for index, block in enumerate(self.encoder_blocks):
            # Forward pass
            input_tensor = block(input_tensor)

            # Kept for later concatenation
            encoder_features.append(input_tensor)

            # We do not pool the last downsampling layer
            if index < len(self.encoder_blocks) - 1:
                input_tensor = self.pooling_layer(input_tensor)

        # Upsampling
        for index in range(len(self.decoder_blocks)):
            # First upsample input
            input_tensor = self.up_convolutions[index](input_tensor)

            # Get corresponding skip connection for concatenation
            skip_connection_feature = encoder_features[-(index + 2)]

            # Handle potential dimension mismatch (can happen with maxpooling if input isn't a power of 2)
            if input_tensor.shape != skip_connection_feature.shape:
                input_tensor = functional.interpolate(
                    input_tensor,
                    size=skip_connection_feature.shape[2:],
                    mode='bilinear', align_corners=True)

            input_tensor = torch.cat([skip_connection_feature, input_tensor], dim=1)
            input_tensor = self.decoder_blocks[index](input_tensor)

        output_tensor = self.final_convolution(input_tensor)

        return output_tensor

    def print_summary(self, image_size: tuple[int, int]) -> None:
        """
        Prints a structural summary of the U-Net architecture given a specific input image size.
        
        This function creates a dummy tensor and passes it through the encoder and decoder 
        paths to trace and display the spatial and channel dimensions at each significant 
        step of the model, which aids in debugging architecture configurations.
        
        Args:
            image_size (tuple[int, int]): The height and width of the input image.
        """
        # Determine input channels from the first encoder block's first convolutional layer
        first_convolution = self.encoder_blocks[0].block[0]
        input_channels = first_convolution.in_channels

        dummy_tensor = torch.zeros((1, input_channels, image_size[0], image_size[1]))
        device_type = next(self.parameters()).device
        dummy_tensor = dummy_tensor.to(device_type)

        print_yellow("U-Net Architecture Summary", add_separators=True)

        print(f"Input image shape: {list(dummy_tensor.shape)}")

        encoder_features = []

        print_red("--- Downscaling Path (Encoder) ---", add_separators=True, upper_space=1)
        for index, block in enumerate(self.encoder_blocks):
            dummy_tensor = block(dummy_tensor)
            encoder_features.append(dummy_tensor)
            print(f" - Encoder Block {index}")
            print(f"   - Encoder Output Shape : {list(dummy_tensor.shape)}")

            if index < len(self.encoder_blocks) - 1:
                dummy_tensor = self.pooling_layer(dummy_tensor)
                print(f"   - After MaxPool Shape  : {list(dummy_tensor.shape)}")

        print_red("--- Upscaling Path (Decoder) ---", add_separators=True, upper_space=1)
        for index in range(len(self.decoder_blocks)):
            print(f" - Decoder Block {index}")
            print(f"   - Decoder Input Shape   : {list(dummy_tensor.shape)}")

            dummy_tensor = self.up_convolutions[index](dummy_tensor)
            skip_connection_feature = encoder_features[-(index + 2)]
            print(f"   - Upsampled Input Shape : {list(dummy_tensor.shape)}")
            print(f"   - Skip Connection Shape : {list(skip_connection_feature.shape)}")

            dummy_tensor = torch.cat([skip_connection_feature, dummy_tensor], dim=1)
            dummy_tensor = self.decoder_blocks[index](dummy_tensor)
            print(f"   - Decoder Output Shape  : {list(dummy_tensor.shape)}")

        dummy_tensor = self.final_convolution(dummy_tensor)
        print_green(f"Final Convolution output shape: {list(dummy_tensor.shape)}", add_separators=True, upper_space=1)
