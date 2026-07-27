import torch
import numpy as np
from typing import Optional, Any

from torch import nn
from utilities.os_utilities import print_blue, print_yellow, print_green


def get_device() -> torch.device:
    """
    Identifies and returns the most efficient available hardware device for tensor computations.

    This utility ensures that the project remains cross-platform compatible by prioritizing
    CUDA (NVIDIA GPUs), then MPS (Apple Silicon GPUs), and falling back to CPU if no
    accelerators are detected. It is used globally across all modules to maintain
    device consistency.

    Returns:
        torch.device: The detected torch.device (e.g., 'cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def print_tensor_shape(tensor: torch.Tensor, name: Optional[str] = "") -> None:
    """
    Logs the shape of a tensor to the console in a formatted blue string.

    In the global project context, this utility is used during development and debugging
    to verify that tensor dimensions align with expected shapes (e.g., MSA or Pair representations)
    after complex transformations or contractions.

    Args:
        tensor (torch.Tensor): The torch.Tensor whose shape will be printed. Shape: (*).
        name (Optional[str], optional): An optional label to identify the tensor in the output.
    """
    print_blue(f"Tensor {name} Is Of Shape : {list(tensor.shape)}")


def print_tensor_type(tensor: torch.Tensor, name: Optional[str] = "") -> None:
    """
    Logs the data type of a tensor to the console in a formatted yellow string.

    Ensures that tensors maintain consistent dtypes (e.g., torch.float32) across different
    architectural modules, preventing type mismatch errors during multi-platform
    execution (CPU/CUDA/MPS).

    Args:
        tensor (torch.Tensor): The torch.Tensor whose dtype will be printed. Shape: (*).
        name (Optional[str], optional): An optional label to identify the tensor in the output.
    """
    print_yellow(f"Tensor {name} Is Of Type : {tensor.dtype}")


def print_tensor_min_max(tensor: torch.Tensor, name: Optional[str] = "") -> None:
    """todo update docstring"""
    print_yellow(f"Tensor {name} Maximum Is : {tensor.max()}")
    print_yellow(f"Tensor {name} Minimum Is : {tensor.min()}")


def print_tensor_device(tensor: torch.Tensor, name: Optional[str] = "") -> None:
    """
    Logs the hardware device of a tensor to the console in a formatted green string.

    Crucial for identifying and resolving device placement issues, ensuring all tensors
    participating in an operation reside on the same hardware (CUDA, MPS, or CPU) as
    mandated by the project's cross-platform compatibility guidelines.

    Args:
        tensor (torch.Tensor): The torch.Tensor whose device will be printed. Shape: (*).
        name (Optional[str], optional): An optional label to identify the tensor in the output.
    """
    print_green(f"Tensor {name} Is On : {tensor.device}")


def print_tensor_status(tensor: torch.Tensor, name: Optional[str] = "") -> None:
    """
    Provides a comprehensive log of a tensor's shape, type, and device.

    Aggregates individual printing utilities to offer a single-point snapshot of a tensor's
    state. This is particularly useful for deep debugging within dense modules like the
    Evoformer or Structure Module where multiple transformations occur.

    Args:
        tensor (torch.Tensor): The torch.Tensor to inspect. Shape: (*).
        name (Optional[str], optional): An optional label to identify the tensor in the output.
    """
    print_blue(60*'-')
    print_tensor_shape(tensor=tensor, name=name)
    print_tensor_type(tensor=tensor, name=name)
    print_tensor_min_max(tensor=tensor,name=name)
    print_tensor_device(tensor=tensor, name=name)
    print_blue(60*'-')


def print_tensor_list(tensor: torch.Tensor, round: int = 4) -> None:
    """
    Converts a tensor to a list and prints it with specified rounding precision.

    Useful for inspecting the numerical values of small tensors or intermediate results
    during the development and testing of architectural components.

    Args:
        tensor (torch.Tensor): The torch.Tensor to print.
        round (int, optional): Number of decimal places for rounding. Defaults to 4.
    """
    print(np.round(tensor.tolist(), round))
