# Image Semantic Segmentation Pipeline

## Table of Contents
- [Project Description](#project-description)
- [Prerequisites](#prerequisites)

## Project Description
This repository contains a modular PyTorch training pipeline for image semantic segmentation. It utilizes a customizable U-Net architecture to classify pixels into multiple labels (such as original, object_square, object_b, object_plus). 

The codebase features:
- A robust trainer with support for resuming training.
- Focal loss computation.
- Periodic validation and model checkpointing.
- Configuration management via YAML files to adjust hyperparameters, dataset paths, and U-Net structural parameters.
- Clear separation of concerns across modules like `trainer`, `model`, `loss`, and `utilities`.

## Prerequisites
- Python 3.8+
- PyTorch
- Additional requirements as specified in project configuration files.
