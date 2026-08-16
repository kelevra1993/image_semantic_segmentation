<h1 align="center">Image Semantic Segmentation Pipeline</h1>

<p align="center">
  <img src="README/progression-sample-1.gif" width="45%" />
  <img src="README/progression-sample-2.gif" width="45%" />
</p>


## Table of Contents
- [Project Description](#project-description)
- [Prerequisites](#prerequisites)
  - [Package Requirements](#package-requirements)
  - [Data Requirements](#data-requirements)

## Project Description
This repository contains a PyTorch training pipeline for image semantic segmentation. It utilizes a U-Net architecture to segment and classify pixels on the Breast Ultrasound Images Dataset into multiple labels, including tumor, square, letter B, and plus sign.

The codebase features:
- Automated dataset retrieval and preprocessing from Kaggle.
  - This is how we get the tumor class.
- Generation of synthetic dummy masks for multi-class training.
  - This is used in order to test multi-label segmentation since the original data only has one class.
- Configuration management via YAML files to adjust hyperparameters, dataset paths, and U-Net structural parameters.
  - More details are given in the configuration section that will be detailed later.

## Prerequisites

### Package Requirements
- Python 3.12+
- uv

To install the project requirements, simply run:

```bash
uv sync
```

### Data Requirements
This project uses the Breast Ultrasound Images Dataset from Kaggle, which provides ultrasound scans along with their corresponding tumor masks. Our data pipeline automatically downloads the raw data, cleans it, and generates synthetic shapes (squares, letter B, and plus signs) to simulate a multi-class segmentation task.

To retrieve, preprocess, and split the data into training, validation, and testing sets, run the data retrieval script from the project root:

```bash
uv run app/utilities/data_utilities/data_retrieval.py
```
