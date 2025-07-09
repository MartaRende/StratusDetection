# StratusDetection

A project aimed at predicting the appearance and disappearance of stratus clouds located in the plains of the Canton of Vaud, with short-term forecasts.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Code Overview](#code_overview)
- [Usage](#usage)
- [Data Sources and Attribution](#data-sources-and-attribution)

## Overview

This project analyzes stratus cloud phenomena by processing weather images from La Dôle and meteorological data from INCA provided by MeteoSwiss.

## Requirements

- This project requires to install [uv](https://pypi.org/project/uv/)
- To ensure the project functions correctly and achieves high-quality training, you will need the following data:

    - **2 years of MeteoSwiss images (2023-2023)** in 512x512 format, with file paths such as `./images/mch/1159/2/2023/01/01/1159_2024-11-16_0000.jpeg`, where `1159` refers to the La Dôle camera. Images should be spaced 10 minutes apart.
    - **2 years of INCA meteorological data (2023-2024)** in binary files, with paths like `./weather/inca/2023/20230101.nc`.
    - **2 years of Idaweb data (2023-2024)**, specifically solar radiation measurements, available from the Idaweb platform.
    - Sufficient computational resources to analyze the data and run the training processes.

Ensure that these datasets are available and organized as described for successful training and inference.

## Project branch
The project is mainly developed on two branches:
- **main**: manages the version of the model that outputs a single prediction (e.g., at 10 min, 30 min, or 1 hour).
- **multiple_prevision**: manages the version of the model that outputs multiple predictions at once (by default, from 10 minutes up to 1 hour).

## Project Structure


```
StratusDetection/
├── data/                     # Directory for input data files (not included in the repository)
├── models/                   # Saved trained models and related outputs
├── data_tools/               # Utilities for data preprocessing, augmentation
├── metrics_analysis/         # Scripts for evaluation metrics and analysis of results
├── docs/                     # Project documentation and research notes
├── data_loader.py            # Module for loading and preparing datasets for training
├── data_analysis.ipynb       # Jupyter notebook for exploratory data analysis
├── training.py               # Script to train the machine learning model
├── inference.py              # Script to run inference and evaluate model predictions
├── inference_sbatch.sh       # SLURM batch script for inference on HPC infrastructure
├── model.py                  # Defines the model architecture
├── prepare_data_inference.py # Prepares data specifically for inference
├── prepareData.py            # Prepares and validates input data for training
├── pyproject.toml            # Project dependencies and build configuration
├── rules.def                 # Apptainer definition file for containerized environments
├── train_sbatch.sh           # SLURM batch script for training on HPC infrastructure
└── README.md                 # Project overview and usage instructions

```


## Installation
### To install locally
1. Clone project
```bash
git clone https://github.com/yourusername/StratusDetection.git
cd StratusDetection
```


### To install on Infrastructure with Apptainer and SLURM

In the context of this project, due to the large volume of data to be processed and the need to run extensive model training, I had access to more powerful computational infrastructure. This allowed me to properly submit jobs using SLURM for large-scale training.

The infrastructure uses Apptainer to execute jobs in isolated and reproducible environments. To launch a job:

1. Build an Apptainer image that includes all project dependencies using the `rules.def` definition file.
    The image definition is provided in `rules.def`.
    You can build the image directly with:

    ```bash
    apptainer build train.sif rules.def
    ```
    This will create the image file `train.sif`.

## Code Overview

The repository is organized into several key scripts and folders:

- **model.py**  
    Defines the architecture of the machine learning model used for stratus detection.

- **training.py**  
    Manages the end-to-end training process, including data preparation, splitting into train/validation/test sets, and transforming data into tensors (leveraging `data_loader.py`).  
    At the end of training, this script creates a subfolder inside the `models/` directory named `model_n`, where `n` is the next available model number. This folder contains the trained model (`model.pth`), the model architecture (`model.py`), the test data used, loss graph, and several files useful for inference.

- **prepareData.py**  
    Contains classes and functions for preparing and validating input data (images and meteorological data) for training.

- **prepare_data_inference.py**  
    Prepares and validates data specifically for inference on the test set.

- **inference.py**  
    Loads the trained model, runs inference on the test data, computes evaluation metrics, and generates result visualizations.  
    Saves a file with the expected and predicted values to avoid having to run inference with the model every time.

- **data_analysis.ipynb**  
    Jupyter notebook with exploratory analyses of the images and meteorological data.

- **rules.def**  
    Apptainer definition file for building a reproducible container image to run jobs on high-performance infrastructure.

- **train_sbatch.sh** and **inference_sbatch.sh**  
    SLURM batch scripts for submitting training and inference jobs on a computing cluster.

### Folders

- **metrics_analysis/**  
    Scripts for generating evaluation metrics, creating plots, and analyzing stratus dissipation delays.

- **data_tools/**  
    Utilities for filtering, augmenting, and visualizing study data, including scripts for image cropping and data augmentation.
    - The `add_idaweb_data.py` script was also used to replace the solar radiation data for Nyon with that from Geneva.
    - The `preprocessing.py` script is used to filter INCA data for La Dôle and solar radiation data for Nyon and La Dôle, saving them in a `.npz` file for use in `training.py`.
    - The `CT_exam.py` file is used for visual analysis of INCA data.

- **docs/**  
    Project documentation, research notes, and progress updates.

- **data/**  
    Directory for input and output data files (not included in the repository to preserve data privacy but essential for running training). 

Each script and module includes inline comments and docstrings for further details. For usage and customization, refer to the documentation within each file.


## Usage

## Preprocessing Before Training

Before starting training, run the `preprocessing.py` script to filter out all null INCA data and save the cleaned data in a `.npz` file. This step ensures more efficient data processing during training. The script also allows you to save the initial Idaweb solar radiation data for La Dôle and Nyon. Make sure the file paths are correct before running the script. This process may take some time locally but only needs to be done once.

To run the preprocessing step, execute:
```bash
uv run -m data_tools.preprocessing
```

The processed file will be saved as `data/complete_data.npz`.
### To run locally
To run training or inference locally, use:

```bash
uv run -m training
uv run -m inference
```

By default, these commands will launch training or inference locally (ensure the images are available locally with the correct file paths), using one camera view from La Dôle, three temporal sequences of images/meteorological data, and the default forecast time.

You can customize the inputs by passing the following arguments:

1. **First argument:** `0` for local execution, `1` for infrastructure execution (changes the image paths).
2. **Second argument:** Number of views to use (`1` for one camera view from La Dôle, `2` for both views).
3. **Third argument:** Number of temporal data points (images + meteorological data).
4. **Fourth argument** (only for the `main` branch; in the `multiple_prevision` branch, the number of prediction steps is fixed at 6, up to 1 hour): desired prediction time in minutes.

Example usage:
```bash
uv run -m training 0 2 3 30
uv run -m inference 0 2 3 30
```
This command runs training or inference locally, using two views, three temporal sequences, and a 30-minute forecast.

### Run on infrastructure
### Running on Infrastructure with Apptainer and SLURM

To perform training and inference on the infrastructure, use the SLURM scripts (`train_sbatch.sh` and `inference_sbatch.sh`) to submit jobs to the computing platform.

These scripts include SLURM directives (`SBATCH options`) to configure resources, execution time, and output logs. The main commands are:

- **For training:**
    ```bash
    apptainer exec --nv --bind /data/datasets/photocast:/data/datasets/photocast /data/datasets/marta.rende/train.sif python3 -u training.py 1 1 3 10
    ```
- **For inference:**
    ```bash
    apptainer exec --nv --bind /data/datasets/photocast:/data/datasets/photocast /data/datasets/marta.rende/train.sif python3 -u inference.py 1 1 3 10
    ```

The arguments after `python3 -u training.py` or `inference.py` are the same as those used locally. The `--bind /data/datasets/photocast:/data/datasets/photocast` option mounts the image directory from the host filesystem inside the container. The Apptainer image (`train.sif`) must have been previously built and, in this example, is located at `/data/datasets/marta.rende/train.sif`.

To launch the scripts:
```bash
sbatch ./train_sbatch.sh
sbatch ./inference_sbatch.sh
```

This setup allows you to leverage advanced computational resources and ensures experiment reproducibility.
> **Note:** In both cases, before running inference, you must open the `inference.py` file and set the desired model number in the `MODEL_NUM` variable.


## Data Sources and Attribution

If you use the data provided in this project, please cite the following sources:

- **La Dôle Weather Images:** Data courtesy of MeteoSwiss.  
    MeteoSwiss, Federal Office of Meteorology and Climatology. [https://www.meteoswiss.admin.ch/](https://www.meteoswiss.admin.ch/)

- **INCA Meteorological Data:**: Provided by MeteoSwiss.  
    INCA (Integrated Nowcasting through Comprehensive Analysis), MeteoSwiss.

- **Idaweb Meteorological Data**:Provided by MeteoSwiss from Idaweb website [https://gate.meteoswiss.ch/idaweb](https://gate.meteoswiss.ch/idaweb)