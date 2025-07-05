# StratusDetection

A project aimed at predicting the appearance and disappearance of stratus clouds located in the plains of the Canton of Vaud, with short-term forecasts.

## Table of Contents

- [Overview](#overview)
- [Code Overview](#code_overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Overview

This project is designed to analyze stratus cloud phenomena by processing weather images from La Dôle and meteorological data from INCA provided by MeteoSwiss.
## Code Overview

The main codebase consists of:

- **model.py**  
    Defines the architecture of the machine learning model.

- **training.py**  
    Handles the training process: prepares meteorological data, images, and labels; splits data into train/validation/test sets; and transforms data into tensors using `data_loader.py`.

- **prepareData.py**  
    Contains a class for preparing and validating input data (images and meteorological data) for training.

- **prepare_data_inference.py**  
    Contains a class for preparing data specifically for inference on the test set.

- **inference.py**  
    Runs the trained model on test data, generates predictions, computes main metrics, and produces result visualizations.
- **data_analysis.ipynb**: 
    Contains preliminary analyses performed on the images and meteorological data.
- **rules.def**: 
    Nel quadro del progetto avevo a disposizione un'infrastruttura molto piu potente che il mio laptop per usarla devo lanciare in miei job con appteiner quindi questo file è la definizione della mia immagine apptainer.
- **train_sbatch.sh and inference_sbatch.sh**: 

- **metrics_analysis/**  
    Scripts for generating metrics, creating various plots, and analyzing stratus dissipation delays.

- **data_tools/**  
    Utilities for filtering study data and visualizing images, including data augmentation and cropping.

- **docs/**  
    Documentation, research notes, and project progress updates.

Each module is documented with inline comments. For more details on usage and customization, refer to the docstrings within each script.

## Requirements

- Python 3.8 or higher
- pip
- Compatible with Linux, macOS, and Windows
- See `requirements.txt` for Python dependencies

## Installation

```bash
git clone https://github.com/yourusername/StratusDetection.git
cd StratusDetection
pip install .
```

## Usage

```bash
python detect_stratus.py --input data/input_file
```

## Project Structure

```
StratusDetection/
├── data/                  # Input and output data files
├── detect_stratus.py      # Main script for detection
├── requirements.txt       # Python dependencies
├── utils/                 # Utility modules and helper functions
├── visualization/         # Data visualization scripts
└── README.md              # Project documentation
```
