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

- **model.py**: Questo file contiene la struttura del modello utilizzato 
- **training.py**: Questo file serve per poter addestrare il modello. In questo file si preparano i dati meteo, le immagini e le labels, si splittano i dati in train/validation/test e si trasformano in tensori.

- **prepareData.py**: Questa classe permette di preparare i dati per poi essere trasformati in tensori nel training. In essa quindi si generano i dati di input e si controlla la loro esistenza(immagini e dati meteo)

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
pip install -r requirements.txt
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
