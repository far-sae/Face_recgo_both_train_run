
# Custom Image Feature Analysis (CIFA) System

This repository contains a Custom Image Feature Analysis (CIFA) system designed to process and analyze images, extracting meaningful features, and making predictions based on a trained model. The system includes scripts for both feature extraction and prediction.

## Features

- **Custom Image Feature Extraction**: The `CIFA.py` script is designed to process images and extract custom features, which can be used for training and analysis.
- **Prediction**: The `predict.py` script utilizes a trained model to make predictions on new images based on the extracted features.

## Installation

To set up the environment, clone this repository and install the required dependencies using `pip`:

```bash
git clone https://github.com/far-sae/Face_recgo_both_train_run.git
cd Face_recgo_both_train_run
pip install -r requirements.txt
```

## Usage

### Step 1: Feature Extraction

Use the `CIFA.py` script to extract features from your image dataset.

```bash
python CIFA.py --input_dir /path/to/your/images --output_dir /path/to/save/features
```

### Step 2: Prediction

After extracting the features, use the `predict.py` script to make predictions using a trained model.

```bash
python predict.py --model /path/to/your/model --features /path/to/extracted/features
```

## Files and Directories

- **`CIFA.py`**: Script for extracting custom image features from a dataset.
- **`predict.py`**: Script for making predictions based on extracted features and a trained model.
- **`requirements.txt`**: File listing the required Python packages.

## Dependencies

The project relies on several Python libraries, which are listed in the `requirements.txt` file. You can install these dependencies using the command mentioned above.


