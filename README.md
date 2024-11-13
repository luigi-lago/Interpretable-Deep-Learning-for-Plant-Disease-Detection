# Plant Disease Detection with ResTS and EfficientNet

This project aims to evaluate and compare the performance of two deep learning architectures, ResTS (Resilient Teacher-Student) and EfficientNet-B0, in the classification of plant diseases from image data. It includes separate scripts and Jupyter notebooks for data preprocessing, model training, evaluation, and Grad-CAM-based explainability visualization.

## Project Structure

Here’s an overview of the directory structure and files in the project:

```bash
├── data
│   ├── raw                  # Raw, unprocessed data
│   ├── results              # Stores results and metrics from training and evaluation
│   ├── test                 # Test dataset
│   ├── train                # Training dataset
│   └── val                  # Validation dataset
├── images
│   ├── .gitkeep             # Placeholder file for version control
├── models
│   ├── EfficientNetB0_trained_model.h5  # Trained EfficientNet-B0 model weights
│   └── ResTS_trained_model.h5           # Trained ResTS model weights
├── notebooks
│   ├── data_separation.ipynb            # Notebook for data preparation
│   ├── EfficientNetB0_training.ipynb    # Notebook for EfficientNet-B0 training
│   ├── Grad_cam.ipynb                   # Notebook for Grad-CAM analysis
│   ├── model_comparison.ipynb           # Notebook for model comparison
│   └── ResTS_training.ipynb             # Notebook for ResTS training
├── .gitignore               # Git ignore file
├── Classes.py               # Python file defining custom classes
├── README.md                # This README file
```

## Getting Started

### Prerequisites

This project requires **Python 3.8.10** and the following libraries. You may install them using the command below:

```bash
pip install tensorflow==2.4.1 numpy==1.19.2 matplotlib pandas scikit-learn seaborn opencv-python-headless
```

## Directory and File Descriptions

### Data Directory (int<data>)

- **raw/**: This folder contains the raw, unprocessed images used for training, validation, and testing.
- **train/, val/, test/**: Processed images split into training, validation, and test datasets, respectively.
- **results/**: Stores model performance metrics and outputs (e.g., CSV files with evaluation metrics).

## Images Directory (images)
This folder holds generated images, such as Grad-CAM visualizations. The .gitkeep file ensures this folder is tracked by version control.

## Models Directory (models)

- **``EfficientNetB0_trained_model.h5``**: Contains the saved weights of the trained EfficientNet-B0 model.
- **``ResTS_trained_model.h5``**: Contains the saved weights of the trained ResTS model.

## Notebooks Directory (``notebooks``)

- **``data_separation.ipynb``**: Preprocesses the raw data and splits it into training, validation, and test datasets. Run this notebook first to prepare the data.

- **``EfficientNetB0_training.ipynb``**: Trains the EfficientNet-B0 model on the training dataset. Adjust parameters as needed, then run to generate EfficientNetB0_trained_model.h5.

- **``ResTS_training.ipynb``**: Trains the ResTS model on the training dataset, producing the ResTS_trained_model.h5. This notebook contains a dual-head training structure for ResTeacher and ResStudent networks.

- **``model_comparison.ipynb``**: Compares the performance of ResTS and EfficientNet-B0 models on the test set. Outputs a table with metrics such as accuracy, F1 score, and processing time.

- **``Grad_cam.ipynb``**: Uses Grad-CAM to generate visual explanations for predictions from ResTS and EfficientNet-B0 models. This notebook includes step-by-step instructions on creating Grad-CAM overlays.

## Classes.py
Defines custom classes and helper functions used in various notebooks.

## Instructions for Replication

**1. Prepare the Data:**

- Run data_separation.ipynb in the notebooks directory. This notebook will split the data into train, val, and test folders and save them in the data/ directory.

**2. Train the Models:**

- Open and run EfficientNetB0_training.ipynb to train EfficientNet-B0. The trained weights will be saved as EfficientNetB0_trained_model.h5 in the models/ directory.
- Similarly, run ResTS_training.ipynb to train the ResTS model, generating ResTS_trained_model.h5.

**3. Model Comparison:**

- Once training is complete, execute model_comparison.ipynb to evaluate both models on the test dataset. The notebook will output metrics for comparison.

**4. Grad-CAM Analysis:**

- For interpretability, run Grad_cam.ipynb to generate Grad-CAM visualizations. The notebook includes code to process both models (ResTS and EfficientNet-B0) with Grad-CAM, highlighting the regions of images that contributed most to the models' predictions.

This project serves as a comprehensive analysis of ResTS and EfficientNet-B0 architectures for plant disease classification, with steps for data processing, model training, evaluation, and visualization. By following the notebooks sequentially, the project can be replicated and adapted for similar classification tasks.