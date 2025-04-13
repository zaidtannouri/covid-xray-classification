# covid-xray-classification
Deep learning models for classifying COVID-19 from X-ray images using single and dual-input architectures.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [License](#license)

## Overview

This repository contains code for training two models to classify X-ray images as either "COVID" or "Non-COVID" using deep learning techniques. The dataset includes X-ray images of patients diagnosed with COVID-19, as well as non-COVID-19 images. 

### Key Features
- Uses a **VGG16** based model for classification.
- Supports single-input and dual-input models.
- Implements a modular training pipeline using PyTorch.
- Visualizes metrics and predictions during and after training.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- scikit-learn
- pandas
- PIL
- KaggleHub (for downloading the dataset)

## Dataset

The **COVID-19 Radiography Database** contains labeled X-ray images of COVID-19 positive and negative cases. The dataset can be easily downloaded using **KaggleHub**.

The dataset contains:
- COVID-19 positive X-ray images
- Non-COVID-19 X-ray images (including pneumonia, healthy, etc.)

You can access the dataset [here](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database).

## Usage

Once the repository is cloned and dependencies are installed, you can use the provided scripts to train and evaluate the models.

## Model Architecture

There are two models in this project:

1. **CovidXRayModel**: A model using a single X-ray image for classification.
2. **DualCovidXRayModel**: A model using two images (X-ray image and lung filter image) to improve prediction accuracy.

Both models are based on **VGG16** architecture, with pre-trained weights, and fine-tuned for this specific classification task.

## Training

The models are trained using **Cross-Entropy Loss** and **Adam Optimizer**. During training, metrics such as loss and accuracy are printed for each epoch, and the best model is saved based on validation accuracy.

## Evaluation

After training, the models can be evaluated on the test dataset to measure accuracy and other metrics such as precision, recall, and F1 score. Visualizations of the predictions and ground truth will also be displayed for a more intuitive understanding.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
