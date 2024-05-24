# MNIST Digit Recognition with PyTorch

## A Deep Learning Approach to Handwritten Digit Recognition

---

### Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

Welcome to our MNIST Digit Recognition project! This repository contains a PyTorch implementation of a convolutional neural network (CNN) for recognizing hand-written digits from the MNIST dataset. Our goal is to provide a reliable and accurate model that can effectively identify digits from 0 to 9.

---

## Installation

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- numpy
- scikit-learn

### Setup

1. Clone the repository:

   ```bash
   https://github.com/Mohamedhendawy312/MNIST-handwritten-digit-classification.git

2. Navigate to the project directory:

   ```bash
   cd MNIST-handwritten-digit-classification

3. Install dependencies:

   ```bash
   pip install -r requirements.txt

### Usage

To train and evaluate the model:

    ```bash
     python train.py

### Model Architecture

  Our neural network architecture consists of:
  
  - Convolutional layers
  - Batch normalization
  - Max pooling
  - Dropout layers
  - Fully connected layers
  The architecture is defined in model.py.

### Training

  We train the model using stochastic gradient descent (SGD) with cross-entropy loss as the loss function. The learning rate is adjusted using a scheduler for better convergence.

### Evaluation

  After training, we evaluate the model on both the validation and test sets. We compute accuracy, precision, recall, and F1 score to assess the model's performance.

### Visualization

  We provide a function to visualize random images from the test set along with their ground truth and predicted labels.

### Results

  Our model achieves an accuracy of 99% on the validation set and 99% on the test set. The precision, recall, and F1 score on the validation set are as follows: Precision: 99%, Recall: 99%, F1 Score: 99%

### Contributing

  We welcome contributions! If you have any ideas for improvement or bug fixes, feel free to open an issue or submit a pull request.

### License

  This project is licensed under the MIT License. See the LICENSE file for details.

