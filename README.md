# Cipher_Byte_Inernship

This repository contains two tasks completed during my internship:

1. **Image Classification using TensorFlow with the Fashion MNIST dataset**
2. **Creating a Chatbot using Transformers with TensorFlow**

## Task 1: Image Classification using TensorFlow with the Fashion MNIST dataset

### Overview

This project involves building a Convolutional Neural Network (CNN) model using TensorFlow to classify images from the Fashion MNIST dataset into one of 10 categories. The dataset consists of 28x28 grayscale images of 70,000 fashion products from 10 categories, with 7,000 images per category.

### Dataset

The dataset used is the Fashion MNIST dataset, which is available directly through TensorFlow's datasets module. The categories include:

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## Model Architecture

The model architecture includes:

- Convolutional layers
- Max pooling layers
- Fully connected layers
- Dropout layers

### Results

The model achieved an accuracy of approximately **87.4%** on the test set.

### How to Run

1. Clone the repository.
2. Install the required dependencies from `requirements.txt`.
3. Run the Jupyter Notebook or Python script to train the model and evaluate its performance.

```bash
git clone <repository_url>
cd <repository_folder>
pip install -r requirements.txt
python image_classification_fashion_mnist.py

#  Dependencies

Python 3.x
TensorFlow
NumPy
Matplotlib

# Task 2: Creating a Chatbot using Transformers with TensorFlow

Overview
This project involves creating a simple chatbot using a Transformer model implemented with TensorFlow. The chatbot is trained on a conversational dataset and can respond to user input in a conversational manner.

Model Architecture

The chatbot uses a Transformer architecture, which includes:
Embedding layers
Multi-head attention mechanisms
Positional encoding
Encoder and decoder layers

Dataset
The dataset used for training the chatbot is a simple conversational dataset, which consists of pairs of questions and answers.

Results
The chatbot is capable of responding to basic conversational inputs with a reasonable degree of accuracy.

How to Run
Clone the repository.
Install the required dependencies from requirements.txt.
Run the Jupyter Notebook or Python script to train the chatbot and start interacting with it.
bash
git clone <repository_url>
cd <repository_folder>
pip install -r requirements.txt
python chatbot_transformer.py
Dependencies
Python 3.x
TensorFlow
NumPy
NLTK (optional, for text preprocessing)
TensorFlow Datasets
Conclusion
These tasks provided valuable hands-on experience with TensorFlow for image classification and natural language processing. The skills gained during this internship will be instrumental in future projects.

License
This project is licensed under the MIT License - see the LICENSE file for details.

### Steps to Follow:

1. Create a `requirements.txt` file listing all the dependencies.
2. Replace `<repository_url>` and `<repository_folder>` with your actual repository URL and folder name.
3. Add the code files (e.g., `image_classification_fashion_mnist.py`, `chatbot_transformer.py`).
4. Update the results sections with the actual performance metrics of your models.
