# Plant Disease Classification using CNN and Transfer Learning

![Animation](https://github.com/user-attachments/assets/b9981d41-2844-4ddb-8944-92edbfe05c3e)




This project leverages Convolutional Neural Networks (CNNs) and Transfer Learning to classify diseases in apple plants. Using multiple deep learning models, including ResNet50, VGG16, InceptionV3, DenseNet121, Xception and MobileNet, the model is trained to identify diseases like black rot, mosaic virus, and apple scab from images. The project also includes a comparison of models trained from scratch versus transfer learning.

## Table of Contents
Project Overview
Dataset
Model Architecture
Training and Evaluation
Key Results
How to Run the Project
Installation
Conclusion
Future Work

## Project Overview
This project addresses the problem of plant disease classification, particularly for apple plants. Traditional methods require expert pathologists, which is time-consuming and expensive. By utilizing deep learning models, this project aims to automate and simplify the identification process, providing a faster and more accessible solution to farmers.
The project uses the NZDLPlantDisease-v1 dataset, consisting of 15,706 images of diseased and healthy apple plants across 7 categories, such as black spot, leaf scab, and mosaic virus.

## Dataset
The dataset contains both healthy and diseased apple plant images, collected under different lighting conditions and angles to simulate real-world horticultural environments.
### Name: 
NZDLPlantDisease-v1
### Size: 
15,706 augmented images of apple plant leaves, stems, and fruits.
### Classes: 
7 (e.g., Black Rot, Leaf Scab, Healthy)
### External Test Dataset: 
726 images from the same 7 classes.

## Model Architecture
The project compares multiple CNN architectures:
ResNet50
VGG16
InceptionV3
Xception
DenseNet121
MobileNet

### Two training strategies are employed:
### Training from scratch:
Models are trained from randomly initialized weights.
### Transfer Learning: 
Models use pre-trained weights from ImageNet, then fine-tuned for plant disease classification.
Each model is evaluated based on performance metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

### Training and Evaluation
The models are trained using Google Colab with GPU support for faster computations. During training:

### Data Augmentation: 
The dataset is augmented with rotations, brightness adjustments, and flips to increase variability and prevent overfitting.
### Callbacks: 
Early stopping and learning rate reduction techniques are used to optimize training.
### Normalization: 
All images are normalized to a range of [0,1] to ensure stable training.

### Evaluation Metrics:
Accuracy
Precision
Recall
F1-score
Confusion Matrix
ROC-AUC

### Best Model:
The DenseNet121 model, trained from scratch, achieved the highest accuracy on the test set at 98.46%. 

![image](https://github.com/user-attachments/assets/71797343-aa4d-480b-9c8c-a149bd5931d3)

However, when tested on an external dataset, the accuracy dropped to 56%, highlighting potential challenges with model generalization.
Models performed well in test datase but had difficulty generalizing to external datasets.

![image](https://github.com/user-attachments/assets/2811e82d-7ac5-457f-a46a-4bb5f7dd5bec)


### Conclusion
This project demonstrates the power of deep learning for prediction of plant diseases in apple plants. The best-performing model, DenseNet121, shows high accuracy in on both training and test datasets, it also was able to predict external datasets but there is need for more iteration and use of different optimization algorithms for better results.

### Future Work
Future work might include comparing transfer learning (feature extraction) and fine tuning on classification problems using the NZDLPlantDisease-v1. Another study can use different optimization algorithms aside from Adam or compare various optimization algorithms to train models with the dataset. 

