# Plant Disease Classification using CNN and Transfer Learning

![Animation](https://github.com/user-attachments/assets/b9981d41-2844-4ddb-8944-92edbfe05c3e)




This project leverages Convolutional Neural Networks (CNNs) both Training from scratch and Transfer Learning to classify diseases in apple plants. Using multiple deep learning models, including ResNet50, VGG16, InceptionV3, DenseNet121, Xception and MobileNet, the models are trained to identify diseases like black rot, Black spot (scab), Glomerella leaf spot,  mosaic virus, and European canker from Apple Leaf, Stem and Fruit images. The project also includes a comparison of models trained from scratch versus transfer learning.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Conclusion](#conclusion)
- [Future Work](#future-work)


## Project Overview
This project addresses the problem of plant disease classification, particularly for apple plants. Traditional methods require expert pathologists, which is time-consuming and expensive. By utilizing deep learning models, this project aims to automate and simplify the identification process, providing a faster and more accessible solution to farmers.
The project uses the NZDLPlantDisease-v1 dataset, consisting of 15,706 images of diseased and healthy apple plants across 7 categories, such as black spot, leaf scab, and mosaic virus.

## Dataset
The dataset contains both healthy and diseased apple plant images, collected under different lighting conditions and angles to simulate real-world horticultural environments.
The image dataset was extracted from this GitHub repository - https://github.com/hsaleem1/NZDLPlantDisease-v1 and was originally used in the work of (Saleem et. al, 2022). The original dataset consists of 5 crops which include Apple, avocado, Grapevine, Kiwi and Pear.

### Size of Dataset used in this Project: 
15,706 augmented images of apple plant leaves, stems, and fruits.
### Classes: 
7 classes that incudes black rot, Black spot (scab), Glomerella leaf spot,  mosaic virus, European canker, Healthy Leaf and Healthy Fruit

### External Test Dataset: 
726 images from the same 7 classes.


### Data Partitioning  
The dataset was split into 3 Subsets, namely. 
- Training
- Validation
- Test
The split of the dataset is in the ratio of 80:20:10. 80% of the dataset for training, 20% for Validation and 10% for testing the models.

### Data Augmentation: 
The dataset is augmented with rotations, brightness adjustments, and flips to increase variability and prevent overfitting.

## Model Architecture
The project compares multiple CNN architectures:
- ResNet50
- VGG16
- InceptionV3
- Xception
- DenseNet121
- MobileNet

### Two training strategies are employed:
-  Training from scratch
-  Transfer Learning: Training CNN architectures using transfer learning involves leveraging the weights of a pre-trained model.

Each model is evaluated based on performance metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

## Training and Evaluation
The models are trained using Google Colab with GPU support for faster computations. 

### Normalization
All images are normalized to a range of [0,1] to ensure stable training.

### Optimization Algorithm 
Early stopping and learning rate reduction techniques are used to optimize training.

### Evaluation Metrics:
Accuracy
Precision
Recall
F1-score
Confusion Matrix - True Positves, False Positives, True Negatives and False Negatives
ROC-AUC

### Best Model:
The DenseNet121 model, trained from scratch, achieved the highest accuracy on the test set at 98.46%. 

![image](https://github.com/user-attachments/assets/71797343-aa4d-480b-9c8c-a149bd5931d3)

However, when tested on an external dataset, the accuracy dropped to 56%, highlighting potential challenges with model generalization.
Models performed well in test datase but had difficulty generalizing to external datasets.

![image](https://github.com/user-attachments/assets/2811e82d-7ac5-457f-a46a-4bb5f7dd5bec)


## Conclusion
This project demonstrates the power of deep learning for prediction of plant diseases in apple plants. The best-performing model, DenseNet121, shows high accuracy in on both training and test datasets, it also was able to predict external datasets but there is need for more iteration and use of different optimization algorithms for better results.

## Future Work
Future work might include comparing transfer learning (feature extraction) and fine tuning on classification problems using the NZDLPlantDisease-v1. Another study can use different optimization algorithms aside from Adam or compare various optimization algorithms to train models with the dataset. 

