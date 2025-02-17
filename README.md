# Pneumonia-Prediction

## 🏢 Description
This project aims to develop a CNN model to detect Pneumonia using the provided dataset.

## 📊 Dataset
In this project, the Kaggle Chest X-Ray Images (Pneumonia) dataset was used. The dataset consists of three separate files: train, validation, and test. Within each, folders contains pneumonia and normal images.

## Project Organization

- 	Importing Dataset
-   Creating and Saving a Model
-   Making Predictions 

## 📦 Project structure
```
.
├── README.md
├── chest_xray
│   └── chest_xray
│       ├── test
│       │   ├── NORMAL
│       │   └── PNEUMONIA
│       ├── train
│       │   ├── NORMAL
│       │   └── PNEUMONIA
│       └── val
│           ├── NORMAL
│           └── PNEUMONIA
├── cnn_model.keras
├── lung_300.jpg
├── main.ipynb
└── main.py

```

## About Model

While developing the CNN model, the Keras and TensorFlow libraries were used.
```
Test Accuracy = 0.754808  
Loss = 1.151425 
```


