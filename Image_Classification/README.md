# Task 1. Image classification + OOP

This project implements a simple image classification system for the MNIST dataset using three different models:  
1. Random Forest Classifier (RF)  
2. Feed-Forward Neural Network (NN)  
3. Convolutional Neural Network (CNN)  

Each model follows an Object-Oriented Programming (OOP) approach and implements a common interface (`MnistClassifierInterface`).  
The `MnistClassifier` class provides a unified way to train and predict using any of the three models by specifying the algorithm type (`'rf'`, `'nn'`, or `'cnn'`).  

---

## **Project Structure**  

```
mnist_classifier/
├── models/
│   ├── mnist_classifier_interface.py    # Abstract interface for all classifiers
│   ├── random_forest_classifier.py      # Random Forest Classifier implementation
│   ├── feed_forward_nn_classifier.py    # Feed-Forward Neural Network implementation
│   ├── cnn_classifier.py                # Convolutional Neural Network implementation
├── mnist_classifier.py                  # Unified MNIST classifier wrapper
├── demo.ipynb                           # Jupyter Notebook for training & testing models
├── requirements.txt                     # Dependencies
├── README.md                            # Project documentation
```

---

## **Installation**  

### **1. Clone the Repository**  
```bash
git https://github.com/lil-meow-meowwy/Winstars-AI-DS-internship-test.git
cd Image_Classification
```

### **2. Create a Virtual Environment (Optional but Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### **3. Install Dependencies**  
```bash
pip install -r requirements.txt
```

## **Usage**  

### **Run the Jupyter Notebook**  
```bash
jupyter notebook demo.ipynb
```

Inside `demo.ipynb`, you can:  
1. Load the MNIST dataset.
2. Train any of the three models (`rf`, `nn`, `cnn`).
3. Make predictions.
---

## **Models**  

Each model implements the `MnistClassifierInterface` with `train()` and `predict()` methods.  

### **1. Random Forest (`rf`)**
- Uses sklearn's `RandomForestClassifier`
- Fast training but less accurate than deep learning models  

### **2. Feed-Forward Neural Network (`nn`)**
- Uses sklearn's `MLPClassifier`
- Fully connected layers, trained with backpropagation  

### **3. Convolutional Neural Network (`cnn`)**
- Built with TensorFlow/Keras
- Uses Conv2D, MaxPooling, and Dense layers
- More accurate but requires more computational power  

---
## **Dependencies**  

This project requires the following Python libraries:  

```
scikit-learn==1.3.0
tensorflow==2.14.0
numpy==1.25.2
matplotlib==3.8.2
notebook==7.0.6
```

Install them using:  
```bash
pip install -r requirements.txt
```

## **License**  
This project is open-source and available under the MIT License.  