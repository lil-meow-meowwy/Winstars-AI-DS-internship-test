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
│   ├── mnist_classifier_interface.py  # Abstract interface for all classifiers
│   ├── random_forest_model.py         # Random Forest Classifier implementation
│   ├── feed_forward_model.py          # Feed-Forward Neural Network implementation
│   ├── cnn_model.py                   # Convolutional Neural Network implementation
├── mnist_classifier.py                # Unified MNIST classifier wrapper
├── main.py                            # Script to train and evaluate models
├── requirements.txt                    # Dependencies
├── README.md                           # Project documentation
```

---

## **Installation**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/yourusername/mnist-classifier.git
cd mnist-classifier
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

---

## **Usage**  

### **Train and Evaluate a Model**  
Run `main.py` with a specified algorithm (`rf`, `nn`, or `cnn`):  

```bash
python main.py
```

Inside `main.py`, you can change the classifier type:  

```python
classifier = MnistClassifier(algorithm='rf')  # Change 'rf' to 'nn' or 'cnn' for different models
```

The script will:  
✅ Load the **MNIST** dataset  
✅ Train the selected model  
✅ Predict labels for test images  
✅ Compute and print **accuracy**  

---

## **Models**  

Each model implements the `MnistClassifierInterface` with `train()` and `predict()` methods.  

### **1. Random Forest (`rf`)**
- Uses **sklearn's** `RandomForestClassifier`
- Fast training but less accurate than deep learning models  

### **2. Feed-Forward Neural Network (`nn`)**
- Uses **sklearn's** `MLPClassifier`
- Fully connected layers, trained with backpropagation  

### **3. Convolutional Neural Network (`cnn`)**
- Built with **TensorFlow/Keras**
- Uses Conv2D, MaxPooling, and Dense layers
- More accurate but requires more computational power  

---

## **Display First 10 Images**  

To visualize the first 10 test images, add this code snippet inside `main.py`:  

```python
import matplotlib.pyplot as plt

# Display the first 10 test images
def plot_images(X_test, y_test):
    fig, axes = plt.subplots(1, 10, figsize=(15, 3))
    for i in range(10):
        axes[i].imshow(X_test[i].reshape(28, 28), cmap='gray')
        axes[i].set_title(f"Label: {y_test[i]}")
        axes[i].axis('off')
    plt.show()

plot_images(X_test, y_test)
```

Run the script again to see the images.  

---

## **Dependencies**  

This project requires the following Python libraries:  

```
scikit-learn==1.3.0
tensorflow==2.14.0
numpy==1.25.2
matplotlib==3.8.2
```

Install them using:  
```bash
pip install -r requirements.txt
```

---

## **Future Improvements**  
✅ Support for additional ML models (e.g., SVM, Decision Trees)  
✅ Hyperparameter tuning for better accuracy  
✅ Experiment with different CNN architectures  

---

## **License**  
This project is open-source and available under the MIT License.  

---

Let me know if you'd like any modifications! 🚀