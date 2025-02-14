# **Named Entity Recognition (NER) and Image Classification**

This repository contains a machine learning pipeline that combines **Named Entity Recognition (NER)** and **Image Classification** to verify if an animal mentioned in a text description matches the animal present in an image. The pipeline is designed to handle user inputs in natural language and images, and it outputs a boolean value indicating whether the text description matches the image content.

---

## **Project Overview**

The project consists of two main components:
1. **Named Entity Recognition (NER):**
   - A transformer-based NER model is trained to extract animal names from text descriptions.
   - The model is fine-tuned on a custom dataset with animal-related entities.

2. **Image Classification:**
   - An image classification model is trained to classify animals in images.
   - The model is based on EfficientNetB3 and is trained on the [Animal Image Dataset (90 Different Animals)](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals) from Kaggle.

The pipeline integrates these two models to:
- Extract animal names from user-provided text.
- Classify the animal in a user-provided image.
- Compare the results and return `True` if the animal in the text matches the animal in the image, otherwise `False`.

---

## **Dataset**

The image classification model is trained on the [**Animal Image Dataset (90 Different Animals)**](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals) from Kaggle. This dataset contains images of 90 different animals, organized into separate folders for each animal class. The dataset is well-suited for training and evaluating image classification models.

### **Dataset Details:**
- **Number of Classes:** 90
- **Total Images:** 5,400
- **Image Format:** JPEG
- **Animal Classes:** Includes common animals such as cats, dogs, cows, elephants, and more exotic animals like flamingos, jellyfish, and kangaroos.

### **Dataset Structure:**
```
animal_image_dataset/
├── antelope/
│   ├── antelope1.jpg
│   ├── antelope2.jpg
│   └── ...
├── badger/
│   ├── badger1.jpg
│   ├── badger2.jpg
│   └── ...
└── ...
```

### **Usage:**
- The dataset is used to train the image classification model (`train_image_classifier.py`).
- Images are resized to 224x224 pixels and preprocessed using EfficientNetB3's preprocessing function.

---

## **Repository Structure**

```
NER_and_Image_Classification/
│
├── data/                                 # Folder to store the dataset
│   ├── images/                           # Folder containing animal images
│   └── animals.txt                       # File listing all animals in dataset
│  
├── models/                               # Folder to save trained models
│   ├── ner_model/                        # Saved NER model
│   └── image_classification_model/       # Saved image classification model
│
├── notebooks/                            # Folder for Jupyter notebooks
│   └── exploratory_data_analysis.ipynb   # EDA notebook
│
├── scripts/                              # Folder for Python scripts
│   ├── train_ner.py                      # Script to train the NER model
│   ├── infer_ner.py                      # Script to run inference with the NER model
│   ├── train_image_classifier.py         # Script to train the image classification model
│   ├── infer_image_classifier.py         # Script to run inference with the image classification model
│   └── pipeline.py                       # Script for the entire pipeline
│
└── requirements.txt                      # File listing all dependencies
```

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/lil-meow-meowwy/Winstars-AI-DS-internship-test.git
cd Winstars-AI-DS-internship-test/NER_and_Image_Classification
```

### **2. Install Dependencies**
Install the required Python packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### **3. Prepare the Dataset**
- Place your animal images in the `data/images/` directory, organized into subfolders by class (e.g., `data/images/cow/`, `data/images/dog/`).

### **4. Train the Models**
- **Train the NER Model:**
  ```bash
  python scripts/train_ner.py
  ```
- **Train the Image Classification Model:**
  ```bash
  python scripts/train_image_classifier.py
  ```

### **5. Run the Pipeline**
Use the `pipeline.py` script to run the entire pipeline:
```bash
python scripts/pipeline.py "A lion is roaming freely in the jungle."  data/images/lion/8caaf87eae.jpg
```

---

## **Scripts Overview**

### **1. `train_ner.py`**
- Trains a transformer-based NER model to extract animal names from text.
- Saves the trained model to `models/ner_model/`.

### **2. `infer_ner.py`**
- Loads the trained NER model and extracts animal names from user-provided text.

### **3. `train_image_classifier.py`**
- Trains an EfficientNetB3-based image classification model on the animal dataset.
- Saves the trained model to `models/image_classification_model/`.

### **4. `infer_image_classifier.py`**
- Loads the trained image classification model and predicts the animal in a user-provided image.

### **5. `pipeline.py`**
- Integrates the NER and image classification models.
- Takes a text description and an image as inputs, and outputs `True` if the animal in the text matches the animal in the image.

---

## **Example Usage**

### **Input:**
```bash
python scripts/pipeline.py "A lion is roaming freely in the jungle."  data/images/lion/8caaf87eae.jpg
```

### **Output:**
Also contains warnings.

```
A lion is roaming freely in the jungle.
Animals mentioned in text: ['lion']
Preprocessing image: data/images/lion/8caaf87eae.jpg
Running prediction...
Predicted animal in image: lion (Confidence: 1.00)
Match found: lion == lion
Result: True
```

---

## **Dependencies**

The project requires the following Python packages:
- `transformers`
- `torch`
- `tensorflow`
- `numpy`
- `pandas`
- `Pillow`


Install them using:
```bash
pip install -r requirements.txt
```

---

## **Contributing**

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
