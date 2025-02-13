# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import pandas as pd
import os

import warnings
warnings.filterwarnings('ignore') 
warnings.filterwarnings('ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # Suppress ABSL logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define the path to the trained model
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'image_classification_model', 'image_classification_model.keras'))

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit(1)

# Load the trained model
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Load class labels from the training generator (or validation generator)
# Replace this with the path to your training data directory
train_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'images'))

# Create an ImageDataGenerator to extract class labels
generator = tf.keras.preprocessing.image.ImageDataGenerator()
train_generator = generator.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Extract class labels from the generator
class_labels = list(train_generator.class_indices.keys())

# Function to preprocess the image for inference
def preprocess_image(img_path, target_size=(224, 224)):
    print(f"Preprocessing image: {img_path}")
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to predict the class of an image
def predict_image_class(img_path):
    # Preprocess the image
    img_array = preprocess_image(img_path)
    
    # Perform prediction
    print("Running prediction...")
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_labels[predicted_class_index]
        
    # Return the predicted class and confidence score
    confidence_score = np.max(predictions)
    return predicted_class, confidence_score

# Main function to run inference on a single image or a directory of images
def infer_image_classifier(image_path):
    print(f"Processing path: {image_path}")
    if os.path.isdir(image_path):
        # If the path is a directory, predict for all images in the directory
        print(f"Path is a directory. Processing all images...")
        results = []
        for img_name in os.listdir(image_path):
            img_full_path = os.path.join(image_path, img_name)
            if os.path.isfile(img_full_path):
                print(f"Processing image: {img_name}")
                predicted_class, confidence_score = predict_image_class(img_full_path)
                results.append({
                    'image_name': img_name,
                    'predicted_class': predicted_class,
                    'confidence_score': confidence_score
                })
        # Convert results to a DataFrame for better visualization
        results_df = pd.DataFrame(results)
        print(results_df)
    
    elif os.path.isfile(image_path):
        # If the path is a single image, predict for that image
        print(f"Path is a single image. Processing...")
        predicted_class, confidence_score = predict_image_class(image_path)
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence Score: {confidence_score:.4f}")
    
    else:
        print(f"The provided path '{image_path}' is not a valid file or directory.")

# Example usage
if __name__ == "__main__":
    import argparse

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Infer image class using a trained model.")
    parser.add_argument("image_path", type=str, help="Path to the image or directory of images to classify.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run inference
    infer_image_classifier(args.image_path)