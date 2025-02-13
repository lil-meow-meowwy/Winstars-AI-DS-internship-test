# Import necessary libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import pandas as pd
import os

# Define constants
MODEL_PATH  = 'NER_and_Image_Classification/models/image_classification_model/image_classification_model.keras'
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Load the trained model
print(f"Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# Load the class labels
class_labels = os.listdir('NER_and_Image_Classification/data/images') 

# Function to preprocess the image for inference
def preprocess_image(img_path, target_size=(224, 224)):
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
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_labels[predicted_class_index]
    
    # Return the predicted class and confidence score
    confidence_score = np.max(predictions)
    return predicted_class, confidence_score

# Main function to run inference on a single image or a directory of images
def infer_image_classifier(image_path):
    if os.path.isdir(image_path):
        # If the path is a directory, predict for all images in the directory
        results = []
        for img_name in os.listdir(image_path):
            img_full_path = os.path.join(image_path, img_name)
            if os.path.isfile(img_full_path):
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