import os
import sys

# Add the 'scripts' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

# Import the modules
from infer_ner import extract_animal
from infer_image_classifier import predict_image_class

def pipeline(text, image_path):
    """
    Pipeline to verify if the animal mentioned in the text matches the animal in the image.

    Args:
        text (str): Text input from the user (e.g., "There is a cow in the picture.").
        image_path (str): Path to the image file.

    Returns:
        bool: True if the animal in the text matches the animal in the image, otherwise False.
    """
    print(text
          )
    # Step 1: Extract animal names from the text using the NER model
    animals_in_text = extract_animal(text)
    print(f"Animals mentioned in text: {animals_in_text}")

    if not animals_in_text:
        print("No animals detected in the text.")
        return False

    # Step 2: Predict the animal in the image using the image classification model
    predicted_class, confidence_score = predict_image_class(image_path)
    print(f"Predicted animal in image: {predicted_class} (Confidence: {confidence_score:.2f})")

    # Step 3: Check if the predicted animal matches any of the animals mentioned in the text
    for animal in animals_in_text:
        if animal.lower() == predicted_class.lower():
            print(f"Match found: {animal} == {predicted_class}")
            return True

    print("No match found.")
    return False

# Example usage
if __name__ == "__main__":
    import argparse

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Pipeline to verify if the animal in the text matches the animal in the image.")
    parser.add_argument("text", type=str, help="Text input from the user (e.g., 'There is a cow in the picture.').")
    parser.add_argument("image_path", type=str, help="Path to the image file.")

    # Parse arguments
    args = parser.parse_args()

    # Run the pipeline
    result = pipeline(args.text, args.image_path)
    print(f"Result: {result}")

# python scripts/pipeline.py "There is a cow in the picture." data/images/antelope/0a37838e99.jpg (falsw)
# python scripts/pipeline.py "A lion is roaming freely in the jungle."  data/images/lion/8caaf87eae.jpg (true)                              