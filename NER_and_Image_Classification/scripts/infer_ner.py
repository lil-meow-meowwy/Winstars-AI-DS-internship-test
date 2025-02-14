from transformers import pipeline
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import argparse


# Load the trained NER model and tokenizer
model_path = os.path.join("..", "models", "ner_model")
ner_pipeline = pipeline("ner", model=model_path, tokenizer=model_path)

# Define the label list (must match the one used during training)
label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC", "B-ANIMAL", "I-ANIMAL"]

# Function to extract animal names from text
def extract_animal(text):
    # Run the NER pipeline on the input text
    entities = ner_pipeline(text)
    
    # Debugging: Print all entities detected by the model
    print("All detected entities:", entities)
    
    # Extract animal names
    animals = []
    for entity in entities:
        if label_list[entity['entity']] == 'B-ANIMAL' or label_list[entity['entity']] == 'I-ANIMAL':
            animals.append(entity['word'])
    
    return animals

# Example usage
if __name__ == "__main__":
    text = "There is a cow and a dog in the picture."
    animals = extract_animal(text)
    print("Extracted animals:", animals)