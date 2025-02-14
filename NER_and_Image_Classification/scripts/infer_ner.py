from transformers import pipeline
import os
import torch

# Load the trained NER model and tokenizer
model_path = os.path.join("models", "ner_model")
ner_pipeline = pipeline("ner", model=model_path, tokenizer=model_path)

# Define label mapping (ensure this matches training)
label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC", "B-ANIMAL", "I-ANIMAL"]

label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

# Function to extract animals
def extract_animal(text):
    entities = ner_pipeline(text)

    animals = []
    for entity in entities:
        entity_label = entity["entity"]  # Use the entity directly

        if entity_label in ["B-ANIMAL", "I-ANIMAL"]:
            animals.append(entity["word"])

    return animals

# Example usage
if __name__ == "__main__":
    text = "Is there a dog in the picture?"
    animals = extract_animal(text)
    print("Extracted animals:", animals)
