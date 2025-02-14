from transformers import pipeline

# Load the trained NER model and tokenizer
model_path = os.path.join("..", "models", "ner_model")
ner_pipeline = pipeline("ner", model=model_path, tokenizer=model_path)

# Function to extract animal names from text
def extract_animal(text):
    # Run the NER pipeline on the input text
    entities = ner_pipeline(text)
    
    # Extract animal names
    animals = []
    current_animal = ""
    
    for entity in entities:
        if entity['entity'] == 'B-ANIMAL':
            if current_animal:
                animals.append(current_animal)
            current_animal = entity['word']
        elif entity['entity'] == 'I-ANIMAL':
            current_animal += " " + entity['word']
    
    if current_animal:
        animals.append(current_animal)
    
    return animals

# Example usage
if __name__ == "__main__":
    text = "There is a cow and a dog in the picture."
    animals = extract_animal(text)
    print("Extracted animals:", animals)