import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate  # Use evaluate instead of load_metric
import numpy as np

# Load the CoNLL-2003 dataset with trust_remote_code=True
dataset = load_dataset("conll2003", trust_remote_code=True)

# Define a mapping for labels
label_list = dataset["train"].features["ner_tags"].feature.names
label_list.append("B-ANIMAL")  # Add a new label for animals
label_list.append("I-ANIMAL")  # Add a new label for animals (inside token)

# List of animals to recognize
animals = [
    "antelope", "badger", "bat", "bear", "bee", "beetle", "bison", "boar", "butterfly",
    "cat", "caterpillar", "chimpanzee", "cockroach", "cow", "coyote", "crab", "crow",
    "deer", "dog", "dolphin", "donkey", "dragonfly", "duck", "eagle", "elephant",
    "flamingo", "fly", "fox", "goat", "goldfish", "goose", "gorilla", "grasshopper",
    "hamster", "hare", "hedgehog", "hippopotamus", "hornbill", "horse", "hummingbird",
    "hyena", "jellyfish", "kangaroo", "koala", "ladybugs", "leopard", "lion", "lizard",
    "lobster", "mosquito", "moth", "mouse", "octopus", "okapi", "orangutan", "otter",
    "owl", "ox", "oyster", "panda", "parrot", "pelecaniformes", "penguin", "pig",
    "pigeon", "porcupine", "possum", "raccoon", "rat", "reindeer", "rhinoceros",
    "sandpiper", "seahorse", "seal", "shark", "sheep", "snake", "sparrow", "squid",
    "squirrel", "starfish", "swan", "tiger", "turkey", "turtle", "whale", "wolf",
    "wombat", "woodpecker", "zebra"
]

# Load a pre-trained tokenizer and model (DistilBERT)
model_name = "distilbert-base-cased"  # Using DistilBERT instead of BERT
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))

# Tokenize the dataset and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,  # Truncate sequences to the model's max length
        padding=True,     # Pad sequences to the same length
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special token (e.g., [CLS], [SEP])
            else:
                # Map the original label to the new label list
                original_label = label[word_idx]
                if original_label == -100:
                    label_ids.append(-100)
                else:
                    label_name = label_list[original_label]
                    if label_name.startswith("B-") or label_name.startswith("I-"):
                        # Replace with animal labels if the word is an animal
                        token = examples["tokens"][i][word_idx].lower()  # Ensure the token is lowercase
                        if token in animals:  # Check if the token is in the animal list
                            if label_name.startswith("B-"):
                                label_ids.append(label_list.index("B-ANIMAL"))
                            else:
                                label_ids.append(label_list.index("I-ANIMAL"))
                        else:
                            label_ids.append(original_label)
                    else:
                        label_ids.append(original_label)
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Apply tokenization and alignment
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="NER_and_Image_Classification/models/ner_model",  # Directory to save the model
    evaluation_strategy="epoch",    # Evaluate every epoch
    learning_rate=2e-5,             # Learning rate
    per_device_train_batch_size=16, # Batch size for training
    per_device_eval_batch_size=16,  # Batch size for evaluation
    num_train_epochs=20,             # Number of epochs
    weight_decay=0.01,              # Weight decay
    save_strategy="epoch",          # Save model every epoch
    logging_dir="logs",             # Directory for logs
    logging_steps=10,               # Log every 10 steps
    report_to="none",               # Disable external logging
)

# Define a function to compute metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Load the seqeval metric using evaluate
    metric = evaluate.load("seqeval")
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("NER_and_Image_Classification/models/ner_model")
tokenizer.save_pretrained("NER_and_Image_Classification/models/ner_model")