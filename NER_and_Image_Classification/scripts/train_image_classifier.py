# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress warnings to keep the output clean
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Define constants for image dimensions, batch size, and number of epochs
IMAGE_PATH = 'NER_and_Image_Classification/data/images'
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 100

# Initialize a dictionary to store image paths and corresponding labels
dataset = {'image_path': [], 'labels': []}

# Loop through each class folder in the dataset directory
for class_name in os.listdir(IMAGE_PATH):
    class_dir = os.path.join(IMAGE_PATH, class_name)

    # Skip if it's not a directory
    if not os.path.isdir(class_dir):
        continue
    
    # Loop through each image in the class folder
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        
        # Append the image path and label to the dataset dictionary
        dataset["image_path"].append(image_path)
        dataset["labels"].append(class_name)

# Convert the dataset dictionary to a pandas DataFrame for easier manipulation
df = pd.DataFrame(dataset) 

# Encode the string labels into integers using LabelEncoder
label_encoder = LabelEncoder()
df['encoded_labels'] = label_encoder.fit_transform(df['labels'])

# Split the dataset into training and validation sets (70% training, 30% validation)
train_df, val_df = train_test_split(df, test_size=0.3, random_state=42)

# Initialize an ImageDataGenerator for preprocessing and potential data augmentation
generator = ImageDataGenerator(
    preprocessing_function = tf.keras.applications.efficientnet.preprocess_input,
)

# Create a data generator for the training set
train_generator = generator.flow_from_dataframe(
    train_df, 
    x_col='image_path', 
    y_col='labels', 
    target_size=(IMG_HEIGHT, IMG_WIDTH), 
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

# Create a data generator for the validation set
val_generator = generator.flow_from_dataframe(
    val_df, 
    x_col='image_path', 
    y_col='labels', 
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Function to create the EfficientNetB3 model
def create_model(num_classes):
    # Load the EfficientNetB3 model pre-trained on ImageNet, excluding the top layers
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base_model.trainable = False  # Freeze the base model to prevent training

    # Build the model by adding custom layers on top of the base model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),  # Pooling to convert the feature map to a 1D vector
        layers.Dense(256, activation='relu'),  # Fully connected layer with 256 units
        layers.BatchNormalization(),  # Normalize the activations of the previous layer
        layers.Dropout(0.45),  # Dropout layer to prevent overfitting
        layers.Dense(num_classes, activation='softmax')  # Output layer with softmax activation
    ])

    # Compile the model with Adam optimizer and categorical crossentropy loss
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    return model

# Determine the number of unique classes in the dataset
num_classes = len(df['labels'].unique())

# Create the model using the defined function
model = create_model(num_classes)

# Define callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=True, mode='min')

# Train the model using the training and validation generators
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stopping, lr_scheduler]
)

# Save the trained model to a file for future use
model.save('NER_and_Image_Classification/models/image_classification_model/image_classification_model.keras')