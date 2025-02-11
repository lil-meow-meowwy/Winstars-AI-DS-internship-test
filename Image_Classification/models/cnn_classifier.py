import tensorflow as tf
from models.mnist_classifier_interface import MnistClassifierInterface

class CNNClassifierModel(MnistClassifierInterface):
    """
    Convolutional Neural Network classifier for MNIST dataset.
    Implements the `MnistClassifierInterface`.
    """
    def __init__(self):
        """
        Initialize the CNN model.
        The model consists of:
        - A 2D convolutional layer with 32 filters and ReLU activation.
        - A max-pooling layer to reduce spatial dimensions.
        - A flattening layer to convert the output to 1D.
        - A dense hidden layer with 128 units and ReLU activation.
        - An output layer with 10 units (one for each class) and softmax activation.
        """
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)), # Conv layer
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), # Max-pooling layer
            tf.keras.layers.Flatten(), # Flatten to 1D
            tf.keras.layers.Dense(128, activation='relu'), # Hidden layer
            tf.keras.layers.Dense(10, activation='softmax') # Output layer
        ])

        # Compile the model with Adam optimizer and sparse categorical crossentropy loss
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train):
        X_train = X_train.reshape(-1, 28, 28, 1) # Reshape to add a channel dimension
        self.model.fit(X_train, y_train, epochs=5)

    def predict(self, X_test):
        X_test = X_test.reshape(-1, 28, 28, 1) # Reshape to add a channel dimension
        return self.model.predict(X_test).argmax(axis=1) # Return the class with the highest probability