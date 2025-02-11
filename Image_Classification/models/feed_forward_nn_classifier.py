from models.mnist_classifier_interface import MnistClassifierInterface
import tensorflow as tf

class FeedForwardNNClassifierModel(MnistClassifierInterface):
    """
    Feed-Forward Neural Network classifier for MNIST dataset.
    Implements the `MnistClassifierInterface`.
    """
    def __init__(self):
        """
        Initialize the Feed-Forward Neural Network model.
        The model consists of:
        - A flattening layer to convert 28x28 images into 1D vectors.
        - A dense hidden layer with 128 units and ReLU activation.
        - An output layer with 10 units (one for each class) and softmax activation.
        """
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)), # Flatten 28x28 images to 1D
            tf.keras.layers.Dense(128, activation='relu'), # Hidden layer
            tf.keras.layers.Dense(10, activation='softmax') # Output layer
        ])

        # Compile the model with Adam optimizer and sparse categorical crossentropy loss
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=5) # Train for 5 epochs

    def predict(self, X_test):
        return self.model.predict(X_test).argmax(axis=1) # Return the class with the highest probability