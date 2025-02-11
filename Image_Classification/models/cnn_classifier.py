import tensorflow as tf
from tensorflow.keras import layers, models
from models.mnist_classifier_interface import MnistClassifierInterface

class CNNClassifierModel(MnistClassifierInterface):
    def __init__(self):
            self.model = models.Sequential([
                layers.Reshape(target_shape=(28, 28, 1), input_shape=(28, 28)),
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dense(10, activation='softmax')
            ])
            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train):
        X_train = X_train.reshape(-1, 28, 28, 1)
        self.model.fit(X_train, y_train, epochs=5)

    def predict(self, X_test):
        X_test = X_test.reshape(-1, 28, 28, 1)
        return self.model.predict(X_test).argmax(axis=1)