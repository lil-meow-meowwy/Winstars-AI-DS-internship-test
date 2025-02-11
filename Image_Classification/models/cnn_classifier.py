import tensorflow as tf
from mnist_classifier_interface import MnistClassifierInterface

class CNNClassifierModel(MnistClassifierInterface):
    def __init__(self):
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train):
        X_train = X_train.reshape(-1, 28, 28, 1)
        self.model.fit(X_train, y_train, epochs=5)

    def predict(self, X_test):
        X_test = X_test.reshape(-1, 28, 28, 1)
        return self.model.predict(X_test).argmax(axis=1)