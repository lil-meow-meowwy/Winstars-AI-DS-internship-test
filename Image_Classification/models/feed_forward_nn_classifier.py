from mnist_classifier_interface import MnistClassifierInterface
import tensorflow as tf

class FeedForwardNNClassifierModel(MnistClassifierInterface):
    def __init__(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=5)

    def predict(self, X_test):
        return self.model.predict(X_test).argmax(axis=1)