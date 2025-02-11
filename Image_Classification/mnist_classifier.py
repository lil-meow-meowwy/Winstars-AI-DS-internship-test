from models.random_forest_classifier import RandomForestClassifierModel
from models.feed_forward_nn_classifier import FeedForwardNNClassifierModel
from models.cnn_classifier import CNNClassifierModel

class MnistClassifier:
    """
    Factory class for MNIST classifiers.
    Takes an algorithm name as input and provides a unified interface for training and prediction.
    """
    def __init__(self, algorithm: str):
        """
        Initialize the classifier based on the specified algorithm.
        :param algorithm: Name of the algorithm ('rf', 'nn', or 'cnn').
        """
        if algorithm == 'rf':
            self.model = RandomForestClassifierModel() # Random Forest
        elif algorithm == 'nn':
            self.model = FeedForwardNNClassifierModel() # Feed-Forward Neural Network
        elif algorithm == 'cnn':
            self.model = CNNClassifierModel() # Convolutional Neural Network
        else:
            raise ValueError("Invalid algorithm name. Choose from: 'rf', 'nn', 'cnn'.")

    def train(self, X_train, y_train):
        self.model.train(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
