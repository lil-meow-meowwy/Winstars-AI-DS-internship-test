from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    """
    Abstract base class for MNIST classifiers.
    All classifiers must implement the `train` and `predict` methods.
    """
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the model on the given training data.
        :param X_train: Training data (features).
        :param y_train: Training labels.
        """
        pass

    @abstractmethod
    def predict(self, X_test):
        """
        Predict labels for the given test data.
        :param X_test: Test data (features).
        :return: Predicted labels.
        """
        pass