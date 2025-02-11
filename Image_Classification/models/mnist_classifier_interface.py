from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        """Train the model on the given data."""
        pass

    @abstractmethod
    def predict(self, X_test):
        """Make predictions using the trained model."""
        pass
