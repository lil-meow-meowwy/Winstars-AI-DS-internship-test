from sklearn.ensemble import RandomForestClassifier
from models.mnist_classifier_interface import MnistClassifierInterface

class RandomForestClassifierModel(MnistClassifierInterface):
    """
    Random Forest classifier for MNIST dataset.
    Implements the `MnistClassifierInterface`.
    """
    def __init__(self):
        """
        Initialize the Random Forest model with 100 trees.
        """
        self.model = RandomForestClassifier(n_estimators=100)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)