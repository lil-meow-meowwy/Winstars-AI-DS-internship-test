from models.random_forest_classifier import RandomForestClassifierModel
from models.feed_forward_nn_classifier import FeedForwardNNClassifierModel
from models.cnn_classifier import CNNClassifierModel

class MnistClassifier:
    def __init__(self, algorithm: str):
        if algorithm == 'rf':
            self.model = RandomForestClassifierModel()
        elif algorithm == 'nn':
            self.model = FeedForwardNNClassifierModel()
        elif algorithm == 'cnn':
            self.model = CNNClassifierModel()
        else:
            raise ValueError("Invalid algorithm name. Choose from: 'rf', 'nn', 'cnn'.")

    def train(self, X_train, y_train):
        self.model.train(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
