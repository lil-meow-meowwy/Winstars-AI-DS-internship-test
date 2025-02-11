from sklearn.ensemble import RandomForestClassifier
from models.mnist_classifier_interface import MnistClassifierInterface

class RandomForestClassifierModel(MnistClassifierInterface):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)