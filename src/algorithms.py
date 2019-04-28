from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.datasets.base import load_data
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score


class Algorithm(ABC):
    """ Abstract class representing a machine learning algorithm.

    Each subclass must implement these functions:

        - def configurations(self)
    """

    def __init__(self):
        super().__init__()
        self.train_feat = []
        self.train_labl = []
        self.test_feat = []
        self.test_labl = []
        self.initSets()

    def initSets(self):
        """Init the datasets features and labels from the data directory's csv files"""
        features, labels, _ = load_data(".", "spambase.csv")
        self.train_feat, self.test_feat, self.train_labl, self.test_labl = train_test_split(features, labels, test_size=0.20)

    def test(self):
        """Run the algorithm and return the accuracy of its predictions"""
        return [
            {
                'function': func.__name__,
                'accuracy': accuracy_score(func(), self.test_labl)
            }
            for func in self.configurations()]

    @abstractmethod
    def configurations(self):
        """Return an array of functions returning predictions"""
        pass


class NaiveBayes(Algorithm):

    def configurations(self):
        return [self.basic, self.unscaled, self.scaled]

    def basic(self):
        """This is an exemple of an algorithm's most basic implementation"""
        # Create a Gaussian Classifier
        model = GaussianNB()
        # Train the model
        model.fit(self.train_feat, self.train_labl)
        # Predict Output
        predicted = model.predict(self.test_feat)
        print("Predicted Value:", predicted)
        return predicted

    def unscaled(self):
        # Create a Gaussian Classifier
        model = make_pipeline(PCA(n_components=2), GaussianNB())

        # Train the model using pipelined GNB and PCA.
        model.fit(self.train_feat, self.train_labl)

        # Predict Output
        return model.predict(self.test_feat)

    def scaled(self):
        # Create a Gaussian Classifier
        model = make_pipeline(StandardScaler(), PCA(
            n_components=2), GaussianNB())

        # Train the model using pipelined scaling, GNB and PCA.
        model.fit(self.train_feat, self.train_labl)

        # Predict Output
        return model.predict(self.test_feat)


class Svm(Algorithm):
    def configurations(self):
        return [self.basic]

    def basic(self):
        model = svm.SVC(gamma='auto')
        model.fit(self.train_feat, self.train_labl)
        return model.predict(self.test_feat)


class Knn(Algorithm):
    pass


class GradientBoosting(Algorithm):
    pass
