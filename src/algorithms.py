from abc import ABC, abstractmethod
from .dataset import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from time import time


class Algorithm(ABC):
    """Abstract class representing a machine learning algorithm.

    Each subclass must implement these functions:

        - def configurations(self)
    """

    def __init__(self, dataset: Dataset):
        super().__init__()
        self.set = dataset

    def test(self):
        """Run the algorithm and return the accuracy of its predictions"""
        results = []
        for func in self.configurations():
            model = func()
            start = time()
            # Train the model
            model.fit(self.set.train_feat, self.set.train_labl)
            fit_end = time()
            # Predict Output
            predicted = model.predict(self.set.test_feat)
            predict_end = time()
            results.append({
                'function': func.__name__,
                'accuracy': accuracy_score(predicted, self.set.test_labl),
                'fit_duration': fit_end - start,
                'predict_duration': predict_end - fit_end,
                'duration': predict_end - start,
            })
        return results

    @abstractmethod
    def configurations(self):
        """Return an array of functions returning a model to fit"""
        pass


class NaiveBayes(Algorithm):

    def configurations(self):
        return [self.basic, self.scaled]

    def basic(self):
        """This is an exemple of an algorithm's most basic implementation"""
        return GaussianNB()

    def scaled(self):
        return make_pipeline(
            StandardScaler(),
            PCA(n_components=2),
            GaussianNB())


class Svm(Algorithm):
    def configurations(self):
        return [self.basic]

    def basic(self):
        return svm.SVC(gamma='auto')

class Knn(Algorithm):
    def configurations(self):
        return [self.basic]

    def basic(self):
        return KNeighborsClassifier(n_neighbors=5)

class RFC(Algorithm):
    def configurations(self):
        return [self.basic]

    def basic(self):
        return RandomForestClassifier(n_estimators=100)

class DecisionTreeClassifier(Algorithm):

    def configurations(self):
        return [self.basic]

    def basic(self):
        return DTC()

class GradientBoosting(Algorithm):

    def configurations(self):
        return [self.basic, self.scaled]

    def basic(self):
        return GBC(loss='deviance',
            learning_rate=0.3, n_estimators=50)

    def scaled(self):
        return make_pipeline(
            StandardScaler(),
            VarianceThreshold(threshold=.35 * (1 - .35)),
            GBC(loss='deviance',
            learning_rate=0.3, n_estimators=50, max_features=0.9))


class Mpl(Algorithm):
    def configurations(self):
        return [self.scaled]

    def scaled(self):
        clf = MLPClassifier(
            solver='lbfgs',
            alpha=0.1,
            hidden_layer_sizes=(5, 2),
            random_state=1)
        return make_pipeline(
            StandardScaler(),
            clf)
