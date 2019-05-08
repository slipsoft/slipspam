from abc import ABC, abstractmethod
from .dataset import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier as GBC


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
        return [{
            'function': func.__name__,
            'accuracy': accuracy_score(func(), self.set.test_labl)
        } for func in self.configurations()]

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
        model.fit(self.set.train_feat, self.set.train_labl)
        # Predict Output
        predicted = model.predict(self.set.test_feat)
        return predicted

    def unscaled(self):
        # Create a Gaussian Classifier
        model = make_pipeline(PCA(n_components=2), GaussianNB())

        # Train the model using pipelined GNB and PCA.
        model.fit(self.set.train_feat, self.set.train_labl)

        # Predict Output
        return model.predict(self.set.test_feat)

    def scaled(self):
        # Create a Gaussian Classifier
        model = make_pipeline(
            StandardScaler(),
            PCA(n_components=2),
            GaussianNB())

        # Train the model using pipelined scaling, GNB and PCA.
        model.fit(self.set.train_feat, self.set.train_labl)

        # Predict Output
        return model.predict(self.set.test_feat)


class Svm(Algorithm):
    def configurations(self):
        return [self.basic]

    def basic(self):
        model = svm.SVC(gamma='auto')
        model.fit(self.set.train_feat, self.set.train_labl)
        return model.predict(self.set.test_feat)


class Knn(Algorithm):
    pass


class GradientBoosting(Algorithm):

    def configurations(self):
        return [self.basic]

    def basic(self):
        model_init = None
        model = GBC(loss='deviance',
            learning_rate=0.3,
            n_estimators=100,
            subsample=1.0,
            criterion='friedman_mse',
            min_samples_split=2, min_samples_leaf=1,
            min_weight_fraction_leaf=0., max_depth=3,
            min_impurity_decrease=0., min_impurity_split=None, 
            init=None, random_state=None, max_features=None, 
            verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto', validation_fraction=0.1, n_iter_no_change=None, tol=1e-4)
        model.fit(self.set.train_feat, self.set.train_labl)
        return model.predict(self.set.test_feat)


class Mpl(Algorithm):
    def configurations(self):
        return [self.scaled]

    def scaled(self):
        clf = MLPClassifier(
            solver='lbfgs',
            alpha=0.1,
            hidden_layer_sizes=(5, 2),
            random_state=1)
        model = make_pipeline(
            StandardScaler(),
            clf)
        model.fit(self.set.train_feat, self.set.train_labl)
        return model.predict(self.set.test_feat)
