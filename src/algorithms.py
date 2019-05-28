from abc import ABC, abstractmethod
from .dataset import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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
            predicted = self.predict_classic(model)
            predict_end = time()
            results.append({
                'function': func.__name__,
                'accuracy': accuracy_score(self.set.test_labl, predicted),
                'fit_duration': fit_end - start,
                'predict_duration': predict_end - fit_end,
                'duration': predict_end - start,
                'confusion': confusion_matrix(self.set.test_labl, predicted),
            })
        return results

    def predict_classic(self, model):
        """Return the predictions of a model over the test features"""
        return model.predict(self.set.test_feat)

    def predict_weight(self, model):
        """Return the predictions of a model over the test features but giving more weight to the non-spam label"""
        return [1 if p[1] > 0.94 else 0 for p in model.predict_proba(self.set.test_feat)]

    @abstractmethod
    def configurations(self):
        """Return an array of functions returning a model to fit"""
        pass


class NaiveBayes(Algorithm):

    def configurations(self):
        return [self.bernouilli]

    def bernouilli(self):
        return BernoulliNB(alpha=0.5, binarize=0.2)

    def separateScale(self):
        ct = ColumnTransformer([
            ("wordFreq", StandardScaler(), slice(0, 48)),
            ("charFreq", StandardScaler(), slice(48, 54)),
            ("continuousCapital", StandardScaler(), slice(54, 56)),
            ("longestCapital", StandardScaler(), [56])
        ])
        return make_pipeline(
            ct,
            BernoulliNB(alpha=0.5, binarize=0.2)
        )


class Svm(Algorithm):
    def configurations(self):
        return []

    def basic(self):
        return svm.SVC(gamma='auto')


class Knn(Algorithm):
    def configurations(self):
        return [self.separateScaleSelected]

    def basic(self):
        return KNeighborsClassifier(n_neighbors=5)

    def scaledUniform(self):
        return make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5, weights="uniform")
        )

    def scaledDistance(self):
        return make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5, weights="distance")
        )

    def separateScale(self):
        ct = ColumnTransformer([
            ("wordFreq", StandardScaler(), slice(0, 48)),
            ("charFreq", StandardScaler(), slice(48, 54)),
            ("continuousCapital", StandardScaler(), slice(54, 56)),
            ("longestCapital", StandardScaler(), [56])
        ])
        return make_pipeline(
            ct,
            KNeighborsClassifier(n_neighbors=5, weights="distance")
        )

    def separateScaleSelected(self):
        ct = ColumnTransformer([
            ("wordFreq", StandardScaler(), slice(0, 48)),
            ("charFreq", StandardScaler(), slice(48, 54)),
            ("continuousCapital", StandardScaler(), slice(54, 56)),
            ("longestCapital", StandardScaler(), [56])
        ])
        return make_pipeline(
            ct,
            PCA(),
            KNeighborsClassifier(n_neighbors=5, weights="distance")
        )


# class LinearDiscriminantAnalysis(Algorithm) :
#     def configurations(self):
#         return [self.basic]

#     def basic(self):
#         return LDA()

# class DecisionTreeClassifier(Algorithm):

#     def configurations(self):
#         return [self.basic]

#     def basic(self):
#         return DTC()


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


class RFC(Algorithm):
    def configurations(self):
        return [self.basic, self.optimize]

    def basic(self):
        return RandomForestClassifier(n_estimators=10)

    def optimize(self):
        return RandomForestClassifier(n_estimators=100, n_jobs=5)
