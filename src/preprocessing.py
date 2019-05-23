from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
try:
    from sklearn.feature_selection import VarianceThreshold
except:
    pass  # ceci pour ne pas avoir une erreur d'importation
from sklearn.pipeline import Pipeline
import numpy as np


class Preprocess(BaseEstimator):
    def __init__(self):
        #pca_boi = PCA(n_components=8)
        filtering = VarianceThreshold(threshold=4)
        #self.transformer = Pipeline([("first", filtering),("second", pca_boi)])
        #self.transformer = pca_boi
        self.transformer = filtering

    def fit(self, X, y=None):
        return np.abs(self.transformer.fit(X, y))
        # return self.transformer.fit(X,y)

    def fit_transform(self, X, y=None):
        return np.abs(self.transformer.fit_transform(X, y))
        # return self.transformer.fit_transform(X,y)

    def transform(self, X, y=None):
        return np.abs(self.transformer.transform(X))
        # return self.transformer.transform(X)
