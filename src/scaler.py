from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler


class scaler(BaseEstimator):

    def fit(self, X):
        scaler = StandardScaler()
        fit1 = scaler.fit()
