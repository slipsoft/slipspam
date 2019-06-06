from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
from sklearn.datasets.base import load_data
import pandas as pd


class Dataset:
    """Class representing a dataset.

    The data set is split into 4 main sets:
    - train_feat: features of each entry on which to train
    - train_labl: label (result) of each entry on which to train
    - test_feat:  features of each entry to test
    - test_labl:  label (result) of each entry to test
    """

    def __init__(self, trainset="spambase.csv", testset: str = None, test_size=0.20, drop_cols=[26, 27]):
        super().__init__()
        self.train_file = 'data/' + trainset
        self.test_file = None if testset is None else 'data/' + testset
        self.test_size = test_size
        self.drop_cols = drop_cols
        self.train_feat = []
        self.train_labl = []
        self.test_feat = []
        self.test_labl = []
        # self.train_index = []
        # self.test_index = []
        self.initSets()

    def initSets(self):
        """Init the datasets features and labels from the data directory's csv files"""
        data_frame = pd.read_csv(self.train_file, header=None)
        data_frame = data_frame.drop(columns=self.drop_cols)
        features = data_frame.iloc[:, :-1].values
        labels = data_frame.iloc[:, -1].values
        if self.test_file is None:
            self.train_feat, self.test_feat, self.train_labl, self.test_labl = train_test_split(features, labels, test_size=self.test_size)
        else:
            data_frame = pd.read_csv(self.test_file, header=None)
            data_frame = data_frame.drop(columns=self.drop_cols)
            self.train_feat, _, self.train_labl, _ = train_test_split(features, labels, test_size=self.test_size)
            features = data_frame.iloc[:, :-1].values
            labels = data_frame.iloc[:, -1].values
            _, self.test_feat, _, self.test_labl = train_test_split(features, labels, test_size=self.test_size)
        # kfold = KFold(n_splits=3)
        # self.train_index, self.test_index = kfold.split(features,labels)
