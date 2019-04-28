from sklearn.model_selection import train_test_split
from sklearn.datasets.base import load_data


class Dataset:
    """Class representing a dataset.

    The data set is split into 4 main sets:
    - train_feat: features of each entry on which to train
    - train_labl: label (result) of each entry on which to train
    - test_feat:  features of each entry to test
    - test_labl:  label (result) of each entry to test
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
