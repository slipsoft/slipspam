from abc import ABC, abstractmethod
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
import csv


class Algorithm(ABC):

    def __init__(self):
        super().__init__()
        self.model = None
        self.trainingset_features = []
        self.trainingset_labels = []
        self.testset_features = []
        self.testset_labels = []
        self.initSets()

    def initSets(self):
        with open('./Data/spambase.trainingset.csv') as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            for row in csvReader:
                self.trainingset_labels.append(int(row.pop()))
                self.trainingset_features.append(list(map(lambda x: float(x), row)))
        with open('./Data/spambase.testset.csv') as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            for row in csvReader:
                self.testset_labels.append(int(row.pop()))
                self.testset_features.append(list(map(lambda x: float(x), row)))

    @abstractmethod
    def run(self):
        pass


class NaiveBayes(Algorithm):

    def run(self):
        # Create a Gaussian Classifier
        self.model = GaussianNB()

        # Train the model using the training sets
        self.model.fit(self.trainingset_features, self.trainingset_labels)

        # Predict Output
        predicted = self.model.predict(self.testset_features)
        print("Predicted Value:", predicted)
