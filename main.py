from src.algorithms import NaiveBayes, Svm
from src.dataset import Dataset

dataset = Dataset()
for algo, name in [
    (NaiveBayes, 'Naive Bayes'),
    (Svm, 'Support Vector Machine')
]:
    instance = algo(dataset)
    print(name, instance.test())
