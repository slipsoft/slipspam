from src.algorithms import NaiveBayes, Svm, Knn, GradientBoosting, Mpl
from src.dataset import Dataset

algos = [
    (NaiveBayes, 'Naive Bayes'),
    (Svm, 'Support Vector Machine'),
    # (Knn, 'K Nearest Neighbours'),
    (GradientBoosting, 'Gradient Boosting'),
    (Mpl, 'Multi-layer Perceptron'),
]
test_size = 0.2
dataset = Dataset(test_size=test_size)
for algo, name in algos:
    instance = algo(dataset)
    for result in instance.test():
        print("%s (%s):\n\taccuracy: %9.6f %%\n\tduration: %9.6f s" % (name,
            result['function'],
            result['accuracy'] * 100,
            result['duration']))
