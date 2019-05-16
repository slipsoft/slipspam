from src.algorithms import NaiveBayes, Svm, Knn, GradientBoosting, Mpl
from src.dataset import Dataset
import matplotlib.pyplot as plt
import numpy as np

algos = [
    (NaiveBayes, 'Naive Bayes'),
    (Svm, 'Support Vector Machine'),
    # (Knn, 'K Nearest Neighbours'),
    (GradientBoosting, 'Gradient Boosting'),
    (Mpl, 'Multi-layer Perceptron'),
]
test_size = 0.2

results = {
    'label': [],
    'accuracy': [],
    'duration': [],
}
dataset = Dataset(test_size=test_size)
for algo, name in algos:
    instance = algo(dataset)
    for result in instance.test():
        label = '%s\n(%s)' % (name, result['function'])
        print('%s:\n\taccuracy: %9.6f %%\n\tduration: %9.6f s' % (label,
            result['accuracy'] * 100,
            result['duration']))
        results['label'].append(label)
        results['accuracy'].append(result['accuracy'])
        results['duration'].append(result['duration'])


objects = results['label']
y_pos = np.arange(len(objects))

plt.bar(y_pos, results['accuracy'], align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Algorithm comparision')
plt.show()
