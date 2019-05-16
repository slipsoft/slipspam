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
        accuracy = result['accuracy'] * 100
        duration = result['duration']
        print('%s:\n\taccuracy: %9.6f %%\n\tduration: %9.6f s' % (
            label,
            accuracy,
            duration))
        results['label'].append(label)
        results['accuracy'].append(accuracy)
        results['duration'].append(duration)


labels = results['label']
n_groups = len(labels)

# create plot
fig, ax1 = plt.subplots()
y_pos = np.arange(n_groups)
bar_width = 0.35
opacity = 0.5

ax1.set_ylabel('Accuracy (%)')
ax1.bar(y_pos, results['accuracy'], bar_width,
alpha=opacity,
color='b',
label='Accuracy')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Duration (s)')
ax2.bar(y_pos + bar_width, results['duration'], bar_width,
alpha=opacity,
color='g',
label='Duration')

plt.title('Algorithm comparision')
plt.xlabel('Algorithms')
plt.xticks(y_pos + bar_width, labels)

fig.legend()
plt.show()
