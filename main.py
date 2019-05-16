from src.algorithms import NaiveBayes, Svm, Knn, GradientBoosting, Mpl
from src.dataset import Dataset
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

algos = [
    (NaiveBayes, 'Naive Bayes'),
    (Svm, 'Support Vector Machine'),
    # (Knn, 'K Nearest Neighbours'),
    (GradientBoosting, 'Gradient Boosting'),
    (Mpl, 'Multi-layer Perceptron'),
]
test_size = 0.2
repetition = 5

results = {
    'label': {},
    'accuracy': defaultdict(lambda: []),
    'fit': defaultdict(lambda: []),
    'predict': defaultdict(lambda: []),
}
for i in range(repetition):
    function = 0
    dataset = Dataset(test_size=test_size)
    for algo, name in algos:
        instance = algo(dataset)
        for result in instance.test():
            label = '%s\n(%s)' % (name, result['function'])
            accuracy = result['accuracy'] * 100
            fit = result['fit_duration']
            predict = result['predict_duration']
            print('%s:\n\taccuracy: %9.6f %%\n\tfit:      %9.6f s\n\tpredict:  %9.6f s' % (
                label,
                accuracy,
                fit,
                predict))
            results['label'][function] = label
            results['accuracy'][function].append(accuracy)
            results['fit'][function].append(fit)
            results['predict'][function].append(predict)
            function += 1


labels = results['label'].values()
fitMeans = np.mean(list(results['fit'].values()), axis=1)
predictMeans = np.mean(list(results['predict'].values()), axis=1)
n_groups = len(labels)

# create plot
fig, ax1 = plt.subplots()
y_pos = range(1, n_groups + 1)
bar_width = 0.35
opacity = 0.5

ax1.set_ylabel('Duration (s)')
ax1.bar(y_pos, fitMeans, bar_width,
alpha=opacity,
color='r',
label='Fit')
ax1.bar(y_pos, predictMeans, bar_width,
bottom=fitMeans,
alpha=opacity,
color='g',
label='Predict')
ax1.legend()

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Accuracy (%)')
ax2.boxplot(list(results['accuracy'].values()))

plt.title('Algorithm comparision')
ax1.set_xticklabels(labels)

plt.tight_layout()
plt.show()
