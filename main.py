from src.algorithms import NaiveBayes, Svm, Knn, GradientBoosting, Mpl, RFC, DecisionTreeClassifier
from src.dataset import Dataset
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
from collections import defaultdict
import sys

algos = [
    (NaiveBayes, 'Naive Bayes'),
    (Svm, 'SVM'),
    (Knn, 'KNN'),
    (GradientBoosting, 'Gradient Boosting'),
    (Mpl, 'MLP'),
    (DecisionTreeClassifier, "DecisionTreeClassifier"),
    (RFC, 'RFC')
]

repetition = None
test_size = 0.2
maxExec = 1000

if len(sys.argv) >= 2:
    try:
        repetition = int(sys.argv[1])
    except ValueError:
        repetition = 5
else:
    repetition = 5

results = {
    'label': {},
    'accuracy': defaultdict(lambda: []),
    'fit': defaultdict(lambda: []),
    'predict': defaultdict(lambda: []),
    'confusion': defaultdict(lambda: []),
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
            confusion = result['confusion']
            print('%s:\n\taccuracy: %9.6f %%\n\tfit:      %9.6f s\n\tpredict:  %9.6f s\n\tconfusion: \n%s' % (
                label,
                accuracy,
                fit,
                predict,
                confusion))
            results['label'][function] = label
            results['accuracy'][function].append(accuracy)
            results['fit'][function].append(fit)
            results['predict'][function].append(predict)
            results['confusion'][function].append(confusion)
            function += 1


labels = list(results['label'].values())
fitMeans = np.mean(list(results['fit'].values()), axis=1)
predictMeans = np.mean(list(results['predict'].values()), axis=1)
confusionMeans = [np.mean(m, axis=0) for m in list(results['confusion'].values())]
conf_labl = ['spam', 'non-spam']
n_groups = len(labels)

# create plot
fig, ax1 = plt.subplots()
y_pos = range(1, n_groups + 1)
bar_width = 0.35
opacity = 0.5

# duration axis with 2 bars
ax1.set_ylabel('Duration (s)')
ax1.bar(y_pos, predictMeans, bar_width,
bottom=fitMeans,
alpha=opacity,
color='g',
label='Predict')
ax1.bar(y_pos, fitMeans, bar_width,
alpha=opacity,
color='b',
label='Fit')
ax1.legend(loc=2)  # add the legend in the top left corner

# instantiate a second axes that shares the same x-axis
# accuracy axis with a boxplot
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy (%)')
ax2.boxplot(list(results['accuracy'].values()))

plt.title('Algorithm comparision (%d executions)' % repetition)
ax1.set_xticklabels(labels)  # set ticks and labels on ax1 (otherwise it does not work)
ax1.tick_params(axis='x', which='major', labelsize=7)  # reduce size of x labels
plt.tight_layout()

for idx, m in enumerate(confusionMeans):
    plt.figure()
    plt.title(labels[idx])
    sn.heatmap(pd.DataFrame(m, conf_labl, conf_labl), annot=True)

plt.show()
