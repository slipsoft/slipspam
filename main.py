#!/usr/bin/env python3

from src.algorithms import NaiveBayes, Svm, Knn, GradientBoosting, Mpl, RFC
from src.benchmark import run_bench
import sys


algos = [
    (NaiveBayes, 'NB'),
    (Svm, 'SVM'),
    (Knn, 'KNN'),
    (GradientBoosting, 'Gb'),
    (Mpl, 'MLP'),
    # (DecisionTreeClassifier, "DecisionTreeClassifier"),
    # (LinearDiscriminantAnalysis, "LinearDiscriminant Analysis"),
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

run_bench(algos, repetition, test_size)
