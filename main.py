#!/usr/bin/env python3
"""SlipSapm.

Usage:
  slipspam bench [options]
  slipspam predict [options] <file>
  slipspam -h | --help
  slipspam --version

Options:
  -h --help                      Show this screen.
  --version                      Show version.
  -e <nb>, --executions=<nb>     Number of executions [default: 5].
  -t <size>, --test-size=<size>  Proportion of the dataset to use for the tests [default: 0.2].
"""
from src.algorithms import NaiveBayes, Svm, Knn, GradientBoosting, Mpl, RFC
from src.benchmark import run_bench
from docopt import docopt

args = docopt(__doc__, version='SlipSpam 1.0-beta.1')
# print(args)

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
repetition = int(args['--executions'])
test_size = float(args['--test-size'])

if args['bench']:
    run_bench(algos, repetition, test_size)
