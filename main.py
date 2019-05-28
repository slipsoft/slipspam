#!/usr/bin/env python3
"""SlipSapm.

Usage:
  slipspam bench [-v] [--executions=<nb>] [--test-size=<size>]
  slipspam predict [-v] [-t] (<email-text> | --in-text=<file> | --in-feat=<file>)
  slipspam -h | --help
  slipspam --version

Options:
  -h --help                    Show this screen.
  --version                    Show version.
  -e <nb>, --executions=<nb>   Number of executions [default: 5].
  --test-size=<size>           Proportion of the dataset to use for the tests [default: 0.8].
  -v                           Verbose
  -t                           Translated for human readability
  --in-text=<file>             Path to a file containing the text of a mail to classify.
  --in-feat=<file>             Path to a file containing a csv of features compliant with spambase.
"""
from src.algorithms import NaiveBayes, Svm, Knn, GradientBoosting, Mpl, Rfc
from src.dataset import Dataset
from src.benchmark import run_bench
from src.utils import text2features, trans_label
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
    (Rfc, 'RFC')
]
repetition = int(args['--executions'])
test_size = float(args['--test-size'])
verbose = args['-v']

if args['bench']:
    run_bench(algos, repetition, test_size)
elif args['predict']:
    if args['<email-text>']:
        text = args['<email-text>']
        features = [text2features(text)]
    if args['--in-text']:
        f = open(args['--in-text'], "r")
        text = f.read()
        features = [text2features(text)]
    if args['--in-feat']:
        raise NotImplementedError('This option have not been implemented yet...')
    dataset = Dataset(test_size=test_size)
    if verbose:
        print(features)
    results = Rfc(dataset).predict('optimize', features)
    if args['-t']:
        print([trans_label(i) for i in results])
    else:
        print(results)
