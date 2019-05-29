#!/usr/bin/env python3
"""SlipSapm.

Usage:
  slipspam bench [-v] [--drop-col=<nb>...] [--executions=<nb>] [--test-size=<size>] [--dataset=<file>]
  slipspam predict (<email-text> | --in-text=<file> | --in-feat=<file>) [-v] [-t] [--drop-col=<nb>...]
                   [--dataset=<file>]
  slipspam parse --in=<file> --out=<file>
  slipspam -h | --help
  slipspam --version

Options:
  -h --help                    Show this screen.
  --version                    Show version.
  -v                           Verbose.
  -d <nb>, --drop-col=<nb>     Drop a column from the dataset (can be repeted) [default: 26 27].
  -e <nb>, --executions=<nb>   Number of executions [default: 5].
  --test-size=<size>           Proportion of the dataset to use for the tests [default: 0.2].
  --dataset=<file>             Path to a dataset (from data/) [default: spambase.csv].
  -t                           Translated for human readability.
  --in-text=<file>             Path to a file containing the text of a mail to classify.
  --in-feat=<file>             Path to a file containing a csv of features compliant with spambase.
  -i <file>, --in=<file>       Path to input file must be a csv with to columns: [text, spam]
  -o <file>, --out=<file>      Path to output file.
"""
from src.algorithms import NaiveBayes, Svm, Knn, GradientBoosting, Mpl, Rfc
from src.dataset import Dataset
from src.benchmark import run_bench
from src.utils import text2features, vtext2features, vtrans_label, vint
import numpy as np
import pandas as pd
from tqdm import tqdm
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
data_file = args['--dataset']
drop_cols = vint(args['--drop-col'])
verbose = args['-v']

if args['bench']:
    run_bench(algos,
        repetition=repetition,
        test_size=test_size,
        file=data_file,
        drop_cols=drop_cols)
elif args['predict']:
    if args['<email-text>']:
        text = args['<email-text>']
        features = [np.delete(text2features(text), drop_cols)]
    if args['--in-text']:
        f = open(args['--in-text'], "r")
        text = f.read()
        f.close()
        features = [np.delete(text2features(text), drop_cols)]
    if args['--in-feat']:
        data_frame = pd.read_csv(args['--in-feat'], header=None).drop(columns=drop_cols)
        features = data_frame.iloc[:, :-1].values
    dataset = Dataset(test_size=test_size, file=data_file, drop_cols=drop_cols)
    if verbose:
        print(features)
    results = Rfc(dataset).predict('optimize', features)
    if args['-t']:
        print(vtrans_label(results).tolist())
    else:
        print(results)
elif args['parse']:
    in_file = args['--in']
    tqdm.pandas()
    email_df = pd.read_csv(in_file)
    features = email_df.progress_apply(lambda x: text2features(x['text']), axis=1, result_type='expand')
    features['spam'] = email_df['spam']
    features.to_csv(args['--out'], header=False, index=False)
