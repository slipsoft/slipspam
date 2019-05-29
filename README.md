# ![SlipSpam](docs/slipspam.png)

Ultimate spam detector

## Presentation

Slipspam consists of a command line program that can do 3 things:
- benchmarks: [`bench`](#bench-command)
- predictions: [`predict`](#predict-command)
- email data minig: [`parse`](#parse-command)

### `bench` command

```
bin/slipspam bench [options]
```

This command runs a full benchmark of every implemented algorithms over a dataset.

Here are the results of the latest benchark over the [spambase dataset](http://archive.ics.uci.edu/ml/datasets/spambase):

![spambase algorithms benchmark chart](docs/benchmark5.png)
![spambase interleaved confusion matrix](docs/confusion-matrix1.png)

And here over a [spam dataset found on kaggle](https://www.kaggle.com/karthickveerakumar/spam-filter):

![algorithms benchmark chart](docs/benchmark-emails1.png)
![interleaved confusion matrix](docs/confusion-matrix-emails1.png)

### `predict` command

```
bin/slipspam predict [options] (<email-text> | --in-text=<file> | --in-feat=<file>)
```

This command enables you to predict if a mail is spam or not. It can also run the algorithm on a full dataset via the `--in-feat` option.
![predict command](./docs/predict-command.png)

### `parse` command

```
bin/slipspam parse --in=<file> --out=<file>
```

This command convert a dataset of text mails into a dataset of features corresponding to the ones our algorithm is trained on.
![parse command](./docs/parse-command.png)

## Getting Started

These instructions will get you a copy of the project up and running on your
local machine for development and testing purposes. See deployment for notes on
how to deploy the project on a live system.

### Prerequisites

-   `python3`
-   `pip3`
-   Tkinter

#### Debian

    sudo apt install python3 python3-pip python3-tk

### Installing

This will install and activate a virtual python environment in the dev directory.

#### Windows

    .\activate

#### Unix

    . activate

### Running

#### Benchmarks

    bin/slipspam bench [-e <nb executions>]

exemple:

    bin/slipspam bench -e 5

####

### Deactivating

To deactivate the virtual environment, simply type `deactivate`

## Deployment

_not deployable yet..._

## Documentation

### CLI

You can get the CLI documentation by running the following command:

    bin/slipspam --help

## Built With

-   [scikit-learn](https://scikit-learn.org/stable/) - Machine learning library

## Authors

-   **Anass ELANBARI** - [anassCoding](https://github.com/anassCoding)
-   **Gwendal LE STANGUENNEC** - [GwendalLES](https://github.com/GwendalLES)
-   **Etienne LELOUËT** - [etienne-lelouet](https://github.com/etienne-lelouet)
-   **Nicolas PEUGNET** - [n-peugnet](https://github.com/n-peugnet)

See also the list of [contributors](https://github.com/slipsoft/slipspam/contributors)
who participated in this project.

