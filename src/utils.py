from numpy import newaxis
from nltk import download
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

NLTK_DATA_DIR = 'dev/nltk_data'


def normalize(cm):
    return cm.astype('float') / cm.sum(axis=1)[:, newaxis]


count_words = [
    'make',
    'address',
    'all',
    '3d',
    'our',
    'over',
    'remove',
    'internet',
    'order',
    'mail',
    'receive',
    'will',
    'people',
    'report',
    'addresses',
    'free',
    'business',
    'email',
    'you',
    'credit',
    'your',
    'font',
    '000',
    'money'
    'hp',
    'hpl',
    'george',
    '650',
    'lab',
    'labs',
    'telnet',
    '857',
    'data',
    '415',
    '85',
    'technology',
    '1999',
    'parts',
    'pm',
    'direct',
    'cs',
    'meeting',
    'original',
    'project',
    're',
    'edu',
    'table',
    'conference',
]


def text2features(text):
    download('punkt', NLTK_DATA_DIR)
    # download('wordnet', NLTK_DATA_DIR)
    tokens = word_tokenize(text)
    total_nb_words = len(tokens)
    # lemmatizer=WordNetLemmatizer()

    counts = [0 for i in count_words]
    features = [0 for i in range(57)]
    for i in tokens:
        # word = lemmatizer.lemmatize(i.lower())
        word = i.lower()
        if word in count_words:
            idx = count_words.index(word)
            counts[idx] += 1
    for idx, nb in enumerate(counts):
        features[idx] = nb / total_nb_words * 100
    return features
