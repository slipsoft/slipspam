from numpy import newaxis
from nltk import download
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np

NLTK_DATA_DIR = 'dev/nltk_data'


def normalize(cm):
    return cm.astype('float') / cm.sum(axis=1)[:, newaxis]


COUNT_WORDS = [
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
    'money',
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

COUNT_CHARS = [
    ';',
    '(',
    '[',
    '!',
    '$',
    '#',
]


def text2features(text: str):
    download('punkt', NLTK_DATA_DIR, quiet=True)
    # download('wordnet', NLTK_DATA_DIR, quiet=True)
    # lemmatizer=WordNetLemmatizer()

    tokens = word_tokenize(text)
    total_nb_words = len(tokens)
    total_nb_chars = len(text)
    count_words = [0 for i in COUNT_WORDS]
    count_chars = [0 for i in COUNT_CHARS]
    capital_sequences = [0]
    id_seq = 0
    prev_is_lower = False
    features = [0 for i in range(57)]

    for i in tokens:
        # word = lemmatizer.lemmatize(i.lower())
        word = i.lower()
        if word in COUNT_WORDS:
            idx = COUNT_WORDS.index(word)
            count_words[idx] += 1

    for char in text:
        if char in COUNT_CHARS:
            idx = COUNT_CHARS.index(char)
            count_chars[idx] += 1
        if char.isalpha():
            if char.isupper():
                if prev_is_lower:
                    id_seq += 1
                    capital_sequences.append(0)
                capital_sequences[id_seq] += 1
                prev_is_lower = False
            else:
                prev_is_lower = True

    for idx, nb in enumerate(count_words):
        features[idx] = nb / total_nb_words * 100
    for idx, nb in enumerate(count_chars):
        features[idx + 48] = nb / total_nb_chars * 100
    features[54] = np.average(capital_sequences)
    features[55] = np.max(capital_sequences)
    features[56] = np.sum(capital_sequences)
    return features


vtext2features = np.vectorize(text2features, otypes=[np.ndarray])


def trans_label(label):
    if label == 0:
        return 'Non-spam'
    elif label == 1:
        return '! SPAM !'
    else:
        return 'Unknown?'


vtrans_label = np.vectorize(trans_label)
