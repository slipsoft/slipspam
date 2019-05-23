from numpy import newaxis


def normalize(cm):
    return cm.astype('float') / cm.sum(axis=1)[:, newaxis]
