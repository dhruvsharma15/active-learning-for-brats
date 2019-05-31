import numpy as np


def entropy_rank(pred):
    en = np.zeros(len(pred))

    for i in range(0, len(pred)):
        en[i] = np.sum(-pred[i] * np.log(pred[i]))

    return np.argsort(en)[::-1], en


def uncertain_set(en, nb_annotations):
    return en[0:nb_annotations]


def certain_set(en, thresh, initial_decay_rate, decay_rate):
    # Threshold updating <-- review
    if thresh == None:
        thresh = max(en) - ((max(en) - min(en)) * initial_decay_rate)
    else:
        thresh = thresh + (max(en) - thresh) * decay_rate

    return np.where(en < thresh)[0], thresh


def predictions_max_class(predictions, nb_classes):
    max_class = np.zeros_like(predictions)

    for i in range(0, len(predictions)):
        max_class[i] = 1

    return max_class


def pseudo_label_error(pseudo_samples, true_samples):
    aux = 0
    true = 0
    for i in range(0, len(pseudo_samples)):
        if (np.sum(pseudo_samples[i]*true_samples[i]) != np.sum(true_samples[i])):
            aux += np.sum(pseudo_samples[i])
            true += np.sum(true_samples[i])
    return aux / true
