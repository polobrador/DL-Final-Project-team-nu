import numpy as np
from sklearn.metrics import roc_auc_score


def competition_macro_roc_auc(y_true, y_pred):
    class_aucs = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i]) > 0:
            class_aucs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    return np.mean(class_aucs)
