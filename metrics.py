import numpy as np
from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, roc_curve)


def custom_metrics(y_true: list, y_pred: list) -> dict:

    TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = TP / (TP + FP) if TP + FP else 0  # Positive predictive value
    recall = TP / (TP + FN) if TP + FN else 0  # Sensitivity, True Positive Rate (TPR)
    specificity = TN / (TN + FP) if TN + FP else 0  # Specificity, selectivity, True Negative Rate (TNR)

    G_mean = np.sqrt(recall * specificity)  # Geometric mean of recall and specificity
    F1 = f1_score(y_true, y_pred, zero_division=0)  # F1-measure

    metrics = {"Gmean": G_mean, "F1": F1, "Precision": precision, "Recall": recall, "TP": TP, "TN": TN, "FP": FP,
                   "FN": FN}

    return metrics