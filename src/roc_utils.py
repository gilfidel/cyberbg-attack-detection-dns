from sklearn.metrics import auc, f1_score, roc_curve, precision_recall_curve, average_precision_score
from matplotlib import pyplot as plt

def _plot(x, y, title, xlabel, ylabel, baseline=([0,1], [0,1])):
    plt.figure()
    plt.plot(x, y, color='darkorange', lw=5, label=title)
    plt.plot(*baseline, color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, threshold = roc_curve(y_true, y_pred_proba[:,1])
    roc_auc_score = auc(fpr, tpr)
    print(roc_auc_score)
    title = f'ROC curve (area = {roc_auc_score:.4f})'

    _plot(fpr, tpr, title, 'False Positive Rate', 'True Positive Rate', baseline=([0,1],[0,1]))

def plot_precision_recall_curve(y_true, y_pred_proba, beta=1.0):
    positive_proba = y_pred_proba[:,1]
    precision, recall, thresholds = precision_recall_curve(y_true, positive_proba)
    plt.figure()
    ap_score = average_precision_score(y_true, positive_proba)
    print(ap_score)
    title = f'Precision-Recall curve (ap-score = {ap_score:.3f}))'

    _plot(precision, recall, title, 'Precision', 'Recall', baseline=([0,1],[0.5,0.5]))
