from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, average_precision_score
import numpy as np

def compute_metrics(y_true, y_scores):
    # y_scores: probability of fake
    y_pred = (np.array(y_scores) > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    cm = confusion_matrix(y_true, y_pred)
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc),
        'average_precision': float(ap),
        'confusion_matrix': cm.tolist()
    }
