import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, f1_score
from matplotlib import pyplot as plt


def metrics_computation(y_true, y_pred, model, type):
    os.makedirs(os.path.join('result', type), exist_ok=True)
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greys)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_true)))
    plt.xticks(tick_marks, np.unique(y_true))
    plt.yticks(tick_marks, np.unique(y_true))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    for i in range(len(np.unique(y_true))):
        for j in range(len(np.unique(y_true))):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='red', fontsize=15)
    
    metrics = {'Accuracy':[f"{accuracy:.3f}"], "Precision":[f"{precision:.3f}"], 'Recall':[f"{recall:.3f}"], 'F1-score':[f"{f1:.3f}"]}
    metric_df = pd.DataFrame(metrics)
    metric_df.to_csv(f"{os.path.join('result', type)}/{model}_metric.csv")
    plt.savefig(f"{os.path.join('result', type)}/{model}_confusion.png")
    plt.close()