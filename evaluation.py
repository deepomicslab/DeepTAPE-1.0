# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 21:23:14 2023

@author: timshen2
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def evaluation_metric(model_result, label, num_classes,u):
    
    # Compute the Receiver Operating Characteristic (ROC) curve and the Area Under the Curve (AUC)
    fpr, tpr, thresholds = roc_curve(label, model_result, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    # Determine the optimal threshold for classification based on the ROC curve
    maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
    threshold = thresholds[maxindex]
    
    # Classify the model results based on the threshold
    pre = []
    for data in model_result:
        if data>=threshold:
            pre.append(1)
        else:
            pre.append(0)
    pre = np.array(pre).flatten()
    
    # Convert the model results and labels into flattened arrays
    model_result = np.array(model_result).flatten()
    label = np.array(label).flatten()
    
    # Compute the confusion matrix and the evaluation metrics
    mask = (label >= 0) & (label < num_classes)
    confusion_matrix = np.bincount(
        num_classes * label[mask].astype(int) +
        pre[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    TP = confusion_matrix[1][1]
    TN = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    accuracy = (TP+TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = (2 * precision * recall) / (precision + recall)
    
    # If u is not 0, do not show the ROC curve plot
    if u==0:
        plt.figure()
        plt.plot(fpr, tpr, 'r-o', label='ROC curve (area=%0.4f)' % roc_auc)
        plt.grid()
        plt.legend(loc="lower right")
        plt.show()
    
    # Return the evaluation metrics
    return accuracy, precision, recall, F1_score, roc_auc
