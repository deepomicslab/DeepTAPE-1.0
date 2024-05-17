# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 18:43:55 2023

@author: timshen2
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from encoding import *
from prediction import *
from models import *
from evaluation import *
from crossvalidation import *
from keras.models import load_model
import random

def train_and_use(data,x_test,y_test,neg_file,pos_file,model_type,model_file):
    # Read the data from the CSV file
    DATA = pd.read_csv(data)
    # Extract the CDR3AA column and convert it to a list
    X = DATA['CDR3AA'].to_list()
    # Extract the positive label column
    Y = DATA['positive']
    # Encode the input data
    x=encoding(len(X),24,X)
    # Create the model object
    model = model_type
    # Set the number of training epochs
    epochs=5
    # Compile the model with RMSprop optimizer, binary cross-entropy loss, and accuracy metric
    model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0001), loss='binary_crossentropy', metrics=['acc'])
    # Train the model
    history = model.fit(x, Y, epochs=epochs, batch_size=32)
    # Save the trained model
    model.save(model_file)
    print('New model saved.')
    # Load the saved model
    model = load_model(model_file)
    # Read the test data
    test_index_X = pd.read_csv(x_test)
    X=np.array(test_index_X.iloc[:,1])
    test_index_Y = pd.read_csv(y_test)
    Y=np.array(test_index_Y.iloc[:,1])
    # Split the test data into negative and positive samples
    neg_test,pos_test,neg_y,pos_y = pos_neg(X,Y,0)
    # Predict the labels for negative and positive test samples
    SNB, SOS_C = pred(neg_file,neg_test, model,2000)
    SPB, SOS_S = pred(pos_file,pos_test, model,2000)
    # Combine the predicted labels
    pre = np.concatenate([SNB,SPB])
    print(pre)
    # Combine the true labels
    Y_test = np.concatenate([neg_y,pos_y])
    # Compute the evaluation metrics
    accuracy, precision, recall, F1_score, roc_auc = evaluation_metric(pre, Y_test, 2,0)
    
    metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'F1_score': F1_score,
    'roc_auc': roc_auc,
    'pre': pre,
    'Y_test': Y_test,
    'SOS_C': SOS_C,
    'SOS_S': SOS_S
    }
    
    return metrics

#--------------------------------------------------------------------------------------------

def train_and_use_with_gene(data,x_test,y_test,neg_file,pos_file,model,model_file):
    # Read the data from the CSV file
    X = pd.read_csv(data)
    # Extract the CDR3AA column and convert it to a list
    X_1 = X['CDR3AA'].to_list()
    # Extract the V_gene column and convert it to a list
    X_2 = X['V_gene'].to_list()
    # Encode the CDR3AA input data
    x_1 = encoding(len(X_1),24, X_1)
    # Encode the V_gene input data
    x_2 = encoding_gene(len(X_2), X_2)
    # Extract the positive label column
    Y = X['positive']
    # Set the number of training epochs
    epochs=5
    # Compile the model with RMSprop optimizer, binary cross-entropy loss, and accuracy metric
    model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0001), loss='binary_crossentropy', metrics=['acc'])
    # Train the model
    history = model.fit([x_1,x_2], Y, epochs=epochs, batch_size=32)
    # Save the trained model
    model.save(model_file)
    print('New model saved.')
    # Load the saved model
    model = load_model(model_file)
    # Read the test data
    test_index_X = pd.read_csv(x_test)
    X=np.array(test_index_X.iloc[:,1])
    test_index_Y = pd.read_csv(y_test)
    Y=np.array(test_index_Y.iloc[:,1])
    # Split the test data into negative and positive samples
    neg_test,pos_test,neg_y,pos_y = pos_neg(X,Y,0)
    # Predict the labels for negative and positive test samples using the model with gene information
    SNB, SOS_C ,DF_N,EDN = pred_with_gene(neg_file,neg_test, model, 0, 1, 2000, 0)
    SPB, SOS_S ,DF_P,EDP = pred_with_gene(pos_file,pos_test, model, 0, 1, 2000, 1)
    # Combine the predicted labels
    pre = np.concatenate([SNB,SPB])
    print(pre)
    # Combine the true labels
    Y_test = np.concatenate([neg_y,pos_y])
    # Compute the evaluation metrics
    accuracy, precision, recall, F1_score, roc_auc = evaluation_metric(pre, Y_test, 2,0)
    
    metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'F1_score': F1_score,
    'roc_auc': roc_auc,
    'pre': pre,
    'DF_N': DF_N,
    'DF_P': DF_P,
    'Y_test': Y_test,
    'EDP': EDP,
    'EDN': EDN
    }

    return metrics

#--------------------------------------------------------------------------------------------


def train_and_use_with_gene_family(data,x_test,y_test,neg_file,pos_file,model,model_file):
    # Read the data from the CSV file
    X = pd.read_csv(data)
    # Extract the CDR3AA column and convert it to a list
    X_1 = X['CDR3AA'].to_list()
    # Extract the V_gene column and convert it to a list
    X_2 = X['V_gene'].to_list()
    # Encode the CDR3AA input data
    x_1 = encoding(len(X_1),24, X_1)
    # Encode the V_gene input data using gene family encoding
    x_2 = encoding_gene_family(len(X_2), X_2)
    # Extract the positive label column
    Y = X['positive']
    # Set the number of training epochs
    epochs=5
    # Compile the model with RMSprop optimizer, binary cross-entropy loss, and accuracy metric
    model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0001), loss='binary_crossentropy', metrics=['acc'])
    # Train the model
    history = model.fit([x_1,x_2], Y, epochs=epochs, batch_size=32)
    # Save the trained model
    model.save(model_file)
    print('New model saved.')
    # Load the saved model
    model = load_model(model_file)
    # Read the test data
    test_index_X = pd.read_csv(x_test)
    X=np.array(test_index_X.iloc[:,1])
    test_index_Y = pd.read_csv(y_test)
    Y=np.array(test_index_Y.iloc[:,1])
    # Split the test data into negative and positive samples
    neg_test,pos_test,neg_y,pos_y = pos_neg(X,Y,0)
    # Predict the labels for negative and positive test samples using the model with gene family information
    SNB, SOS_C ,DF_N, EDN = pred_with_gene_family(neg_file,neg_test, model, 0, 1, 2000, 0)
    SPB, SOS_S ,DF_P, EDP = pred_with_gene_family(pos_file,pos_test, model, 0, 1, 2000, 1)
    # Combine the predicted labels
    pre = np.concatenate([SNB,SPB])
    print(pre)
    # Combine the true labels
    Y_test = np.concatenate([neg_y,pos_y])
    # Compute the evaluation metrics
    accuracy, precision, recall, F1_score, roc_auc = evaluation_metric(pre, Y_test, 2,0)
    
    metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'F1_score': F1_score,
    'roc_auc': roc_auc,
    'pre': pre,
    'DF_N': DF_N,
    'DF_P': DF_P,
    'Y_test': Y_test,
    'EDP': EDP,
    'EDN': EDN
    }

    return metrics

#--------------------------------------------------------------------------------------------


def use_model(model_file,x_test,y_test,neg_file,pos_file):
    # Load the saved model
    model = load_model(model_file)
    # Read the test data
    test_index_X = pd.read_csv(x_test)
    X=np.array(test_index_X.iloc[:,1])
    test_index_Y = pd.read_csv(y_test)
    Y=np.array(test_index_Y.iloc[:,1])
    # Split the test data into negative and positive samples
    neg_test,pos_test,neg_y,pos_y = pos_neg(X,Y,0)
    # Combine the true labels
    Y_test = np.concatenate([neg_y,pos_y])
    # Predict the labels for negative test samples
    SNB, SOS_C = pred(neg_file,neg_test, model,2000)
    # Predict the labels for positive test samples
    SPB, SOS_S = pred(pos_file,pos_test, model,2000)
    # Combine the predicted labels
    pre = np.concatenate([SNB,SPB])
    # Compute the evaluation metrics
    accuracy, precision, recall, F1_score, roc_auc = evaluation_metric(pre, Y_test, 2,0)
    return accuracy, precision, recall, F1_score, roc_auc , pre, Y_test , SOS_C ,SOS_S

#--------------------------------------------------------------------------------------------

def use_model_with_gene(model_file,x_test,y_test,neg_file,pos_file):
    # Load the saved model
    model = load_model(model_file)
    # Read the test data
    test_index_X = pd.read_csv(x_test)
    X=np.array(test_index_X.iloc[:,1])
    test_index_Y = pd.read_csv(y_test)
    Y=np.array(test_index_Y.iloc[:,1])
    
    # Split the test data into negative and positive samples
    neg_test,pos_test,neg_y,pos_y = pos_neg(X,Y,0)

    # Predict the labels for negative test samples using the model with gene information
    SNB, SOS_C ,DF_N,EDN = pred_with_gene(neg_file,neg_test, model, 0, 1, 2000, 0)
    # Predict the labels for positive test samples using the model with gene information
    SPB, SOS_S ,DF_P,EDP = pred_with_gene(pos_file,pos_test, model, 0, 1, 2000, 1)
    # Combine the predicted labels
    pre = np.concatenate([SNB,SPB])
    print(pre)
    # Combine the true labels
    Y_test = np.concatenate([neg_y,pos_y])
    # Compute the evaluation metrics
    accuracy, precision, recall, F1_score, roc_auc = evaluation_metric(pre, Y_test, 2,0)
    return accuracy, precision, recall, F1_score, roc_auc, pre, DF_N, DF_P, Y_test


#--------------------------------------------------------------------------------------------


def use_model_with_gene_family(model_file,x_test,y_test,neg_file,pos_file):
    # Load the saved model
    model = load_model(model_file)
    # Read the test data
    test_index_X = pd.read_csv(x_test)
    X=np.array(test_index_X.iloc[:,1])
    test_index_Y = pd.read_csv(y_test)
    Y=np.array(test_index_Y.iloc[:,1])
    # Split the test data into negative and positive samples
    neg_test,pos_test,neg_y,pos_y = pos_neg(X,Y,0)
    # Predict the labels for negative test samples using the model with gene family information
    SNB, SOS_C ,DF_N, EDN = pred_with_gene_family(neg_file,neg_test, model, 0, 1, 2000, 0)
    # Predict the labels for positive test samples using the model with gene family information
    SPB, SOS_S ,DF_P, EDP = pred_with_gene_family(pos_file,pos_test, model, 0, 1, 2000, 1)
    # Combine the predicted labels
    pre = np.concatenate([SNB,SPB])
    print(pre)
    # Combine the true labels
    Y_test = np.concatenate([neg_y,pos_y])
    # Compute the evaluation metrics
    accuracy, precision, recall, F1_score, roc_auc = evaluation_metric(pre, Y_test, 2,0)
    return accuracy, precision, recall, F1_score, roc_auc ,pre, DF_N, DF_P, Y_test, EDP, EDN