# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 20:58:44 2023

@author: timshen2
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
import sklearn
from sklearn.cluster import KMeans

def pos_neg(X,Y,u):
    
    # Initialize empty lists for positive and negative sequences
    pos = []
    neg = []
    
    # Initialize empty lists for positive and negative labels
    pos_y = []
    neg_y = []
    
    # If u equals 1, meaning only positive and negative sequences are returned
    if u == 1:
        # Iterate over X and Y, and separate positive and negative sequences
        for i in range(0, len(X)):
            if Y[i] == 0:
                neg.append(X[i])
            else:
                pos.append(X[i])
                
        # Convert the lists of positive and negative sequences into numpy arrays and return them
        neg = np.array(neg)
        pos = np.array(pos)
        return neg, pos
    
    # If u is not equal to 1, meaning both the sequences and labels are returned
    else:
        # Iterate over X and Y, and separate positive and negative sequences and labels
        for i in range(0, len(X)):
            if Y[i] == 0:
                neg.append(X[i])
                neg_y.append(Y[i])
            else:
                pos.append(X[i])
                pos_y.append(Y[i])
                
        # Convert the lists of positive and negative sequences and labels into numpy arrays and return them
        neg = np.array(neg)
        pos = np.array(pos)
        return neg, pos, neg_y, pos_y


#--------------------------------------------------------------------------------------------
     
    
def get_data(x_neg, x_pos, file_destination):
    # Concatenate negative and positive data
    DATA = pd.concat([x_neg, x_pos])
    
    # Drop duplicated sequences, keep only the first occurrence
    DATA.drop_duplicates(subset=['CDR3AA'], keep='first', inplace=True)
    
    # Remove any sequences labeled as "positive" (if any)
    DATA = DATA.drop(DATA[DATA['CDR3AA'] == 'positive'].index)
    
    # Shuffle the data
    DATA = DATA.sample(frac=1).reset_index(drop=True)
    
    # Split data into positive and negative sequences
    POS = DATA[DATA['positive'] == 1]
    NEG = DATA[DATA['positive'] == 0]
    
    # Keep only as many negative sequences as there are positive sequences
    NEG = NEG[:len(POS)]
    
    # Concatenate positive and negative sequences
    DATA = pd.concat([POS, NEG])
    
    # Shuffle the data again
    DATA = DATA.sample(frac=1).reset_index(drop=True)
    
    # Extract sequences and labels
    X = DATA['CDR3AA'].to_list()
    Y = DATA['positive']
    
    # Save data to a CSV file
    DATA.to_csv(file_destination)
    
    # Return sequences, labels, and the full data
    return X, Y, DATA


#--------------------------------------------------------------------------------------------
     
    
def get_data_gene(x_neg, x_pos):
    
    # concatenate negative and positive samples
    DATA = pd.concat([x_neg, x_pos])
   
    # remove duplicate sequences
    DATA.drop_duplicates(subset=['Mix'], keep='first', inplace=True)
   
    # remove rows with positive mixtures
    DATA = DATA.drop(DATA[DATA['Mix'] == 'positive'].index)
    
    # shuffle the data
    DATA = DATA.sample(frac=1).reset_index(drop=True)
    
    # extract positive and negative samples
    POS = DATA[DATA['positive'] == 1]
    NEG = DATA[DATA['positive'] == 0]
   
    # make negative and positive samples equal
    NEG = NEG[:len(POS)]
   
    # combine negative and positive samples
    DATA = pd.concat([POS, NEG])
   
    # shuffle the data again
    DATA = DATA.sample(frac=1).reset_index(drop=True)
    
    # split the 'Mix' column into two columns: 'CDR3AA' and 'V_gene'
    DATA['CDR3AA'], DATA['V_gene'] = DATA['Mix'].str.split('_', 1).str
    
    # remove the 'Mix' column
    DATA = DATA.drop('Mix', axis=1)
    
    # get the CDR3AA and V_gene sequences and the labels
    X_1 = DATA['CDR3AA'].to_list()
    X_2 = DATA['V_gene'].to_list()
    Y = DATA['positive']
    
    
    return X_1, X_2, Y, DATA



#--------------------------------------------------------------------------------------------
     
    
def crossvalidation(train_X, train_Y):
    
    # Create a KFold cross-validation object with 5 splits
    kf = KFold(n_splits=5)
    
    # Shuffle the data
    train_X, train_Y = sklearn.utils.shuffle(train_X, train_Y)
    
    # Convert the data to numpy arrays
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    t = 0
    
    # Split the data into training and test sets using the KFold object
    for train_index, test_index in kf.split(train_X):
        t+=1
        
        # Get the training and test sets for the current split
        X_train, X_test = train_X[train_index], train_X[test_index]
        Y_train, Y_test = train_Y[train_index], train_Y[test_index]
        
        # Convert the data to pandas dataframes and save them to csv files
        x_train,x_test,y_train,y_test = pd.DataFrame(X_train),pd.DataFrame(X_test),pd.DataFrame(Y_train),pd.DataFrame(Y_test)
        x_train.to_csv('{}.csv'.format(str(t)+'X_train'))
        x_test.to_csv('{}.csv'.format(str(t)+'X_test'))
        y_train.to_csv('{}.csv'.format(str(t)+'Y_train'))
        y_test.to_csv('{}.csv'.format(str(t)+'Y_test'))
        
