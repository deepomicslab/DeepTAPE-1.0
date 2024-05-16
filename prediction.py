# -*- coding: utf-8 -*-
"""
Created on Sun May 12 01:06:37 2024

@author: yd123
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from encoding import *

#--------------------------------------------------------------------------------------------
def pred_by_DeepTAPE_A_VF_indepence(upper_file, file_name, model, AA_column, gene_column, num_freq):
    # Initialize empty lists to store the results
    score = []
    socres_of_seq = pd.DataFrame()
    PRE = []
    AA = []
    VF = []
    
    # Iterate through the file names
    for name in file_name:
        print(upper_file + name)
        
        # Read the CSV file
        pt = pd.read_csv(upper_file + name, sep=',', skiprows=1, header=None)
        
        # Select the relevant columns
        pt = pt.iloc[:, [AA_column, gene_column]]
        pt = pt.dropna(axis=0)
        
        # Clean the data and create a new DataFrame
        u = 0
        mix = []
        for i in pt.iloc[:, 0]:
            if 24 >= len(str(i)) >= 10 and i[0] == 'C' and i[-1] == 'F' and ('*' not in i) and ('x' not in i):
                g = str(pt.iloc[u, 1])
                if g[5] != '0':
                    g_c = g[5:]
                else:
                    g_c = g[6:]
                G = 'TRBV' + g_c
                mix.append(i + '_' + G)
            u += 1
        pt_cleaned = {'Mix': mix}
        ptc = pd.DataFrame(pt_cleaned)
        ptc = ptc.value_counts(ascending=False).rename_axis('Mix').reset_index(name='counts')
        ptc.sort_values(by='counts')
        if len(ptc) >= num_freq:
            ptc = ptc[:num_freq]
        ptc = ptc.drop('counts', axis=1)
        ptc[['CDR3AA', 'V_gene']] = ptc['Mix'].str.split('_', n=1, expand=True)
        
        # Create a dictionary for valid gene families
        valid_dict = {'TRBV4': 21, 'TRBV16': 22, 'TRBV19': 23, 'TRBV27': 24, 'TRBV2': 25, 'TRBV28': 26, 'TRBV25': 27, 'TRBV21': 28, 'TRBV10': 29, 'TRBV15': 30, 'TRBV5': 31, 'TRBV3': 32, 'TRBV14': 33, 'TRBV26': 34, 'TRBV20': 35, 'TRBV12': 36, 'TRBV9': 37, 'TRBV13': 38, 'TRBV29': 39, 'TRBV1': 40, 'TRBV23': 41, 'TRBV11': 42, 'TRBV7': 43, 'TRBV18': 44, 'TRBV30': 45, 'TRBV6': 46, 'TRBV24': 47}
        
        # Filter the DataFrame to include only valid gene families
        ptc = ptc[ptc['V_gene'].isin(valid_dict.keys())]
        ptc = ptc.drop('Mix', axis=1)
        
        # Encode the data for the model
        X_1 = ptc['CDR3AA'].to_list()
        X_2 = ptc['V_gene'].to_list()
        x_1 = encoding(len(X_1), 24, X_1)
        x_2 = encoding_gene_family(len(X_2), X_2)
        print(x_2)
        print(x_1)
        
        # Make predictions with the model
        predictions = model.predict([x_1, x_2])
        E = float(predictions.mean())
        score.append(E)
        
        # Concatenate the predictions with the original data
        seq = ptc['CDR3AA']
        pre = pd.DataFrame(predictions)
        SOS = pd.concat([seq, pre], axis=1)
        socres_of_seq = pd.concat([socres_of_seq, SOS])
        
        # Append the predictions, amino acid sequences, and gene families to the respective lists
        predict = predictions.reshape(-1)
        pred = predict.tolist()
        PRE = PRE + pred
        aa = []
        v = []
        for a in ptc['CDR3AA']:
            aa.append(a)
        AA = AA + aa
        for vf in ptc['V_gene']:
            v.append(vf)
        VF = VF + v
    
    # Create a DataFrame with the final results
    df = {'AA': AA, 'VGeneFam': VF, 'predictions': PRE}
    DF = pd.DataFrame(df)
    return score, socres_of_seq, DF

#--------------------------------------------------------------------------------------------
def pred_by_DeepTAPE_A_V_indepence(upper_file, file_name, model, AA_column, gene_column, num_freq):
    score = []
    
    # Create a dictionary to map gene family names to indices
    A = {'TRBV12-1': 21, 'TRBV24-2': 22, 'TRBV6-8': 23, 'TRBV29-1': 24, 'TRBV1': 25, 'TRBV14': 26, 'TRBV6-1': 27, 'TRBV4-1': 28, 'TRBV7-8': 29, 'TRBV6-3': 30, 'TRBV27': 31, 'TRBV19': 32, 'TRBV12-4': 33, 'TRBV23-2': 34, 'TRBV7-3': 35, 'TRBV5-4': 36, 'TRBV11-3': 37, 'TRBV4-3': 38, 'TRBV5-7': 39, 'TRBV6-5': 40, 'TRBV25-1': 41, 'TRBV10-3': 42, 'TRBV6-6': 43, 'TRBV16': 44, 'TRBV5-6': 45, 'TRBV6-4': 46, 'TRBV12-3': 47, 'TRBV10-2': 48, 'TRBV13': 49, 'TRBV3-1': 50, 'TRBV15': 51, 'TRBV7-4': 52, 'TRBV21-1': 53, 'TRBV7-7': 54, 'TRBV5-5': 55, 'TRBV6-7': 56, 'TRBV7-6': 57, 'TRBV11-1': 58, 'TRBV9': 59}
    
    socres_of_seq = pd.DataFrame()
    PRE = []
    AA = []
    VF = []
    
    # Iterate through the file names
    for name in file_name:
        # Read the CSV file
        pt = pd.read_csv(upper_file + name, sep=',', skiprows=1, header=None)
        pt = pt.iloc[:, [AA_column, gene_column]]
        pt = pt.dropna(axis=0)
        
        u = 0
        mix = []
        for i in pt.iloc[:, 0]:
            # Clean the data and create a new list
            if 24 >= len(str(i)) >= 10 and i[0] == 'C' and i[-1] == 'F' and ('*' not in i) and ('x' not in i):
                g = str(pt.iloc[u, 1])
                if len(g) >= 10:
                    if g[5] != '0':
                        g_c = g[5:8] + g[9]
                    else:
                        g_c = g[6:8] + g[9]
                    G = 'TRBV' + g_c
                elif '0' in g:
                    G = 'TRBV' + g[:g.index('0')] + g[g.index('0') + 1:]
                else:
                    G = 'TRBV' + g
                if G in A:
                    mix.append(i + '_' + G)
            u += 1
        
        pt_cleaned = {'Mix': mix}
        ptc = pd.DataFrame(pt_cleaned)
        ptc = ptc.value_counts(ascending=False).rename_axis('Mix').reset_index(name='counts')
        ptc.sort_values(by='counts')
        if len(ptc) >= num_freq:
            ptc = ptc[:num_freq]
        ptc = ptc.drop('counts', axis=1)
        ptc[['CDR3AA', 'V_gene']] = ptc['Mix'].str.split('_', n=1, expand=True)
        
        # Filter the DataFrame to include only valid gene families
        ptc = ptc[ptc['V_gene'].isin(A.keys())]
        ptc = ptc.drop('Mix', axis=1)
        
        # Encode the data for the model
        X_1 = ptc['CDR3AA'].to_list()
        X_2 = ptc['V_gene'].to_list()
        x_1 = encoding(len(X_1), 24, X_1)
        x_2 = encoding_gene(len(X_2), X_2)
        print(name)
        
        # Make predictions with the model
        predictions = model.predict([x_1, x_2])
        E = float(predictions.mean())
        score.append(E)
        
        # Concatenate the predictions with the original data
        seq = ptc['CDR3AA']
        pre = pd.DataFrame(predictions)
        SOS = pd.concat([seq, pre], axis=1)
        socres_of_seq = pd.concat([socres_of_seq, SOS])
        
        # Append the predictions, amino acid sequences, and gene families to the respective lists
        predict = predictions.reshape(-1)
        pred = predict.tolist()
        PRE = PRE + pred
        aa = []
        v = []
        for a in ptc['CDR3AA']:
            aa.append(a)
        AA = AA + aa
        for vf in ptc['V_gene']:
            v.append(vf)
        VF = VF + v
    
    # Create a final DataFrame with the results
    df = {'AA': AA, 'VGene': VF, 'predictions': PRE}
    DF = pd.DataFrame(df)
    return score, socres_of_seq, DF


#--------------------------------------------------------------------------------------------
def pred_by_DeepTAPE_A_indepence(upper_file, file_name, model, col_name, col_name2, num_freq):
    score = []
    socres_of_seq = pd.DataFrame()
    
    # Iterate through the file names
    for name in file_name:
        # Initialize the amino acid sequence list
        AA = []
        
        # Read the CSV file and select the appropriate column
        pt = pd.read_csv(upper_file + name, sep=',')
        if col_name in pt.columns:
            pt = pt.loc[:, col_name]
        elif col_name2 in pt.columns:
            pt = pt.loc[:, col_name2]
        
        # Clean the data and append the valid sequences to the AA list
        for i in pt:
            if 24 >= len(str(i)) >= 10 and i[0] == 'C' and i[-1] == 'F' and ('*' not in i) and ('x' not in i):
                AA.append(i)
        
        # Create a new DataFrame with the cleaned data
        pt_cleaned = {'CDR3AA': AA}
        ptc = pd.DataFrame(pt_cleaned)
        
        # Count the frequency of each sequence and filter the top num_freq sequences
        ptc = ptc.value_counts(ascending=False).rename_axis('CDR3AA').reset_index(name='counts_aa')
        ptc.sort_values(by='counts_aa')
        if len(ptc) >= num_freq:
            ptc = ptc[:num_freq]
        
        # Encode the data for the model
        X = ptc['CDR3AA'].to_list()
        x = encoding(len(X), 24, X)
        print(x)
        print(type(x)) 
        # Make predictions with the model
        predictions = model.predict(x)
        E = float(predictions.mean())
        score.append(E)
        
        # Concatenate the predictions with the original data
        seq = ptc['CDR3AA']
        pre = pd.DataFrame(predictions)
        SOS = pd.concat([seq, pre], axis=1)
        socres_of_seq = pd.concat([socres_of_seq, SOS])
    
    return score, socres_of_seq


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from encoding import *
from models import *
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Lambda, dot, Activation, concatenate


def pred(upper_file, file_name, model, num_freq):
    
    # list to store the predicted scores for each file
    score = []
    
    # dataframe to store the predicted scores for each sequence
    scores_of_seq = pd.DataFrame()  
    for name in file_name:
        
        # list to store the cleaned sequences for each file
        AA = []  
        if name.endswith('.csv'):
            sep = ','
        else:
            sep = '\t'
        pt = pd.read_csv(upper_file + name)

        
        # get the 8th column of the file
        pt = pt.iloc[:,0]  
        
        
        for i in pt:
            if 24 >= len(str(i)) >= 10 and i[0] == 'C' and i[-1] == 'F' and ('*' not in i) and ('x' not in i):
                
                # only keep the sequences with length between 10 and 24, start with "C" and end with "F", and have no "*" or "x"
                AA.append(i)  
                
        pt_cleaned = {'CDR3AA': AA}
        ptc = pd.DataFrame(pt_cleaned)
        ptc = ptc.value_counts(ascending=False).rename_axis('CDR3AA').reset_index(name='counts_aa')
        ptc.sort_values(by='counts_aa')
        
        # only keep the top `num_freq` most frequent sequences
        ptc = ptc[:num_freq]  
        X = ptc['CDR3AA'].to_list()
        
        # convert the cleaned sequences to numerical representation
        x = encoding(len(X), 24, X)  
        
        # use the input model to make predictions
        predictions = model.predict(x)  
        
        # compute the average prediction score
        E = float(predictions.mean()) 
        score.append(E)
        seq = ptc['CDR3AA']
        pre = pd.DataFrame(predictions)
        sos = pd.concat([seq, pre], axis=1)
        scores_of_seq = pd.concat([scores_of_seq, sos])
        
        
    return score, scores_of_seq





#--------------------------------------------------------------------------------------------

def pred_with_gene(upper_file,file_name,model,AA_column,gene_column,num_freq, c): #有attention

    score=[]
    socres_of_seq = pd.DataFrame()
    PRE = []
    AA = []
    VF = []

    # Loop through each input file
    for name in file_name:
        if name.endswith('.csv'):
            sep = ','
        else:
            sep = '\t'
        pt = pd.read_csv(upper_file + name)

        pt=pt.iloc[:,[AA_column,gene_column]]
        u=0
        mix = []

        # Select only valid CDR3AA sequences from input file
        for i in pt.iloc[:,0]:
            if 24 >= len(str(i)) >= 10 and i[0] == 'C' and i[-1] == 'F' and ('*' not in i) and ('x' not in i):
                g = pt.iloc[u,1]
                g_c = g[:g.index('*')]

                # Handle V gene format
                if '/' in g_c:
                    g_c = g_c[:g_c.index('/')] + g_c[-2:]
                mix.append(i+'_'+g_c)
            u+=1

        # Create a cleaned and reduced dataframe with only frequent peptides
        pt_cleaned = {'Mix':mix}
        ptc = pd.DataFrame(pt_cleaned)
        ptc = ptc.value_counts(ascending=False).rename_axis('Mix').reset_index(name='counts')
        ptc.sort_values(by='counts')
        ptc = ptc[:num_freq]
        ptc = ptc.drop('counts',axis = 1)
        ptc[['CDR3AA', 'V_gene']] = ptc['Mix'].str.split('_', n=1, expand=True)
        ptc = ptc.drop('Mix',axis = 1)
        valid_dict = {'TRBV12-1': 21, 'TRBV24-2': 22, 'TRBV6-8': 23, 'TRBV29-1': 24, 'TRBV1': 25, 'TRBV14': 26, 'TRBV6-1': 27, 'TRBV4-1': 28, 'TRBV7-8': 29, 'TRBV6-3': 30, 'TRBV27': 31, 'TRBV19': 32, 'TRBV12-4': 33, 'TRBV23-2': 34, 'TRBV7-3': 35, 'TRBV5-4': 36, 'TRBV11-3': 37, 'TRBV4-3': 38, 'TRBV5-7': 39, 'TRBV6-5': 40, 'TRBV25-1': 41, 'TRBV10-3': 42, 'TRBV6-6': 43, 'TRBV16': 44, 'TRBV5-6': 45, 'TRBV6-4': 46, 'TRBV12-3': 47, 'TRBV10-2': 48, 'TRBV13': 49, 'TRBV3-1': 50, 'TRBV15': 51, 'TRBV7-4': 52, 'TRBV21-1': 53, 'TRBV7-7': 54, 'TRBV5-5': 55, 'TRBV6-7': 56, 'TRBV7-6': 57, 'TRBV11-1': 58, 'TRBV9': 59, 'TRBV12-5': 60, 'TRBV24-1': 61, 'TRBV5-1': 62, 'TRBV2': 63, 'TRBV10-1': 64, 'TRBV7-9': 65, 'TRBV28': 66, 'TRBV6-9': 67, 'TRBV11-2': 68, 'TRBV18': 69, 'TRBV3-2': 70, 'TRBV6-2': 71, 'TRBV30': 72, 'TRBV4-2': 73, 'TRBV7-2': 74, 'TRBV5-8': 75, 'TRBV20-1': 76, 'TRBV26-2': 77, 'TRBV5-3': 78, 'TRBV12-2':79,'TRBV20-2':80}
        ptc = ptc[ptc['V_gene'].isin(valid_dict.keys())]
        # Encode peptides and genes for prediction
        X_1 = ptc['CDR3AA'].to_list()
        X_2 = ptc['V_gene'].to_list()    
        x_1 = encoding(len(X_1),24,X_1)
        x_2 = encoding_gene(len(X_2),X_2)

        # Predict binding affinity scores and record results
        predictions=model.predict([x_1,x_2])

        E=float(predictions.mean())
        score.append(E)
        seq = ptc['CDR3AA']
        pre = pd.DataFrame(predictions)
        SOS = pd.concat([seq,pre],axis = 1)
        socres_of_seq = pd.concat([socres_of_seq,SOS])
        predict = predictions.reshape(-1)
        pred = predict.tolist()
        PRE = PRE + pred
        aa = []
        v = []
        
        for a in ptc['CDR3AA']:
            aa.append(a)
        AA = AA + aa
        for vf in ptc['V_gene']:
            v.append(vf)
        VF = VF + v
    df = {'AA':AA,'VGene':VF,'predictions':PRE}
    DF = pd.DataFrame(df)
    if c == 1:
        extreme_DF = DF.sort_values(by='predictions', ascending=False).iloc[:100]
    else:
        extreme_DF = DF.sort_values(by='predictions', ascending=False).tail(100)
    return score, socres_of_seq ,DF, extreme_DF






#--------------------------------------------------------------------------------------------


def pred_with_gene_family(upper_file,file_name,model,AA_column,gene_column,num_freq, c): #有attention
    score=[]
    socres_of_seq = pd.DataFrame()
    PRE = []
    AA = []
    VF = []
    for name in file_name:
        if name.endswith('.csv'):
            sep = ','
        else:
            sep = '\t'
        pt = pd.read_csv(upper_file + name)
        pt=pt.iloc[:,[AA_column,gene_column]]
        u=0
        mix = []
        for i in pt.iloc[:,0]:
            if 24 >= len(str(i)) >= 10 and i[0] == 'C' and i[-1] == 'F' and ('*' not in i) and ('x' not in i):
                g = pt.iloc[u,1]
                g_c = g[:g.index('*')]
                if '-' in g_c:
                    g_c = g_c[:g_c.index('-')]
                if '/' in g_c:
                    g_c = g_c[:g_c.index('/')]
                mix.append(i+'_'+g_c)
            u+=1
        pt_cleaned = {'Mix':mix}
        ptc = pd.DataFrame(pt_cleaned)
        ptc = ptc.value_counts(ascending=False).rename_axis('Mix').reset_index(name='counts')
        ptc.sort_values(by='counts')
        ptc = ptc[:num_freq]
        ptc = ptc.drop('counts',axis = 1)
        ptc[['CDR3AA', 'V_gene']] = ptc['Mix'].str.split('_', n=1, expand=True)
        ptc = ptc.drop('Mix',axis = 1)
        X_1 = ptc['CDR3AA'].to_list()
        X_2 = ptc['V_gene'].to_list()    
        x_1 = encoding(len(X_1),24,X_1)
        x_2 = encoding_gene_family(len(X_2),X_2)
        print(name)
        # Predict binding affinity scores and record results
        predictions=model.predict([x_1,x_2])
        E=float(predictions.mean())
        score.append(E)
        seq = ptc['CDR3AA']
        pre = pd.DataFrame(predictions)
        SOS = pd.concat([seq,pre],axis = 1)
        socres_of_seq = pd.concat([socres_of_seq,SOS])
        
        predict = predictions.reshape(-1)
        pred = predict.tolist()
        PRE = PRE + pred
        aa = []
        v = []
        for a in ptc['CDR3AA']:
            aa.append(a)
        AA = AA + aa
        for vf in ptc['V_gene']:
            v.append(vf)
        VF = VF + v
    df = {'AA':AA,'VGeneFam':VF,'predictions':PRE}
    DF = pd.DataFrame(df)
    if u == 1:
        extreme_DF = DF.sort_values(by='predictions', ascending=False).iloc[:100]
    else:
        extreme_DF = DF.sort_values(by='predictions', ascending=False).tail(100)
    return score, socres_of_seq ,DF,extreme_DF


#--------------------------------------------------------------------------------------------


def pred_with_gene_family_indepence(upper_file,file_name,model,AA_column,gene_column,num_freq):
    score=[]
    socres_of_seq = pd.DataFrame()
    PRE = []
    AA = []
    VF = []
    
    for name in file_name:
        if name.endswith('.csv'):
            sep = ','
        else:
            sep = '\t'
        pt = pd.read_csv(upper_file + name, sep=sep, skiprows=1, header=None)
        pt=pt.iloc[:,[AA_column,gene_column]]
        pt = pt.dropna(axis=0)
        u=0
        mix = []
        for i in pt.iloc[:,0]:
            if 24 >= len(str(i)) >= 10 and i[0] == 'C' and i[-1] == 'F' and ('*' not in i) and ('x' not in i):
                g = str(pt.iloc[u,1])
                if g[5] != '0':
                    g_c = g[5:]
                else:
                    g_c = g[6:]
                G = 'TRBV'+g_c
                mix.append(i+'_'+G)
            u+=1
        pt_cleaned = {'Mix':mix}
        ptc = pd.DataFrame(pt_cleaned)
        ptc = ptc.value_counts(ascending=False).rename_axis('Mix').reset_index(name='counts')
        ptc.sort_values(by='counts')
        if len(ptc) >= num_freq:
            ptc = ptc[:num_freq]
        ptc = ptc.drop('counts',axis = 1)
        ptc[['CDR3AA', 'V_gene']] = ptc['Mix'].str.split('_', n=1, expand=True)
        valid_dict = {'TRBV4': 21, 'TRBV16': 22, 'TRBV19': 23, 'TRBV27': 24, 'TRBV2': 25, 'TRBV28': 26, 'TRBV25': 27, 'TRBV21': 28, 'TRBV10': 29, 'TRBV15': 30, 'TRBV5': 31, 'TRBV3': 32, 'TRBV14': 33, 'TRBV26': 34, 'TRBV20': 35, 'TRBV12': 36, 'TRBV9': 37, 'TRBV13': 38, 'TRBV29': 39, 'TRBV1': 40, 'TRBV23': 41, 'TRBV11': 42, 'TRBV7': 43, 'TRBV18': 44, 'TRBV30': 45, 'TRBV6': 46, 'TRBV24': 47}
        ptc = ptc[ptc['V_gene'].isin(valid_dict.keys())]
        ptc = ptc.drop('Mix',axis = 1)
        X_1 = ptc['CDR3AA'].to_list()
        X_2 = ptc['V_gene'].to_list()    
        x_1 = encoding(len(X_1),24,X_1)
        x_2 = encoding_gene_family(len(X_2),X_2)
        print(name)
        predictions=model.predict([x_1,x_2])
        E=float(predictions.mean())
        score.append(E)
        seq = ptc['CDR3AA']
        pre = pd.DataFrame(predictions)
        SOS = pd.concat([seq,pre],axis = 1)
        socres_of_seq = pd.concat([socres_of_seq,SOS])
        
        predict = predictions.reshape(-1)
        pred = predict.tolist()
        PRE = PRE + pred
        aa = []
        v = []
        for a in ptc['CDR3AA']:
            aa.append(a)
        AA = AA + aa
        for vf in ptc['V_gene']:
            v.append(vf)
        VF = VF + v
    df = {'AA':AA,'VGeneFam':VF,'predictions':PRE}
    DF = pd.DataFrame(df)
    return score, socres_of_seq, DF