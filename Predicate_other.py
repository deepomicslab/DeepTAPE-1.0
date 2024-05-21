# -*- coding: utf-8 -*-
"""
Created on Mon May 13 18:21:29 2024

@author: yd123
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from encoding import *
from prediction import *
from SPCC import *
from keras.models import load_model

def predict_other_autoimmune_disease(file_path, model_dir):
    """
    Predict autoimmune disease using DeepTAPE models.

    Parameters:
    file_path (str): The path to the folder containing the data to be predicted.
    model_dir (str): The directory where the trained DeepTAPE models are stored.

    Returns:
    pd.DataFrame: A DataFrame containing the prediction results, including the filename, score, and result (Positive/Negative).
    """
    name = os.listdir(file_path)

    print('**************Diagnostic prediction starts.**************')

    # Threshold values obtained from cross-validation
    ThreShold_A_VF = [0.4861144423484802]
    ThreShold_A_V = [0.4925624430179596]
    ThreShold_A = [0.4786588251590729]

    # Create a DataFrame to store the results
    result_df = pd.DataFrame(columns=['Filename', 'Score', 'Result'])

    # Calculate the ratio between the Pearson correlation coefficients
    r = Ratio(file_path)

    if r[0] <= 1:
        model = load_model(os.path.join(model_dir, 'DeepTAPE_A_VF.h5'))
        score, socres_of_seq, DF = pred_by_DeepTAPE_A_VF_indepence(file_path, name, model, 1, 2, 2000)
        for i, s in enumerate(score):
            if s > ThreShold_A_VF[0]:
                result = "Positive"
            else:
                result = "Negative"
            new_row = {'Filename': name[i], 'Score': s, 'Result': result}
            result_df = pd.concat([result_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)


    elif r[1] <= 1:
        model = load_model(os.path.join(model_dir, 'DeepTAPE_A_V.h5'))
        score, socres_of_seq, DF = pred_with_gene_indepence(file_path, name, model, 1, 3, 2000)
        for i, s in enumerate(score):
            if s > ThreShold_A_V[0]:
                result = "Positive"
            else:
                result = "Negative"
            new_row = {'Filename': name[i], 'Score': s, 'Result': result}
            result_df = pd.concat([result_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)


    else:
        model = load_model(os.path.join(model_dir, 'DeepTAPE_A.h5'))
        score, socres_of_seq = pred_by_DeepTAPE_A_indepence(file_path, name, model, 'aminoAcid', 'amino_acid', 2000)
        for i, s in enumerate(score):
            if s > ThreShold_A[0]:
                result = "Positive"
            else:
                result = "Negative"
            new_row = {'Filename': name[i], 'Score': s, 'Result': result}
            result_df = pd.concat([result_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)


    print('**************The diagnostic prediction results are as follows:**************')
    print(result_df)


    return result_df

def predict_sle_by_DeepTAPE_A_VF(file_path, model_dir):
    """
    Predicts SLE diagnosis using the DeepTAPE-A_VF model.
    
    Args:
        file_path (str): Path to the input file.
        
    Returns:
        pd.DataFrame: A DataFrame containing the prediction results.
    """
    print('**************Diagnostic prediction starts.**************')
    # Threshold values obtained from cross-validation
    ThreShold_A_VF = [0.4861144423484802]
    ThreShold_A_V = [0.4925624430179596]
    ThreShold_A = [0.4786588251590729]

    model = load_model(os.path.join(model_dir, 'DeepTAPE_A_VF.h5'))
    
    name = os.listdir(file_path)
    score, scores_of_seq, DF = pred_by_DeepTAPE_A_VF_indepence(file_path, name, model, 1, 2, 2000)
    
    result_df = pd.DataFrame()
    for i, s in enumerate(score):
        if s > ThreShold_A_VF[0]:
            result = "Positive"
        else:
            result = "Negative"
        new_row = {'Filename': name[i], 'Score': s, 'Result': result}
        result_df = pd.concat([result_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
        
    print('**************The diagnostic prediction results are as follows:**************')
    print(result_df)
    
    return result_df
