# -*- coding: utf-8 -*-
"""
Created on Sun May 12 22:10:37 2024

@author: yd123
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from encoding import *
from prediction import *
from scipy.stats import chi2_contingency
import random
import scipy.stats as stats
from scipy.stats import entropy
from scipy.stats import pearsonr
import os

#--------------------------------------------------------------------------------------------


def random_files(folder, n):
    """
    Selects a random sample of n files from the given folder.
    
    Args:
        folder (str): The path to the folder containing the files.
        n (int): The number of files to select.
    
    Returns:
        list: A list of n randomly selected file names.
    """
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    return random.sample(files, n)


#--------------------------------------------------------------------------------------------


def frequency_difference_VGeneFam(df1_file, df_compared):
    """
    Computes the Pearson correlation coefficient between the frequency distributions of V gene families
    between two datasets.
    
    Args:
        df1_file (str): The path to the folder containing the first dataset.
        df_compared (str): The path to the file containing the second dataset.
    
    Returns:
        float: The Pearson correlation coefficient.
    """
    # Read in the second dataset
    df2 = pd.read_csv(df_compared)
    
    # Get the list of files in the first dataset folder
    files = os.listdir(df1_file)
    num_png = len(files)
    
    # Select a random sample of 10 files if there are more than 10, otherwise use all files
    if num_png > 10:
        file_name = random_files(df1_file, 10)
    else:
        file_name = files
    
    # Initialize a DataFrame to store the data from the first dataset
    df1 = pd.DataFrame()
    
    # Iterate through the selected files and process the data
    for name in file_name:
        pt = pd.read_csv(df1_file + name, sep=',', skiprows=1, header=None)
        pt = pt.iloc[:, [1, 2]]
        pt = pt.dropna(axis=0)
        
        u = 0
        mix = []
        for i in pt.iloc[:, 0]:
            # Clean the data and create a new list
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
        if len(ptc) >= 2000:
            ptc = ptc[:2000]
        ptc = ptc.drop('counts', axis=1)
        ptc[['CDR3AA', 'V_gene']] = ptc['Mix'].str.split('_', expand=True)
        df1 = pd.concat([df1, ptc], axis=0)
    
    # Extract the V gene family frequencies from the two datasets
    df1 = df1.iloc[:, 2]
    df2 = df2.iloc[:, 2]
    freq1 = df1.value_counts(normalize=True)
    freq2 = df2.value_counts(normalize=True)
    
    # Convert the frequencies to DataFrames and merge them
    freq1_df = pd.DataFrame({'freq1': freq1})
    freq2_df = pd.DataFrame({'freq2': freq2})
    freq_df = pd.concat([freq1_df, freq2_df], axis=1).fillna(0)
    
    # Calculate the Pearson correlation coefficient
    pearson_corr, _ = pearsonr(freq_df['freq1'], freq_df['freq2'])
    
    return pearson_corr


#--------------------------------------------------------------------------------------------


def frequency_difference_VGene(df1_file, df_compared):
    """
    Computes the Pearson correlation coefficient between the frequency distributions of V genes
    between two datasets.
    
    Args:
        df1_file (str): The path to the folder containing the first dataset.
        df_compared (str): The path to the file containing the second dataset.
    
    Returns:
        float: The Pearson correlation coefficient.
    """
    # Read in the second dataset
    df2 = pd.read_csv(df_compared)
    
    # Get the list of files in the first dataset folder
    files = os.listdir(df1_file)
    num_png = len(files)
    
    # Select a random sample of 10 files if there are more than 10, otherwise use all files
    if num_png > 10:
        file_name = random_files(df1_file, 10)
    else:
        file_name = files
    
    # Initialize a DataFrame to store the data from the first dataset
    df1 = pd.DataFrame()
    
    # Define a dictionary to map V gene names to numerical indices
    A = {'TRBV12-1': 21, 'TRBV24-2': 22, 'TRBV6-8': 23, 'TRBV29-1': 24, 'TRBV1': 25, 'TRBV14': 26, 'TRBV6-1': 27, 'TRBV4-1': 28, 'TRBV7-8': 29, 'TRBV6-3': 30, 'TRBV27': 31, 'TRBV19': 32, 'TRBV12-4': 33, 'TRBV23-2': 34, 'TRBV7-3': 35, 'TRBV5-4': 36, 'TRBV11-3': 37, 'TRBV4-3': 38, 'TRBV5-7': 39, 'TRBV6-5': 40, 'TRBV25-1': 41, 'TRBV10-3': 42, 'TRBV6-6': 43, 'TRBV16': 44, 'TRBV5-6': 45, 'TRBV6-4': 46, 'TRBV12-3': 47, 'TRBV10-2': 48, 'TRBV13': 49, 'TRBV3-1': 50, 'TRBV15': 51, 'TRBV7-4': 52, 'TRBV21-1': 53, 'TRBV7-7': 54, 'TRBV5-5': 55, 'TRBV6-7': 56, 'TRBV7-6': 57, 'TRBV11-1': 58, 'TRBV9': 59, 'TRBV12-5': 60, 'TRBV24-1': 61, 'TRBV5-1': 62, 'TRBV2': 63, 'TRBV10-1': 64, 'TRBV7-9': 65, 'TRBV28': 66, 'TRBV6-9': 67, 'TRBV11-2': 68, 'TRBV18': 69, 'TRBV3-2': 70, 'TRBV6-2': 71, 'TRBV30': 72, 'TRBV4-2': 73, 'TRBV7-2': 74, 'TRBV5-8': 75, 'TRBV20-1': 76, 'TRBV26-2': 77}
    
    # Iterate through the selected files and process the data
    for name in file_name:
        pt = pd.read_csv(df1_file + name, sep=',', skiprows=1, header=None)
        pt = pt.iloc[:, [1, 3]]
        pt = pt.dropna(axis=0)
        
        u = 0
        mix = []
        num_freq = 2000
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
                    G = 'TRBV' + g[:g.index('0')] + g[g.index('0')+1:]
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
        ptc[['CDR3AA', 'V_gene']] = ptc['Mix'].str.split('_', expand=True)
        df1 = pd.concat([df1, ptc], axis=0)
    
    # Extract the V gene frequencies from the two datasets
    df1 = pd.DataFrame(df1.iloc[:, 2])
    df2 = pd.DataFrame(df2.iloc[:, 2])
    freq1 = df1.value_counts(normalize=True)
    freq2 = df2.value_counts(normalize=True)
    
    # Convert the frequencies to DataFrames and merge them
    freq1_df = pd.DataFrame({'freq1': freq1})
    freq2_df = pd.DataFrame({'freq2': freq2})
    freq_df = pd.concat([freq1_df, freq2_df], axis=1).fillna(0)
    
    # Calculate the Pearson correlation coefficient
    pearson_corr, _ = pearsonr(freq_df['freq1'], freq_df['freq2'])
    
    return pearson_corr


#--------------------------------------------------------------------------------------------


def Ratio(df1_file):
    """
    Computes the ratio of the absolute difference between the Pearson correlation coefficients
    of the V gene family frequencies and the V gene frequencies between two datasets.
    
    Args:
        df1_file (str): The path to the folder containing the first dataset.
    
    Returns:
        list: A list containing the ratios.
    """
    Ratio = []
    
    # Compute the ratio for V gene family frequencies
    ratio = abs(frequency_difference_VGeneFam(df1_file, 'Standard/HI_VF.csv') /
                frequency_difference_VGeneFam(df1_file, 'Standard/SLE_VF.csv'))
    Ratio.append(ratio)
    
    # Compute the ratio for V gene frequencies
    ratio = abs(frequency_difference_VGene(df1_file, 'Standard/HI_V.csv') /
                frequency_difference_VGene(df1_file, 'Standard/SLE_V.csv'))
    
    Ratio.append(ratio)
    return Ratio
    
    