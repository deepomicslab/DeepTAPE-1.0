# -*- coding: utf-8 -*-
"""
Created on Sat May 11 18:18:51 2024

@author: yd123
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# A function to encode gene sequences based on a predefined mapping
def encoding_gene(number,X):

    # Create an array of zeros to store the encoded sequences
    x = np.zeros([number,1])

    # Create a list of unique genes
    gene = list(set(X))

    # Define a mapping of each gene to a numeric code
    nuc_d = {'TRBV12-1': 21, 'TRBV24-2': 22, 'TRBV6-8': 23, 'TRBV29-1': 24, 'TRBV1': 25, 'TRBV14': 26, 'TRBV6-1': 27, 'TRBV4-1': 28, 'TRBV7-8': 29, 'TRBV6-3': 30, 'TRBV27': 31, 'TRBV19': 32, 'TRBV12-4': 33, 'TRBV23-2': 34, 'TRBV7-3': 35, 'TRBV5-4': 36, 'TRBV11-3': 37, 'TRBV4-3': 38, 'TRBV5-7': 39, 'TRBV6-5': 40, 'TRBV25-1': 41, 'TRBV10-3': 42, 'TRBV6-6': 43, 'TRBV16': 44, 'TRBV5-6': 45, 'TRBV6-4': 46, 'TRBV12-3': 47, 'TRBV10-2': 48, 'TRBV13': 49, 'TRBV3-1': 50, 'TRBV15': 51, 'TRBV7-4': 52, 'TRBV21-1': 53, 'TRBV7-7': 54, 'TRBV5-5': 55, 'TRBV6-7': 56, 'TRBV7-6': 57, 'TRBV11-1': 58, 'TRBV9': 59, 'TRBV12-5': 60, 'TRBV24-1': 61, 'TRBV5-1': 62, 'TRBV2': 63, 'TRBV10-1': 64, 'TRBV7-9': 65, 'TRBV28': 66, 'TRBV6-9': 67, 'TRBV11-2': 68, 'TRBV18': 69, 'TRBV3-2': 70, 'TRBV6-2': 71, 'TRBV30': 72, 'TRBV4-2': 73, 'TRBV7-2': 74, 'TRBV5-8': 75, 'TRBV20-1': 76, 'TRBV26-2': 77, 'TRBV5-3': 78, 'TRBV12-2':79,'TRBV20-2':80, 'TRBV29-2': 81}

    # Encode each sequence in X using the mapping
    for seq_index in range(number):
        seq = X[seq_index]
        seq = nuc_d[seq]
        x[seq_index,0] = seq

    # Return the encoded sequences
    return x



#--------------------------------------------------------------------------------------------



def encoding_gene_family(number, X):
    
    # Initialize a 2D array with zeros, number of rows = number, number of columns = 1
    x = np.zeros([number,1])
    
    # Get a list of unique gene family names
    gene = list(set(X))
    
    # Create a dictionary with gene family name as key and an integer as value
    nuc_d = {'TRBV4': 21, 'TRBV16': 22, 'TRBV19': 23, 'TRBV27': 24, 'TRBV2': 25, 'TRBV28': 26, 'TRBV25': 27, 'TRBV21': 28, 'TRBV10': 29, 'TRBV15': 30, 'TRBV5': 31, 'TRBV3': 32, 'TRBV14': 33, 'TRBV26': 34, 'TRBV20': 35, 'TRBV12': 36, 'TRBV9': 37, 'TRBV13': 38, 'TRBV29': 39, 'TRBV1': 40, 'TRBV23': 41, 'TRBV11': 42, 'TRBV7': 43, 'TRBV18': 44, 'TRBV30': 45, 'TRBV6': 46, 'TRBV24': 47}
    
    # Loop through each sequence index
    for seq_index in range(number):
        
        # Get the gene family name
        seq = X[seq_index]
        
        # Encode the gene family name using the dictionary nuc_d
        seq = nuc_d[seq]
        
        # Store the encoded gene family name in the array x
        x[seq_index,0] = seq
        
    # Return the encoded array
    return x

#--------------------------------------------------------------------------------------------



def encoding(number, seq_length, X):
    
    
    # Initialize a 2D array with zeros, number of rows = number, number of columns = seq_length
    x = np.zeros([number, seq_length])
    
    # Create a dictionary with nucleotide as key and an integer as value
    nuc_d = {'0': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4,'F': 5,'G': 6,'H': 7,'I': 8,'K': 9,'L': 10,'M': 11,'N': 12,'P': 13,'Q': 14,'R': 15,'S': 16,'T': 17,'V': 18,'W': 19,'Y': 20}
    
    # Loop through each sequence index
    for seq_index in range(number):
        
        # Get the sequence and convert it to uppercase
        seq = X[seq_index].upper()
        
        # Keep only the last seq_length characters and pad the left with zeros if the sequence is shorter than seq_length
        seq = seq[0-seq_length:].rjust(seq_length, '0')
        
        # Loop through each nucleotide position in the sequence
        for n, base in enumerate(seq):
            
            # Encode the nucleotide using the dictionary nuc_d
            x[seq_index][n] = nuc_d[base]
   
    # Return the encoded array
    return x
