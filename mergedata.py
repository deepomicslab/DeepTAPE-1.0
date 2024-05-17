# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 20:54:59 2023

@author: timshen2
"""
import numpy as np
import pandas as pd
import os
from crossvalidation import *


def mergedata(file_neg, file_pos, num_freq):
    total = pd.DataFrame()
    
    # Loop through the file names in the file lists
    for file_list, zero_one in [(file_neg, '0'), (file_pos, '1')]:
        for name in os.listdir(file_list):
            # Read the file into a pandas dataframe and select the 8th column
            pt = pd.read_csv(file_list + name)
            pt = pt.iloc[:, 0]
            
            # Create a new list and add sequences that meet certain criteria
            AA = []
            for i in pt:
                if 24 >= len(str(i)) >= 10 and i[0] == 'C' and i[-1] == 'F' and ('*' not in i) and ('x' not in i):
                    AA.append(i)
            
            # Create a new pandas dataframe from the cleaned list of sequences
            pt_cleaned = {'CDR3AA': AA}
            ptc = pd.DataFrame(pt_cleaned)
            
            # Get the top N most frequently occurring CDR3AA sequences
            ptc = ptc.value_counts(ascending=False).rename_axis('CDR3AA').reset_index(name='counts_aa')
            ptc.sort_values(by='counts_aa')
            ptc = ptc[:num_freq]
            ptc = ptc.drop('counts_aa', axis=1) 
            
            # Add a column to the dataframe with the binary classification label
            ptc['positive'] = zero_one
            
            # Concatenate the dataframes into a single one
            total = pd.concat([total, ptc], ignore_index=True)
            
            # Remove duplicates from the final dataframe
            total.drop_duplicates(subset=['CDR3AA'], keep='first', inplace=True)
            print(name, len(total))
        
    # Randomly shuffle the rows
    total = total.sample(frac=1).reset_index(drop=True)
    
    # Make the number of 0s and 1s equal
    min_count = min(total['positive'].value_counts())
    total = pd.concat([total[total['positive'] == '0'].sample(min_count), total[total['positive'] == '1'].sample(min_count)], ignore_index=True)
    
    # Randomly shuffle the rows again
    total = total.sample(frac=1).reset_index(drop=True)
    
    return total



#--------------------------------------------------------------------------------------------



def mergedata_with_gene(file_neg, file_pos, AA_column, gene_column, num_freq):
    # Initialize an empty dataframe
    total = pd.DataFrame()
    
    # Loop through the file names in the file lists
    for file_list, zero_one in [(file_neg, '0'), (file_pos, '1')]:
        for name in os.listdir(file_list):
            mix = []
            # Read in the tab-separated file and extract columns with amino acid sequences and gene names
            pt = pd.read_csv(file_list + name)
            pt = pt.iloc[:, [AA_column, gene_column]]
            
            u = 0
            # Iterate through the amino acid sequences and gene names, and keep only those that meet certain criteria
            for i in pt.iloc[:, 0]:
                if 24 >= len(str(i)) >= 10 and i[0] == 'C' and i[-1] == 'F' and ('*' not in i) and ('x' not in i):
                    g = pt.iloc[u, 1]
                    g_c = g[:g.index('*')]
                    if '/' in g_c:
                        g_c = g_c[:g_c.index('/')] + g_c[-2:]
                    mix.append(i + '_' + g_c)
                u += 1
            
            # Create a dataframe with the amino acid/gene name combinations and the frequency of each combination
            pt_cleaned = {'Mix': mix}
            ptc = pd.DataFrame(pt_cleaned)
            ptc = ptc.value_counts(ascending=False).rename_axis('Mix').reset_index(name='counts')
            ptc.sort_values(by='counts')
            ptc = ptc[:num_freq]
            ptc = ptc.drop('counts', axis=1) 
            
            
            
            # If this is the first iteration, set the total dataframe to the current one
            # Otherwise, concatenate the current dataframe with the total dataframe
            if total.empty:
                total = ptc
            else:
                total = pd.concat([total, ptc], ignore_index=True)  
            
            
            # Drop duplicates every 10 iterations
            if len(total) % 10 == 0:
                total.drop_duplicates(subset=['Mix'], keep='first', inplace=True)
            
            print(name, len(total))
        
    # Drop duplicates again at the end
    total.drop_duplicates(subset=['Mix'], keep='first', inplace=True)
    
    total[['CDR3AA', 'V_gene']] = total['Mix'].str.split('_', expand=True)
    
    total = total[['CDR3AA', 'V_gene', 'positive']]

    # Randomly shuffle the rows
    total = total.sample(frac=1).reset_index(drop=True)
    
    # Make the number of 0s and 1s equal
    min_count = min(total['positive'].value_counts())
    total = pd.concat([total[total['positive'] == '0'].sample(min_count), total[total['positive'] == '1'].sample(min_count)], ignore_index=True)
    
    # Randomly shuffle the rows again
    total = total.sample(frac=1).reset_index(drop=True)
    
    return total



#--------------------------------------------------------------------------------------------



def mergedata_with_gene_family(file_neg, file_pos, AA_column, gene_column, num_freq):
    # Initialize an empty dataframe
    total = pd.DataFrame()
    
    # Loop through the file names in the file lists
    for file_list, zero_one in [(file_neg, '0'), (file_pos, '1')]:
        for name in os.listdir(file_list):
            mix = []
            # Read in the tab-separated file and extract columns with amino acid sequences and gene names
            pt = pd.read_csv(file_list + name)
            pt = pt.iloc[:, [AA_column, gene_column]]
            
            u = 0
            # Iterate through the amino acid sequences and gene names, and keep only those that meet certain criteria
            for i in pt.iloc[:, 0]:
                if 24 >= len(str(i)) >= 10 and i[0] == 'C' and i[-1] == 'F' and ('*' not in i) and ('x' not in i):
                    g = pt.iloc[u, 1]
                    g_c = g[:g.index('*')]
                    if '-' in g_c:
                        g_c = g_c[:g_c.index('-')]
                    if '/' in g_c:
                        g_c = g_c[:g_c.index('/')]
                    mix.append(i + '_' + g_c)
                u += 1
            
            # Create a dataframe with the amino acid/gene name combinations and the frequency of each combination
            pt_cleaned = {'Mix': mix}
            ptc = pd.DataFrame(pt_cleaned)
            ptc = ptc.value_counts(ascending=False).rename_axis('Mix').reset_index(name='counts')
            ptc.sort_values(by='counts')
            ptc = ptc[:num_freq]
            ptc = ptc.drop('counts', axis=1) 
            
            # Add a column to the dataframe with the binary classification label
            ptc['positive'] = zero_one
            
            # If this is the first iteration, set the total dataframe to the current one
            # Otherwise, concatenate the current dataframe with the total dataframe
            if total.empty:
                total = ptc
            else:
                total = pd.concat([total, ptc], ignore_index=True)
            
            # Drop duplicates every 10 iterations
            if len(total) % 10 == 0:
                total.drop_duplicates(subset=['Mix'], keep='first', inplace=True)
            
           
            print(name, len(total))
        
    # Drop duplicates again at the end
    total.drop_duplicates(subset=['Mix'], keep='first', inplace=True)
    # Randomly shuffle the rows
    total = total.sample(frac=1).reset_index(drop=True)
    
    # Split the Mix column into CDR3AA and V_gene_family columns
            
    total[['CDR3AA', 'V_gene_family']] = total['Mix'].str.split('_', expand=True)
    
    total = total[['CDR3AA', 'V_gene_family', 'positive']]
    
    # Make the number of 0s and 1s equal
    min_count = min(total['positive'].value_counts())
    total = pd.concat([total[total['positive'] == '0'].sample(min_count), total[total['positive'] == '1'].sample(min_count)], ignore_index=True)
    
    # Randomly shuffle the rows again
    total = total.sample(frac=1).reset_index(drop=True)
    
    return total