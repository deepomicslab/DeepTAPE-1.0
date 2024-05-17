# -*- coding: utf-8 -*-
"""
Created on Tue May  2 20:03:53 2023

@author: timshen2
"""

import numpy as np
import pandas as pd
import os
from encoding import *
from prediction import *
from SPCC import *
from keras.models import load_model
from Predicate_other import *
from mergedata import *
from train_and_use import *


# # If the data you put in the folder is for diagnosing SLE, you can directly choose DeepTAPE-A_VF, the combination of V gene family and amino acid features performs best for diagnosing SLE.
# # --------------------------------------------------------------------------------------------

# result_df = predict_sle_by_DeepTAPE_A_VF('Data/', 'Trained_DeepTAPE/')
# # Output the result DataFrame
# print(result_df)


# # If the data you put in the folder is for diagnosing other autoimmune diseases, you need to use SPCC to select the best feature combination.
# # --------------------------------------------------------------------------------------------

# result_df = predict_other_autoimmune_disease('Data/', 'Trained_DeepTAPE/')
# print(result_df)





# # --------------------------------------------------------------------------------------------

# # If you have curated training data with AA and V_gene that you haven't used to train a model yet, use the following functions.
# result = train_and_use_with_gene('Train_and_Test_Data/Data_for_train/Sample_A_V.csv',
#                                 'Train_and_Test_Data/X_test.csv',
#                                 'Train_and_Test_Data/Y_test.csv',
#                                 'Train_and_Test_Data/neg_Data/',
#                                 'Train_and_Test_Data/pos_Data/',
#                                 cnn_lstm_res_gene,
#                                 'Trained_DeepTAPE/DeepTAPE_A_V_new.h5')

# # If you have curated training data with AA and V_gene_family that you haven't used to train a model yet, use the following functions.
# result = train_and_use_with_gene_family('Train_and_Test_Data/Data_for_train/Sample_A_VF.csv',
#                                         'Train_and_Test_Data/X_test.csv',
#                                         'Train_and_Test_Data/Y_test.csv',
#                                         'Train_and_Test_Data/neg_Data/',
#                                         'Train_and_Test_Data/pos_Data/',
#                                         cnn_lstm_res_gene,
#                                         'Trained_DeepTAPE/DeepTAPE_A_VF_new.h5')

# result = train_and_use('Train_and_Test_Data/Data_for_train/Sample_A.csv',
#                     'Train_and_Test_Data/X_test.csv',
#                     'Train_and_Test_Data/Y_test.csv',
#                     'Train_and_Test_Data/neg_Data/',
#                     'Train_and_Test_Data/pos_Data/',
#                       cnn_lstm_res,
#                       'Trained_DeepTAPE/DeepTAPE_A_new.h5')






# # --------------------------------------------------------------------------------------------

# merged_data_A = mergedata('Train_and_Test_Data/neg_Data/', 'Train_and_Test_Data/pos_Data/', 2000)
# merged_data_A_V = mergedata_with_gene('Train_and_Test_Data/neg_Data/', 'Train_and_Test_Data/pos_Data/', 0, 1, 2000)
# merged_data_A_VF = mergedata_with_gene_family('Train_and_Test_Data/neg_Data/', 'Train_and_Test_Data/pos_Data/', 0, 1, 2000)