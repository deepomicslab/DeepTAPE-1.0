
# DeepTAPE-1.0 Package Documentation

The DeepTAPE package provides a deep learning model capable of reference diagnostics for Systemic Lupus Erythematosus (SLE) and other autoimmune diseases.

## Requirements

It is recommended to run this package within a conda environment. The preferred conda environment can be found in the `requirements.txt` file.

This package includes pre-trained models that can be used directly for diagnostics. It also supports training new models within the existing framework and validating them afterward. To use, please operate within the conda environment and execute `main.py`, which also provides usage examples and samples.

## Using Pre-Trained Models

### Diagnosing SLE
To diagnose SLE, you can use the function `result_df = predict_sle_by_DeepTAPE_A_VF(file_path, model_dir)` to directly apply DeepTAPE-A_VF, where:

- `file_path`: The folder containing the samples to be predicted, for example, 'Data/'. Please format the files in the folder as follows:

- `model_dir`: The location of the trained DeepTAPE model `.h5` file, for example, 'Trained_DeepTAPE/'.

- `result_df`: The output is a DataFrame, which is saved and appears as follows:

### Diagnosing Other Autoimmune Diseases

For diagnosing other autoimmune diseases, you can use the function `result_df = predict_other_autoimmune_disease(file_path, model_dir)`. This function automatically selects the feature combination to be used based on the Self-adaptive mechanism based on the Pearson Correlation Coefficient (SPCC).

- `file_path`: The folder containing the samples to be predicted, for example, 'Data/'. Please format the files in the folder as above.

- `model_dir`: The location of the trained DeepTAPE model `.h5` file, for example, 'Trained_DeepTAPE/'.

- `result_df`: The output is also a DataFrame, which is saved and appears as above.

## Self-Training and Validation of New Models

### Training a Model Based on Amino Acid Sequence Features

If you wish to train a model based solely on amino acid sequences, you can use the function `accuracy, precision, recall, F1_score, roc_auc, pre, Y_test, SOS_C, SOS_S = train_and_use(data, x_test, y_test, neg_file, pos_file, model_type, model_file)`, where the input hyperparameters are:

- `data`: The file path for the training data, which should be a CSV file containing features (CDR3AA sequences). For example, 'Train&Test_Data\\Data_for_train\\Sample_A.csv'. Please format the file as follows:

- `x_test`: The file name for testing, for example, 'Train&Test_Data\\X_test.csv', which appears as follows:

- `y_test`: The file path for the labels in the test dataset (such as positive or negative samples). For example, 'Train&Test_Data\\X_test.csv', which appears as follows:

- `neg_file`: The file path for negative samples, used to separate negative samples during prediction. For example, 'Train&Test_Data\\neg_Data'.

- `pos_file`: The file path for positive samples, used to separate positive samples during prediction. For example, 'Train&Test_Data\\pos_Data'.

- `model_type`: The type of machine learning model used for training, saved in `models.py`. For example, cnn_lstm_res.

- `model_file`: The file path for saving the trained model. For example, 'Trained_DeepTAPE/DeepTAPE_A_new.h5'.

The output results are:

- `accuracy`: The accuracy of the prediction.
- `precision`: The precision of the prediction.
- `recall`: The recall rate of the prediction.
- `F1_score`: The F1 score of the prediction.
- `roc_auc`: The ROC-AUC value of the prediction.
- `pre`: The predicted values, i.e., the Autoimmune Risk Score.
- `Y_test`: The array of true labels for the test data.
- `SOS_C`: The predicted values for each sequence of negative samples.
- `SOS_S`: The predicted values for each sequence of positive samples.

### Training a Model Based on V-Gene and Amino Acid Sequence Features

To train a model based on V-gene and amino acid sequence feature combinations, use the function `accuracy, precision, recall, F1_score, roc_auc, pre, DF_N, DF_P, Y_test, EDP, EDN = train_and_use_with_gene(data, x_test, y_test, neg_file, pos_file, model, model_file)`, where the input hyperparameters include:

- `data`: The file path for the training data, which should be a CSV file containing features (CDR3AA sequences and corresponding V-genes). For example, 'Train&Test_Data\\Data_for_train\\Sample_A_V.csv'. Please format the file as follows:

- `x_test`, `y_test`, `neg_file`, `pos_file`, `model`, `model_file`: The meanings are the same as above, with `model` example changed to `cnn_lstm_res_gene`.

The output results are:

- `accuracy`, `precision`, `recall`, `F1_score`, `roc_auc`, `pre`, `Y_test`: The meanings are the same as above.

- `DF_N`: A DataFrame for negative samples with columns {'AA': AA, 'VGene': VF, 'predictions': PRE}, representing the amino acid sequence, its corresponding V-gene, and the prediction results.

- `DF_P`: A DataFrame for positive samples with columns {'AA': AA, 'VGene': VF, 'predictions': PRE}, representing the amino acid sequence, its corresponding V-gene, and the prediction results.

- `EDP`: A DataFrame of the top 100 peptide segments with the highest prediction scores among positive samples, for further research.

- `EDN`: A DataFrame of the bottom 100 peptide segments with the lowest prediction scores among negative samples, for further research.

### Training a Model Based on V-Gene Family and Amino Acid Sequence Features

To train a model based on V-gene family and amino acid sequence feature combinations, use the function `accuracy, precision, recall, F1_score, roc_auc, pre, DF_N, DF_P, Y_test, EDP, EDN = train_and_use_with_gene_family(data, x_test, y_test, neg_file, pos_file, model, model_file)`, where the input hyperparameters include:

- `data`: The file path for the training data, which should be a CSV file containing features (CDR3AA sequences and corresponding V-gene families). For example, 'Train&Test_Data\\Data_for_train\\Sample_A_VF.csv'. Please format the file as follows:

The output results are:

- `accuracy`, `precision`, `recall`, `F1_score`, `roc_auc`, `pre`, `Y_test`, `EDP`, `EDN`: The meanings are the same as above.

- `DF_N`: A DataFrame for negative samples with columns {'AA': AA, 'VGeneFam': VF, 'predictions': PRE}, representing the amino acid sequence, its corresponding V-gene family, and the prediction results.

- `DF_P`: A DataFrame for positive samples with columns {'AA': AA, 'VGeneFam': VF, 'predictions': PRE}, representing the amino acid sequence, its corresponding V-gene family, and the prediction results.

## Generating Training Set Merge Data Tool

If you wish to generate a training set similar to the example provided, we offer three functions as tools for generating merged training data: `merged_data_A = mergedata(file_neg, file_pos, num_freq)`, `merged_data_A_V = mergedata_with_gene(file_neg, file_pos, AA_column, gene_column, num_freq)`, and `merged_data_A_VF = mergedata_with_gene_family(file_neg, file_pos, AA_column, gene_column, num_freq)`. The input hyperparameters are:

- `file_neg`: The file path for the folder containing negative sample files.

- `file_pos`: The file path for the folder containing positive sample files.

- `num_freq`: In the `mergedata` function, this is the number of most frequently occurring CDR3AA sequences to be selected from each file. In the `mergedata_with_gene` and `mergedata_with_gene_family` functions, this is the number of most frequently occurring peptide and gene or gene family combinations to be selected from each file.

- `AA_column`: (Used only in `mergedata_with_gene` and `mergedata_with_gene_family`) Specifies the index of the column containing the amino acid sequences.

- `gene_column`: (Used only in `mergedata_with_gene` and `mergedata_with_gene_family`) Specifies the index of the column containing the genes or gene families.

---
