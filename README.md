
# DeepTAPE-1.0 Package Documentation

The Deep Learning-based TCRβ-utilized Autoimmune Disease Prediction Engine (DeepTAPE) package provides a deep learning model capable of reference diagnostics for Systemic Lupus Erythematosus (SLE) and other autoimmune diseases.



## Requirements

It is recommended to run this package within a conda environment, and Anaconda Prompt is suggested for use. From downloading the package to satisfying the environment requirements, and then to running the script, the following Prompt commands should be executed in order:

```bash
git clone https://github.com/deepomicslab/DeepTAPE-1.0.git
cd DeepTAPE-1.0
conda create --name DeepTAPE python=3.8
conda activate DeepTAPE
conda install --file requirements_conda.txt
pip install -r requirements_pip.txt
python main.py
```

After these steps, you will be able to run subsequent functions as required by the tool.



This package includes pre-trained models that can be used directly for diagnostics. It also supports training new models within the existing framework and validating them afterward. To use, please operate within the conda environment and execute `main.py`, which also provides usage examples and samples. You can comment out the functions that are not needed in this usage.

## Using Pre-Trained Models

### Diagnosing SLE
To diagnose SLE, you can use the function `result_df = predict_sle_by_DeepTAPE_A_VF(file_path, model_dir)` to directly apply DeepTAPE-A_VF, where:

- `file_path`: The folder containing the samples to be predicted, for example, 'Data/'. Please format the files in the folder as follows:

  <img width="258" alt="image" src="https://github.com/SHENTongfei/DeepTAPE-1.0/assets/116341224/db032603-734b-4ba2-8cdf-1d54fac3b34b">


- `model_dir`: The location of the trained DeepTAPE model `.h5` file, for example, 'Trained_DeepTAPE/'.

- `result_df`: The output is a DataFrame, which is saved and appears as follows:

<img width="255" alt="image" src="https://github.com/SHENTongfei/DeepTAPE-1.0/assets/116341224/25988201-0a0b-40ce-aa18-d27196256da3">

### Diagnosing Other Autoimmune Diseases

For diagnosing other autoimmune diseases, you can use the function `result_df = predict_other_autoimmune_disease(file_path, model_dir)`. This function automatically selects the feature combination to be used based on the Self-adaptive mechanism based on the Pearson Correlation Coefficient (SPCC).

- `file_path`: The folder containing the samples to be predicted, for example, 'Data/'. Please format the files in the folder as above.

- `model_dir`: The location of the trained DeepTAPE model `.h5` file, for example, 'Trained_DeepTAPE/'.

- `result_df`: The output is also a DataFrame, which is saved and appears as above.

## Self-Training and Validation of New Models

To control the training of the model, use the function call:

`results = train_and_use_model(data_type, data, x_test, y_test, neg_file, pos_file, model_type, model_file)`

The `data_type` parameter determines the feature combination used for training the model. There are three options available: “A”, “A_V”, and “A_VF”.

### Training a Model Based on Amino Acid Sequence Features

If you wish to train a model based solely on amino acid sequences, you would set the `data_type` parameter to "A"，where the input other hyperparameters are:

- `data`: The file path for the training data, which should be a CSV file containing features (CDR3AA sequences). For example, 'Train_and_Test_Data\\Data_for_train\\Sample_A.csv'. Please format the file as follows:

  <img width="222" alt="image" src="https://github.com/SHENTongfei/DeepTAPE-1.0/assets/116341224/af51e51f-8d81-448c-b30f-b0f6b780705a">


- `x_test`: The file name for testing, for example, 'Train_and_Test_Data\\X_test.csv', which appears as follows:

  <img width="109" alt="image" src="https://github.com/SHENTongfei/DeepTAPE-1.0/assets/116341224/2182782a-195b-4b88-ad26-507fa21f1130">


- `y_test`: The file path for the labels in the test dataset (such as positive or negative samples). For example, 'Train_and_Test_Data\\X_test.csv', which appears as follows:

  <img width="109" alt="image" src="https://github.com/SHENTongfei/DeepTAPE-1.0/assets/116341224/4721ae96-16e4-4051-8250-639ec961a20a">


- `neg_file`: The file path for negative samples, used to separate negative samples during prediction. For example, 'Train_and_Test_Data\\neg_Data'.

- `pos_file`: The file path for positive samples, used to separate positive samples during prediction. For example, 'Train_and_Test_Data\\pos_Data'.

- `model_type`: The type of machine learning model used for training, saved in `models.py`. For example, cnn_lstm_res.

- `model_file`: The file path for saving the trained model. For example, 'Trained_DeepTAPE/DeepTAPE_A_new.h5'.

The output results are:

- `results['accuracy']`: Represents the overall correctness of the model's predictions.
  
- `results['precision']`: Measures the proportion of true positive predictions in the positive class.

- `results['recall']`: Indicates the model's ability to find all the relevant cases within a dataset.
  
- `results['F1_score']`: The harmonic mean of precision and recall, providing a balance between the two.
  
- `results['roc_auc']`: Reflects the likelihood that the model ranks a random positive example more highly than a random negative example.
  
- `results['pre']`: The predicted values, i.e., the Autoimmune Risk Score.
  
- `results['Y_test']`: The array of true labels for the test data.
  
- `results['SOS_C']`: The predicted scores for each sequence of negative samples.
  
- `results['SOS_S']`: The predicted scores for each sequence of positive samples.


### Training a Model Based on V-Gene and Amino Acid Sequence Features

To train a model based on V-gene and amino acid sequence feature combinations, you would set the `data_type` parameter to "A_V"，where the input other hyperparameters are:

- `data`: The file path for the training data, which should be a CSV file containing features (CDR3AA sequences and corresponding V-genes). For example, 'Train_and_Test_Data\\Data_for_train\\Sample_A_V.csv'. Please format the file as follows:

  <img width="109" alt="image" src="https://github.com/SHENTongfei/DeepTAPE-1.0/assets/116341224/fc067f24-5c1a-4664-9ed9-fcfae5c34d4c">


- `x_test`, `y_test`, `neg_file`, `pos_file`, `model`, `model_file`: The meanings are the same as above, with `model` example changed to `cnn_lstm_res_gene`.

The output results are:

- `results[accuracy]`, `results[precision]`, `results[recall]`, `results[F1_score]`, `results[roc_auc]`, `results[pre]`, `results[Y_test]`: The meanings are the same as above.

- `results['DF_N']`: A DataFrame for negative samples with columns {'AA': AA, 'VGene': VF, 'predictions': PRE}, representing the amino acid sequence, its corresponding V-gene, and the prediction results.
- `results['DF_P']`: A DataFrame for positive samples with columns {'AA': AA, 'VGene': VF, 'predictions': PRE}, representing the amino acid sequence, its corresponding V-gene, and the prediction results.
- `results['EDP']`: A DataFrame of the top 100 peptide segments with the highest prediction scores among positive samples, for further research.
- `results['EDN']`: A DataFrame of the bottom 100 peptide segments with the lowest prediction scores among negative samples, for further research.

### Training a Model Based on V-Gene Family and Amino Acid Sequence Features

To train a model based on V-gene family and amino acid sequence feature combinations, you would set the `data_type` parameter to "A_VF"，where the input other hyperparameters are:


- `data`: The file path for the training data, which should be a CSV file containing features (CDR3AA sequences and corresponding V-gene families). For example, 'Train_and_Test_Data\\Data_for_train\\Sample_A_VF.csv'. Please format the file as follows:

  <img width="279" alt="image" src="https://github.com/SHENTongfei/DeepTAPE-1.0/assets/116341224/f54a2797-c45b-4a6f-90ad-34969e9d8ebd">


The output results are:

- `results[accuracy]`, `results[precision]`, `results[recall]`, `results[F1_score]`, `results[roc_auc]`, `results[pre]`, `results[Y_test]`, `results[EDP]`, `results[EDN]`: The meanings are the same as above.

- `results[DF_N]`: A DataFrame for negative samples with columns {'AA': AA, 'VGeneFam': VF, 'predictions': PRE}, representing the amino acid sequence, its corresponding V-gene family, and the prediction results.

- `results[DF_P]`: A DataFrame for positive samples with columns {'AA': AA, 'VGeneFam': VF, 'predictions': PRE}, representing the amino acid sequence, its corresponding V-gene family, and the prediction results.

## Generating Training Set Merge Data Tool

If you wish to generate a training set similar to the example provided, we offer three functions as tools for generating merged training data: `merged_data_A = mergedata(file_neg, file_pos, num_freq)`, `merged_data_A_V = mergedata_with_gene(file_neg, file_pos, AA_column, gene_column, num_freq)`, and `merged_data_A_VF = mergedata_with_gene_family(file_neg, file_pos, AA_column, gene_column, num_freq)`. The input hyperparameters are:

- `file_neg`: The file path for the folder containing negative sample files.

- `file_pos`: The file path for the folder containing positive sample files.

- `num_freq`: In the `mergedata` function, this is the number of most frequently occurring CDR3AA sequences to be selected from each file. In the `mergedata_with_gene` and `mergedata_with_gene_family` functions, this is the number of most frequently occurring peptide and gene or gene family combinations to be selected from each file.

- `AA_column`: (Used only in `mergedata_with_gene` and `mergedata_with_gene_family`) Specifies the index of the column containing the amino acid sequences.

- `gene_column`: (Used only in `mergedata_with_gene` and `mergedata_with_gene_family`) Specifies the index of the column containing the genes or gene families.

---
