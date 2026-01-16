## Source Data

This directory contains the source data files used to generate the figures and perform the analyses presented in the manuscript.

---

### File Descriptions



#### `Score_of_CDR3_HI.csv` and `Score_of_CDR3_SLE.csv`

- **Description**: These files provide the source data for Figures 4 and 5, and Tables 1 and 2. They contain scores for representative T-cell receptor (TCR) CDR3 sequences from the test set repertoires, as inferred by the DeepTAPE TCR-level classifier trained to identify autoimmune-associated CDR3s.
  - `Score_of_CDR3_HI.csv`: Contains scores for sequences from the Healthy Individuals (HI) cohort.
  - `Score_of_CDR3_SLE.csv`: Contains scores for sequences from the SLE patient cohort.
- **Columns**:
  - `CDR3AA`: The amino acid sequence of the deduplicated TCR CDR3 region.
  - `Score`: The autoimmune-associated score of the sequence, inferred by the DeepTAPE TCR classifier.

#### `top2000_autoimmune-associated_CDR3.csv`

- **Description**: This file contains the top 2000 TCR CDR3 sequences with the highest autoimmune-associated scores, selected from the `Score_of_CDR3_SLE.csv` file. This curated list is used to identify potential antigens and genes related to SLE, and discover essential oligopeptides that may serve as potential biomarkers.
- **Columns**:
  - `Top2000_HighScoreSeqs`: The amino acid sequence of the high score TCR CDR3 region.
  - `Score`: The autoimmune-associated score inferred by the DeepTAPE TCR classifier.
 
#### `3_mer_result.csv`

- **Description**: This file contains the 3-mer oligopeptides identified from the `top2000_autoimmune-associated_CDR3.csv` file. It includes their frequency and saliency scores as determined by our analysis.
- **Columns**:
  - `3-mer`: The 3-amino-acid oligopeptide sequence.
  - `Frequency_freq`: The number of times this 3-mer appears in the high-scoring TCR sequences.
  - `Average Score`: The average saliency score indicating the motif's contribution to the model's prediction.

#### `epotope_antigen.csv`

- **Description**: This file lists potential epitopes and their corresponding antigens, identified from the `top2000_autoimmune-associated_CDR3.csv` sequences using the TCRanno tool. The results include hit counts and annotation details.

#### `InnateDB_genes.csv`

- **Description**: This file contains a subset of the InnateDB database, used in our study to filter and functionally annotate the antigens identified by TCRanno.

  
## Baseline Models

The `baseline_models` directory contains the repertoire classifier models implemented for the method comparison section of our study. These baseline models were trained using three different input feature configurations, distinguished by their file suffixes:

-   **`-A.h5`**: Models that use only the CDR3 amino acid sequence as input.
-   **`-A_V.h5`**: Models that use the CDR3 amino acid sequence and the V gene category as input features.
-   **`-A_VF.h5`**: Models that use the CDR3 amino acid sequence and the V gene family category as input features.

The corresponding results and performance evaluation of these models can be found in **Supplementary Tables S1 and S2** of the manuscript.


