## Source Data

This directory contains the source data files used to generate the figures and perform the analyses presented in the manuscript.

---

### File Descriptions

#### `SLE_Clinical_Info_brief.new.csv`

- **Description**: This file contains key clinical information for the Systemic Lupus Erythematosus (SLE) patient cohort.
- **Columns**:
  - `C3`: A complement protein; low levels suggest higher disease activity.
  - `C4`: A complement protein; low levels are associated with increased disease activity.
  - `Anti-dsDNA`: Autoantibodies; high levels correlate with greater disease activity.
  - `damage`: Categorizes cumulative, irreversible organ damage based on the number of affected tissue types.

#### `Score_of_CDR3_HI.csv` and `Score_of_CDR3_SLE.csv`

- **Description**: These files provide the source data for Figures 4 and 5, and Tables 1 and 2. They contain scores for representative T-cell receptor (TCR) CDR3 sequences from the test set repertoires, as inferred by the DeepTAPE TCR-level classifier trained to identify autoimmune-associated CDR3s.
  - `Score_of_CDR3_HI.csv`: Contains scores for sequences from the Healthy Individuals (HI) cohort.
  - `Score_of_CDR3_SLE.csv`: Contains scores for sequences from the SLE patient cohort.
- **Columns**:
  - `CDR3AA`: The amino acid sequence of the deduplicated TCR CDR3 region.
  - `Score`: The autoimmune-associated score of the sequence, inferred by the DeepTAPE TCR classifier.

#### `top2000_autoimmune-associated_CDR3.csv`

- **Description**: This file contains the top 2000 TCR CDR3 sequences with the highest autoimmune-associated scores, selected from the `Score_of_CDR3_SLE.csv` file. This curated list is intended for downstream analyses, such as identifying potential antigens and genes related to SLE, and discovering essential oligopeptides that may serve as potential biomarkers.
- **Columns**:
  - `Top2000_HighScoreSeqs`: The amino acid sequence of the high score TCR CDR3 region.
  - `Score`: The autoimmune-associated score inferred by the DeepTAPE TCR classifier.
 
  
## Baseline Models

The `baseline_models` directory contains the repertoire classifier models implemented for the method comparison section of our study. These baseline models were trained using three different input feature configurations, distinguished by their file suffixes:

-   **`-A.h5`**: Models that use only the CDR3 amino acid sequence as input.
-   **`-A_V.h5`**: Models that use the CDR3 amino acid sequence and the V gene category as input features.
-   **`-A_VF.h5`**: Models that use the CDR3 amino acid sequence and the V gene family category as input features.

The corresponding results and performance evaluation of these models can be found in **Supplementary Tables S1 and S2** of the manuscript.
