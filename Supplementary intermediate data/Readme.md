# Source Data

This directory contains the source data files used to generate the figures and perform the analyses presented in the manuscript.

---

## File Descriptions

### `SLE_Clinical_Info_brief.new.csv`

- **Description**: This file contains key clinical information for the Systemic Lupus Erythematosus (SLE) patient cohort.
- **Columns**:
  - `C3`: A complement protein where low levels suggest higher disease activity.
  - `C4`: A complement protein where low levels are associated with increased disease activity.
  - `Anti-dsDNA`: Autoantibodies where high levels correlate with greater disease activity.
  - `damage`: Categorizes cumulative, irreversible organ damage based on the number of affected tissue types.

### `autoimmune-associated_score_in_HI.csv` and `autoimmune-associated_score_in_SLE.csv`

- **Description**: These files provide the source data for Figure 4 and 5. They contain scores for representative T-cell receptor (TCR) CDR3 sequences, as inferred by the DeepTAPE TCR-level classifier trained to identify autoimmune-associated CDR3s. The sequences have been deduplicated by their CDR3 amino acid sequence, and the score from their first appearance is retained.
  - `autoimmune-associated_score_in_HI.csv`: Contains scores for sequences from the Healthy Individuals (HI) cohort.
  - `autoimmune-associated_score_in_SLE.csv`: Contains scores for sequences from the SLE patient cohort.
- **Columns**:
  - `CDR3AA`: The amino acid sequence of the deduplicated TCR CDR3 region.
  - `SeqScore`: The autoimmune-associated score assigned to the sequence by the model.