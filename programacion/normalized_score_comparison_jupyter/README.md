# Normalized SOC Score Comparison — Jupyter

Three notebooks apply the two formulations:

1. `01_model_relative_minmax_score.ipynb`
2. `02_soc_range_normalized_score.ipynb`
3. `03_comparison_and_recommendation.ipynb`

`Soc_model_results.csv` contains reproducible illustrative results for five
models and five land-use categories.

## Formula A
Model-relative min-max normalization across competing models within category.

## Formula B
SOC-range normalization:
`R_C = SOCmax,C - SOCmin,C`,
then `nRMSE=RMSE/R_C`, `nMAE=MAE/R_C`, `nBias=abs(Bias)/R_C`,
with higher-is-better transformations and equal weighting.

## Run
`pip install numpy pandas matplotlib seaborn jupyter`
then `jupyter lab`.

Formula B is recommended for cross-land-use-category comparison because its
normalization does not depend on the set of competing models.
