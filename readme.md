
# Feature Relevance Explainers in Tabular Anomaly Detection

This repository contains the code for the experiments in the dissertation "Feature Relevance Explainers in Tabular Anomaly Detection".

## Data

`data` contains the smaller data sets used and has downloading instructions for the larger data sets.  

## Training detectors

Parameter studies can be conducted using `erp_param_search.py` or `cidds_param_search.py`.

The best performing models and hyperparameters used in the experiments are available in `outputs/models/`.

## Generating Feature Relevance Explanations

Trained models can be explained using `erp_xai.py` or `cidds_xai.py`.

**Note**: Additional setup is required for running SHAP with *optimized* reference data.
To integrate the optimization procedure directly within kernel-SHAP,
this implementation requires to manually override the `shap/explainer/_kernel.py` script within the SHAP package.
For this, either override the contents of `shap/explainer/_kernel.py` entirely
with the backup file provided in `xai/backups/shap_kernel_backup.py`
or add the small segments marked with `# NEWCODE` within `xai/backups/shap_kernel_backup.py` in the
original library file of `shap/explainer/_kernel.py`.


## Running Explainer Evaluations

Correctness and completeness evaluations are part of `erp_xai.py` or `cidds_xai.py`.
Consistency evaluations are in `erp_rashomon_eval.py` or `cidds_rashomon_eval.py`.
Compactness evaluations are in `erp_size_eval.py` or `cidds_size_eval.py`.
Continuity and contrastivity heatmaps are created through `plotting/heatmap_plots_erp.py` or `plotting/heatmap_plots_cidds.py`.
