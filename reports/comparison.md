# Model Benchmarking and Comparison Report

- **Branch**: `pipeline-A`
- **Winning Architecture**: **1D CNN**
- **Artifacts Saved**: `reports/comparison.csv`, `reports/macro_f1_kappa_comparison.png`, `reports/hard_class_recall_comparison.png`, `reports/generalization_gap_check.png`

---

## 1. Cross-Model Performance Summary Table

| Model               | CV Macro-F1 (Mean +/- Std)   |   Train Acc (%) |   Val Acc (%) |   Test Acc (%) |   Train Macro-F1 |   Val Macro-F1 |   Test Macro-F1 |   Cohen Kappa |   ROC-AUC (OvR) |   N1 Recall (%) |   REM Recall (%) |   Gen Gap (Train-Test F1 %) |   Train Time (s) |   Latency (ms/epoch) |   Model Size (KB) |
|:--------------------|:-----------------------------|----------------:|--------------:|---------------:|-----------------:|---------------:|----------------:|--------------:|----------------:|----------------:|-----------------:|----------------------------:|-----------------:|---------------------:|------------------:|
| Logistic Regression | 0.6492 +/- 0.0200            |           70.26 |         71.25 |          69.36 |           0.6665 |         0.654  |          0.6596 |        0.5628 |          0.9276 |           61.54 |            75.33 |                        0.69 |             61.9 |                0     |               7   |
| Random Forest       | 0.5945 +/- 0.0278            |           73.85 |         71.8  |          67.72 |           0.6632 |         0.6174 |          0.6061 |        0.5366 |          0.9069 |           49.04 |            71.5  |                        5.71 |             21.4 |                0.016 |           15491.6 |
| SVM                 | 0.6673 +/- 0.0122            |           74.73 |         73.96 |          74.39 |           0.6629 |         0.636  |          0.6539 |        0.5753 |          0.9225 |           32.54 |            73.79 |                        0.91 |            776.9 |                0.002 |              21.3 |
| XGBoost             | 0.6664 +/- 0.0384            |           79.74 |         75.37 |          73.37 |           0.7527 |         0.6833 |          0.6795 |        0.6116 |          0.9349 |           57.21 |            77.79 |                        7.32 |             31.9 |                0.002 |            3292.5 |
| 1D CNN              | N/A*                         |           81.49 |         74.97 |          72.77 |           0.7807 |         0.6977 |          0.6896 |        0.6023 |          0.9288 |           60.31 |            78    |                        9.11 |            139.3 |                0.029 |             665.9 |
| LSTM                | N/A*                         |           70.55 |         69.22 |          66.13 |           0.6686 |         0.6518 |          0.6269 |        0.5146 |          0.9111 |           53.27 |            68.05 |                        4.17 |            114.6 |                0.035 |             346.9 |

> **Table Notes**:
> - `*` GroupKFold 3-fold full training was omitted for neural networks (1D CNN & LSTM) due to CPU training latency; models were optimized with early stopping on the Validation split.
> - Sample standard deviation ($ddof=1$) is reported for all GroupKFold folds.
> - **Fair Evaluation**: All models are evaluated on the exact same row instances (dropping the first 9 epochs of each session from validation and test splits, matching the 10-epoch sliding receptive field of sequence models).

---

## 2. Visual Comparative Charts

- **Macro-F1 & Cohen's Kappa**: `reports/macro_f1_kappa_comparison.png`
- **Hard Class Recall (N1 & REM)**: `reports/hard_class_recall_comparison.png`
- **Train vs. Val vs. Test Generalization Gap**: `reports/generalization_gap_check.png`

---

## 3. Generalization Gap Analysis

Evaluating the difference between Train Macro-F1 and Test Macro-F1:
- A large gap (> 15%) indicates memorization / overfitting on training subjects.
- Classical tree ensembles (Random Forest, XGBoost) and deep networks were evaluated strictly on unseen subjects (15 test subjects, 99,855 matched epochs).
- Minimal generalization gaps across validation and test splits confirm robust cross-subject feature invariance.
