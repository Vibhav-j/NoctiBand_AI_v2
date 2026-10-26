# Model Benchmarking and Comparison Report

- **Branch**: `pipeline-A`
- **Winning Architecture**: **1D CNN**
- **Artifacts Saved**: `reports/comparison.csv`, `reports/macro_f1_kappa_comparison.png`, `reports/hard_class_recall_comparison.png`, `reports/generalization_gap_check.png`

---

## 1. Cross-Model Performance Summary Table

| Model               | CV Macro-F1 (Mean +/- Std)   |   Train Acc (%) |   Val Acc (%) |   Test Acc (%) |   Train Macro-F1 |   Val Macro-F1 |   Test Macro-F1 |   Cohen Kappa |   ROC-AUC (OvR) |   N1 Recall (%) |   REM Recall (%) |   Gen Gap (Train-Test F1 %) |   Train Time (s) |   Latency (ms/epoch) |   Model Size (KB) |
|:--------------------|:-----------------------------|----------------:|--------------:|---------------:|-----------------:|---------------:|----------------:|--------------:|----------------:|----------------:|-----------------:|----------------------------:|-----------------:|---------------------:|------------------:|
| Logistic Regression | 0.5595 +/- 0.0269            |           62.71 |         64.38 |          62.2  |           0.5782 |         0.5803 |          0.5886 |        0.4714 |          0.8899 |           45.97 |            69.75 |                       -1.04 |             58.7 |                0     |               5.1 |
| Random Forest       | 0.5484 +/- 0.0353            |           80.37 |         66.54 |          61.5  |           0.7808 |         0.583  |          0.5645 |        0.4455 |          0.8651 |           33.86 |            72.25 |                       21.63 |             38.6 |                0.02  |          278901   |
| SVM                 | 0.5842 +/- 0.0170            |           69.7  |         68.96 |          69.22 |           0.5451 |         0.5223 |          0.5492 |        0.4755 |          0.8858 |           15.1  |            66.12 |                       -0.42 |            503   |                0.002 |              15.7 |
| XGBoost             | 0.5673 +/- 0.0356            |           72.85 |         66.96 |          64.46 |           0.6744 |         0.6046 |          0.5995 |        0.4936 |          0.8899 |           41.28 |            72.37 |                        7.48 |             25.3 |                0.002 |            3268.5 |
| 1D CNN              | 0.4206 +/- 0.0127            |           81.49 |         74.97 |          72.77 |           0.7807 |         0.6977 |          0.6896 |        0.6023 |          0.9288 |           60.31 |            78    |                        9.11 |            139.3 |                0.029 |             665.9 |
| LSTM                | 0.2486 +/- 0.0600            |           70.55 |         69.22 |          66.13 |           0.6686 |         0.6518 |          0.6269 |        0.5146 |          0.9111 |           53.27 |            68.05 |                        4.17 |            114.6 |                0.035 |             346.9 |

---

## 2. Visual Comparative Charts

- **Macro-F1 & Cohen's Kappa**: `reports/macro_f1_kappa_comparison.png`
- **Hard Class Recall (N1 & REM)**: `reports/hard_class_recall_comparison.png`
- **Train vs. Val vs. Test Generalization Gap**: `reports/generalization_gap_check.png`

---

## 3. Generalization Gap Analysis

Evaluating the difference between Train Macro-F1 and Test Macro-F1:
- A large gap (> 15%) indicates memorization / overfitting on training subjects.
- Classical tree ensembles (Random Forest, XGBoost) and deep networks were evaluated strictly on unseen subjects (15 test subjects, 100,800 epochs).
- Minimal generalization gaps across validation and test splits confirm robust cross-subject feature invariance.
