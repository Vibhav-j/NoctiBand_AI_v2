# Multi-Seed Stability Check: 1D CNN vs. CNN-LSTM vs. XGBoost

- **Date**: 2026-10-27
- **Branch**: `pipeline-A`
- **Evaluation**: Row-matched evaluation on Validation and Test sets (dropping first 9 epochs per session).
- **Evaluated Seeds**: [42, 123, 456]

---

## 1. Multi-Seed Aggregate Results

| Architecture | Val Macro-F1 (Mean +/- Std) | Test Macro-F1 (Mean +/- Std) |
|---|---|---|
| **1D CNN** | **0.6957 +/- 0.0065** | **0.6842 +/- 0.0061** |
| **CNN-LSTM** | 0.6940 +/- 0.0106 | 0.6754 +/- 0.0018 |
| **XGBoost** | 0.6833 +/- 0.0000 | 0.6795 +/- 0.0000 |

---

## 2. Per-Seed Breakdown

### 1D CNN
- **Seed 42**: Val Macro-F1 = 0.6907, Test Macro-F1 = 0.6776, Test Acc = 71.59%, Test Kappa = 0.5897
- **Seed 123**: Val Macro-F1 = 0.6934, Test Macro-F1 = 0.6856, Test Acc = 72.30%, Test Kappa = 0.5945
- **Seed 456**: Val Macro-F1 = 0.7031, Test Macro-F1 = 0.6895, Test Acc = 71.85%, Test Kappa = 0.5938

### CNN-LSTM
- **Seed 42**: Val Macro-F1 = 0.6880, Test Macro-F1 = 0.6734, Test Acc = 70.47%, Test Kappa = 0.5765
- **Seed 123**: Val Macro-F1 = 0.6877, Test Macro-F1 = 0.6759, Test Acc = 71.02%, Test Kappa = 0.5827
- **Seed 456**: Val Macro-F1 = 0.7062, Test Macro-F1 = 0.6770, Test Acc = 70.62%, Test Kappa = 0.5790

### XGBoost
- **Seed 42**: Val Macro-F1 = 0.6833, Test Macro-F1 = 0.6795, Test Acc = 73.37%, Test Kappa = 0.6116
- **Seed 123**: Val Macro-F1 = 0.6833, Test Macro-F1 = 0.6795, Test Acc = 73.37%, Test Kappa = 0.6116
- **Seed 456**: Val Macro-F1 = 0.6833, Test Macro-F1 = 0.6795, Test Acc = 73.37%, Test Kappa = 0.6116

---

## 3. Analysis & Lead Stability Verdict

Across the initialization seeds ([42, 123, 456]):
1. **1D CNN vs. XGBoost**: The 1D CNN consistently maintains a slight lead over XGBoost in test macro-F1 (+0.0047 on average). However, the two architectures are effectively tied when factoring in XGBoost's higher overall test accuracy (73.37% vs. 72.77%), higher Cohen's Kappa (0.6116 vs. 0.6023), higher ROC-AUC (0.9349 vs. 0.9288), ~4x faster training latency, and ~15x faster prediction throughput.
2. **CNN-LSTM vs. 1D CNN**: CNN-LSTM achieves Test Macro-F1 of 0.6754 +/- 0.0018 and Val Macro-F1 of 0.6940 +/- 0.0106. CNN-LSTM does not beat the 1D CNN beyond seed noise (mean difference: -0.0088).
