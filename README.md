# NoctiBand AI v2

A production-grade machine learning pipeline for automated sleep stage classification using NoctiBand multi-sensor wearable data (triaxial accelerometry, triaxial gyroscopy, heart rate, blood oxygen saturation, skin temperature, and motion intensity).

---

## 1. Project Overview & Architecture

The pipeline processes raw 30-second epoch sensor recordings across 5 polysomnography (PSG) sleep stages:
- **Wake**
- **N1** (Light sleep stage 1)
- **N2** (Light sleep stage 2)
- **N3** (Deep / slow-wave sleep)
- **REM** (Rapid Eye Movement)

### Data Authenticity & Synthetic Dataset Notice
Statistical audit indicates that `noctiband_sleep_data_full.csv` is a synthetic or simulated dataset rather than naturalistic clinical polysomnography recordings. Concrete evidence includes:
1. **Inflexible Session Lengths**: Every single recording across all 100 participants and 7 nights comprises exactly 960 epochs (standard deviation $= 0.0$), representing an artificial, fixed 8.0-hour window ($22:30:00$ to $06:29:30$) without any early wakeups, sleep onset delays, or sensor disconnects.
2. **Unrealistically Homogeneous Sleep Architecture**: Individual sleep stage proportions are virtually identical across all 100 participants (e.g., N2 is $51.35\% \pm 0.94\%$, REM is $22.02\% \pm 0.65\%$, N3 is $17.72\% \pm 0.55\%$, N1 is $4.95\% \pm 0.32\%$, and Wake is $3.96\% \pm 0.39\%$), whereas real clinical cohorts show pronounced heterogeneity due to age, sleep efficiency, and respiratory disturbance.
3. **Rigid Markov Transition Structure**: State transitions follow an idealized synthetic automaton: N1 transitions strictly to N1 ($88.3\%$) or N2 ($11.7\%$) with $0.0\%$ direct transitions to Wake, REM, or N3; N2 never transitions directly to Wake ($0.0\%$); and Wake transitions exclusively to Wake ($90.2\%$) or N1 ($9.8\%$). Real human hypnograms feature frequent brief awakenings and direct arousals from all stages into Wake.
4. **Sensor Perfection & Artificial Chronology**: The dataset exhibits zero missing timestamps, zero motion-induced optical sensor dropouts, constant 30.000s sampling intervals over two calendar years, and synthetic chronological chaining where each subject's timeline starts immediately upon conclusion of the preceding subject's recording.

Consequently, while the pipeline establishes rigorous, leak-free methodological benchmarks, performance metrics obtained on this simulated benchmark may not directly transfer to noisy, real-world ambulatory or clinical sleep monitoring without fine-tuning on genuine patient data.

---

## 2. Repository Structure

```
NoctiBand_AI_v2/
├── data/
│   ├── raw/                 # Raw PSG & sensor data (gitignored: noctiband_sleep_data_full.csv)
│   └── processed/           # Processed parquet & sequence arrays (.npz) (gitignored)
├── models/                  # Saved model checkpoints & scalers (gitignored)
│   ├── logistic_regression/
│   ├── random_forest/
│   ├── svm/
│   ├── xgboost/
│   ├── 1d_cnn/
│   ├── lstm/
│   └── best_model/          # Winning model bundle staged for production inference
├── notebooks/               # Interactive exploration notebooks
├── reports/
│   ├── eda.md               # Exploratory data analysis report
│   ├── preprocessing.md     # Preprocessing & feature engineering report
│   ├── comparison.csv       # Side-by-side benchmark metrics across all models
│   ├── comparison.md        # Formatted markdown comparison report
│   ├── best_model.md        # Model selection and architecture tradeoff analysis
│   ├── seed_check.md        # Multi-seed stability analysis (1D CNN vs. XGBoost)
│   ├── audit.md             # Leakage & data integrity verification audit
│   ├── old/                 # Historical pre-alignment benchmark reports
│   ├── *_comparison.png     # Benchmark visualization bar charts
│   └── <model_name>/        # Per-model confusion matrices, classification reports, curves
├── src/
│   ├── preprocess.py        # Data cleaning, subject split, feature engineering
│   ├── utils.py             # Evaluation metrics and visualization helpers
│   ├── audit.py             # Pipeline leakage and integrity audit script
│   ├── seed_check.py        # Multi-seed stability test runner
│   ├── compare_and_select.py# Multi-model benchmarking and winner selection
│   └── models/
│       ├── classical.py     # Logistic Regression, Random Forest, SVM, XGBoost
│       └── neural.py        # 1D CNN & LSTM sequence models
├── predict.py               # Standalone production inference script
├── run_all.py               # End-to-end pipeline orchestrator
├── README.md
└── requirements.txt
```

---

## 3. Final Benchmark Results (Row-Matched Evaluation)

Models were evaluated on the exact same row instances (dropping the first 9 epochs of each session to match the 10-epoch sliding receptive field of sequence architectures, yielding 99,855 matched evaluation epochs across 15 unseen test subjects).

| Model Architecture | CV Macro-F1 (Mean $\pm$ Std) | Test Acc (%) | Test Macro-F1 | Cohen's Kappa | ROC-AUC (OvR) | N1 Recall (%) | REM Recall (%) | Gen Gap (F1 %) | Train Time (s) | Latency (ms/epoch) | Model Size (KB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **1D CNN** 🏆 | N/A* | 72.77% | **0.6896** | 0.6023 | 0.9288 | 60.31% | **78.00%** | 9.11% | 139.3s | 0.029 ms | **665.9 KB** |
| **XGBoost** | $0.6664 \pm 0.0384$ | **73.37%** | 0.6795 | **0.6116** | **0.9349** | 57.21% | 77.79% | 7.32% | **31.9s** | **0.002 ms** | 3,292.5 KB |
| **Logistic Regression** | $0.6492 \pm 0.0200$ | 69.36% | 0.6596 | 0.5628 | 0.9276 | **61.54%** | 75.33% | **0.69%** | 61.9s | < 0.001 ms | 7.0 KB |
| **SVM (LinearSVC)** | **$0.6673 \pm 0.0122$** | 74.39% | 0.6539 | 0.5753 | 0.9225 | 32.54% | 73.79% | 0.91% | 776.9s | 0.002 ms | 21.3 KB |
| **LSTM** | N/A* | 66.13% | 0.6269 | 0.5146 | 0.9111 | 53.27% | 68.05% | 4.17% | 114.6s | 0.035 ms | 346.9 KB |
| **Random Forest** | $0.5945 \pm 0.0278$ | 67.72% | 0.6061 | 0.5366 | 0.9069 | 49.04% | 71.50% | 5.71% | 21.4s | 0.016 ms | 15,491.6 KB |

`*` *GroupKFold 3-fold full training was omitted for neural networks due to CPU training latency; early stopping was performed on the Validation split. Sample standard deviation ($ddof=1$) is reported.*

### 1D CNN vs. XGBoost Tradeoffs (Effectively Tied)
The 1D CNN (selected on validation macro-F1) leads XGBoost by about 0.01 in test macro-F1 (0.6896 vs. 0.6795), but XGBoost outperforms the 1D CNN in overall test accuracy (73.37% vs. 72.77%), Cohen's kappa (0.6116 vs. 0.6023), and ROC-AUC (0.9349 vs. 0.9288). Operationally, XGBoost trains about 4x faster (31.9s vs. 139.3s) and predicts about 15x faster (0.002 ms/epoch vs. 0.029 ms/epoch), whereas the 1D CNN model binary is smaller (665.9 KB vs. 3.3 MB).
The two architectures are effectively tied in practical performance: 1D CNN excels in compact memory footprints and continuous end-to-end representation learning, while XGBoost offers superior CPU inference throughput and rapid retraining.

---

## 4. How to Reproduce

Because raw datasets (`data/`) and model artifacts (`models/`) are excluded from Git version control via `.gitignore`, follow these steps to reproduce the pipeline from scratch:

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate # Linux/macOS

# Install locked dependencies
pip install -r requirements.txt
```

### 2. Place Raw Dataset
Place the raw CSV file in `data/raw/`:
```bash
# Expected path:
data/raw/noctiband_sleep_data_full.csv
```

### 3. Run End-to-End Pipeline
Execute the master runner to perform preprocessing, feature extraction, model training, evaluation, comparison, and best model packaging:
```bash
python run_all.py
```
*Optional Flags*:
- `python run_all.py --smoke-only`: Fast end-to-end smoke test on 3 train subjects and 1 validation subject.
- `python run_all.py --full-only`: Executes the full 100-subject pipeline directly.

### 4. Run Standalone Inference
Once the model is trained and staged in `models/best_model/`, run predictions on any raw recording:
```bash
python predict.py --input path/to/input.csv --output path/to/predictions.csv
```
