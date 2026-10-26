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

### Directory Structure

```
NoctiBand_AI_v2/
├── data/
│   ├── raw/                 # Raw PSG & sensor data (noctiband_sleep_data_full.csv)
│   └── processed/           # Processed parquet & sliding sequence arrays (.npz)
├── models/
│   ├── logistic_regression/ # Trained Logistic Regression artifact
│   ├── random_forest/       # Trained Random Forest artifact
│   ├── svm/                 # Calibrated LinearSVC artifact
│   ├── xgboost/             # Trained XGBoost artifact
│   ├── 1d_cnn/              # Trained 1D CNN Keras model
│   ├── lstm/                # Trained LSTM Keras model
│   └── best_model/          # Winning model bundle staged for production
├── notebooks/               # Interactive exploration notebooks
├── reports/
│   ├── eda.md               # Comprehensive EDA report
│   ├── preprocessing.md     # Preprocessing & feature engineering report
│   ├── comparison.csv       # Side-by-side benchmark metrics across all models
│   ├── comparison.md        # Formatted comparison report
│   ├── best_model.md        # Architectural selection report
│   ├── *_comparison.png     # Benchmark visualization bar charts
│   └── <model_name>/        # Per-model confusion matrices, classification reports, curves
├── src/
│   ├── preprocess.py        # Data cleaning, subject split, feature engineering
│   ├── utils.py             # Evaluation metrics and visualization helpers
│   ├── compare_and_select.py# Multi-model benchmarking and winner selection
│   └── models/
│       ├── classical.py     # Logistic Regression, Random Forest, SVM, XGBoost
│       └── neural.py        # 1D CNN & LSTM sequence models
├── predict.py               # Production standalone inference script
├── run_all.py               # End-to-end pipeline runner
├── README.md
└── requirements.txt
```

---

## 2. Experimental Results & Model Comparison

Models were tuned using `GroupKFold` strictly on the 70 training subjects (never exposing validation or test subjects during tuning or scaling). Evaluation was performed on 15 completely unseen subjects (100,800 epochs).

| Model Architecture | Test Macro-F1 | Test Kappa | Test Acc (%) | ROC-AUC (OvR) | N1 Recall (%) | REM Recall (%) | Generalization Gap (F1 %) | Inference Latency (ms/epoch) | Model Size (KB) |
|---|---|---|---|---|---|---|---|---|---|
| **1D CNN** (Winner) | **0.6896** | **0.6023** | **72.77%** | **0.9288** | **60.31%** | **78.00%** | **9.11%** | **0.029 ms** | **665.9 KB** |
| **LSTM** | 0.6269 | 0.5146 | 66.13% | 0.9111 | 53.27% | 68.05% | 4.17% | 0.035 ms | 346.9 KB |
| **XGBoost** | 0.5995 | 0.4936 | 64.46% | 0.8899 | 41.28% | 72.37% | 7.48% | 0.002 ms | 3,268.5 KB |
| **Logistic Regression** | 0.5886 | 0.4714 | 62.20% | 0.8899 | 45.97% | 69.75% | -1.04% | < 0.001 ms | 5.1 KB |
| **Random Forest** | 0.5645 | 0.4455 | 61.50% | 0.8651 | 33.86% | 72.25% | 21.63% | 0.020 ms | 278,901.3 KB |
| **SVM (LinearSVC)** | 0.5492 | 0.4755 | 69.22% | 0.8858 | 15.10% | 66.12% | -0.42% | 0.002 ms | 15.7 KB |

### Key Findings
1. **1D CNN** outperformed all other models across all primary clinical metrics:
   - Top **Macro-F1 (0.6896)** and **Cohen's Kappa (0.6023)**.
   - Highest recall on hard minority stages: **60.31% on N1** and **78.00% on REM**.
   - Ultra-compact footprint (**665.9 KB**) and lightning-fast inference (**0.029 ms/epoch**).
2. Deep sequence architectures leverage multi-epoch temporal context significantly better than tabular window summaries.
3. Random Forest suffered substantial overfitting on subject baselines (21.6% generalization gap) and generated a massive 278 MB binary.

---

## 3. How to Run

### Installation & Environment Setup

```bash
# 1. Activate Python virtual environment
.venv\Scripts\activate   # Windows
# source .venv/bin/activate # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt
```

### End-to-End Pipeline Execution

To run the complete pipeline (smoke test, full preprocessing, tuning, training of all 6 models, evaluation, and selection):

```bash
python run_all.py
```

Options:
- `python run_all.py --smoke-only`: Fast smoke test on 3 train subjects and 1 validation subject.
- `python run_all.py --full-only`: Runs the full 100-subject training pipeline directly.

### Standalone Inference (`predict.py`)

Run inference on any raw NoctiBand CSV recording:

```bash
python predict.py --input path/to/input.csv --output path/to/predictions.csv
```
