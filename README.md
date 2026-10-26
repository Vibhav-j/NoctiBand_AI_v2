# NoctiBand AI v2

A machine learning pipeline for sleep data analysis and sleep stage classification using NoctiBand sensor recordings.

## Project Structure

```
NoctiBand_AI_v2/
├── data/
│   ├── raw/          # Raw input data (e.g., noctiband_sleep_data_full.csv)
│   └── processed/    # Cleaned and preprocessed datasets
├── models/           # Trained models and checkpoints
├── notebooks/        # Jupyter notebooks for exploration and experimentation
├── reports/          # Markdown reports, EDA, and evaluation artifacts
├── src/              # Reusable Python modules and pipeline code
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup & Installation

1. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
