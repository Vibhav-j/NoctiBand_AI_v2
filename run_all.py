"""
run_all.py
End-to-end pipeline runner for NoctiBand AI.
Executes:
1. Smoke test on a tiny subset to verify pipeline integrity.
2. Full preprocessing & feature engineering.
3. Sequential model training & git commits:
   - Logistic Regression
   - Random Forest
   - SVM (Calibrated LinearSVC)
   - XGBoost
   - 1D CNN
   - LSTM
4. Cross-model comparison and optimal model selection.
5. Verification of predict.py inference.
"""

import os
import sys
import json
import subprocess
import time
import numpy as np
import pandas as pd

from src.preprocess import run_pipeline
from src.models.classical import (
    train_logistic_regression,
    train_random_forest,
    train_svm,
    train_xgboost
)
from src.models.neural import (
    train_1d_cnn,
    train_lstm
)
from src.compare_and_select import compare_models


def git_commit(message):
    """
    Commits changes with the exact required commit message on current branch.
    """
    try:
        subprocess.run(["git", "add", "."], check=True)
        # Check if there are staged changes
        res = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if res.returncode != 0:
            subprocess.run(["git", "commit", "-m", message], check=True)
            print(f"Git committed: '{message}'")
        else:
            print(f"No changes to commit for '{message}'")
    except Exception as e:
        print(f"Warning: Git commit failed: {e}")


def run_smoke_test():
    """
    Executes a fast end-to-end smoke test on 3 train subjects and 1 val subject.
    """
    print("\n" + "="*70)
    print("STEP 2: RUNNING SMOKE TEST ON TINY SUBSET")
    print("="*70)
    
    smoke_processed_dir = "data/processed_smoke"
    smoke_reports_dir = "reports/smoke_test"
    smoke_models_dir = "models/smoke_test"
    
    # 1. Run preprocessing in smoke mode
    meta = run_pipeline(
        csv_path="data/raw/noctiband_sleep_data_full.csv",
        processed_dir=smoke_processed_dir,
        models_dir=smoke_models_dir,
        reports_dir=smoke_reports_dir,
        smoke_test=True
    )
    
    # Load tabular data
    train_df = pd.read_parquet(os.path.join(smoke_processed_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(smoke_processed_dir, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(smoke_processed_dir, "test.parquet"))
    
    feature_cols = meta['feature_cols']
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    groups_train = train_df['subject_id'].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df['label'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    
    # Load sequences
    seq_train = np.load(os.path.join(smoke_processed_dir, "seq_train.npz"))
    seq_val = np.load(os.path.join(smoke_processed_dir, "seq_val.npz"))
    seq_test = np.load(os.path.join(smoke_processed_dir, "seq_test.npz"))
    
    # Test each model briefly
    print("Testing Logistic Regression...")
    _ = train_logistic_regression(
        X_train, y_train, groups_train, X_val, y_val, X_test, y_test,
        feature_cols, reports_dir=smoke_reports_dir, models_dir=smoke_models_dir, smoke_test=True
    )
    
    print("Testing Random Forest...")
    _ = train_random_forest(
        X_train, y_train, groups_train, X_val, y_val, X_test, y_test,
        feature_cols, reports_dir=smoke_reports_dir, models_dir=smoke_models_dir, smoke_test=True
    )
    
    print("Testing SVM...")
    _ = train_svm(
        X_train, y_train, groups_train, X_val, y_val, X_test, y_test,
        feature_cols, reports_dir=smoke_reports_dir, models_dir=smoke_models_dir, smoke_test=True
    )
    
    print("Testing XGBoost...")
    _ = train_xgboost(
        X_train, y_train, groups_train, X_val, y_val, X_test, y_test,
        feature_cols, meta['class_weights'], reports_dir=smoke_reports_dir, models_dir=smoke_models_dir, smoke_test=True
    )
    
    print("Testing 1D CNN...")
    _ = train_1d_cnn(
        seq_train['X'], seq_train['y'], seq_train['groups'],
        seq_val['X'], seq_val['y'],
        seq_test['X'], seq_test['y'],
        meta['class_weights'], reports_dir=smoke_reports_dir, models_dir=smoke_models_dir, smoke_test=True
    )
    
    print("Testing LSTM...")
    _ = train_lstm(
        seq_train['X'], seq_train['y'], seq_train['groups'],
        seq_val['X'], seq_val['y'],
        seq_test['X'], seq_test['y'],
        meta['class_weights'], reports_dir=smoke_reports_dir, models_dir=smoke_models_dir, smoke_test=True
    )
    
    print("\n>>> SMOKE TEST PASSED SUCCESSFULLY ON ALL 6 ARCHITECTURES! <<<")


def run_full_pipeline():
    """
    Executes the full pipeline on complete 100-subject dataset.
    """
    print("\n" + "="*70)
    print("RUNNING FULL NOCTIBAND ML PIPELINE")
    print("="*70)
    
    processed_dir = "data/processed"
    reports_dir = "reports"
    models_dir = "models"
    
    # Step 1: Preprocess
    print("\n--- STEP 1: PREPROCESSING FULL DATASET ---")
    meta = run_pipeline(
        csv_path="data/raw/noctiband_sleep_data_full.csv",
        processed_dir=processed_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        seed=42,
        smoke_test=False
    )
    
    # Load full datasets
    train_df = pd.read_parquet(os.path.join(processed_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(processed_dir, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(processed_dir, "test.parquet"))
    
    feature_cols = meta['feature_cols']
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    groups_train = train_df['subject_id'].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df['label'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    
    # Masks for fair row-matched evaluation
    val_mask = val_df['epoch_in_session'].values >= 9
    test_mask = test_df['epoch_in_session'].values >= 9
    
    seq_train = np.load(os.path.join(processed_dir, "seq_train.npz"))
    seq_val = np.load(os.path.join(processed_dir, "seq_val.npz"))
    seq_test = np.load(os.path.join(processed_dir, "seq_test.npz"))
    
    all_results = []
    
    # --- MODEL 1: Logistic Regression ---
    res_lr = train_logistic_regression(
        X_train, y_train, groups_train, X_val, y_val, X_test, y_test,
        feature_cols, reports_dir=reports_dir, models_dir=models_dir, smoke_test=False,
        val_mask=val_mask, test_mask=test_mask
    )
    all_results.append(res_lr)
    git_commit("model: Logistic Regression")
    
    # --- MODEL 2: Random Forest ---
    res_rf = train_random_forest(
        X_train, y_train, groups_train, X_val, y_val, X_test, y_test,
        feature_cols, reports_dir=reports_dir, models_dir=models_dir, smoke_test=False,
        val_mask=val_mask, test_mask=test_mask
    )
    all_results.append(res_rf)
    git_commit("model: Random Forest")
    
    # --- MODEL 3: SVM ---
    res_svm = train_svm(
        X_train, y_train, groups_train, X_val, y_val, X_test, y_test,
        feature_cols, reports_dir=reports_dir, models_dir=models_dir, smoke_test=False,
        val_mask=val_mask, test_mask=test_mask
    )
    all_results.append(res_svm)
    git_commit("model: SVM")
    
    # --- MODEL 4: XGBoost ---
    res_xgb = train_xgboost(
        X_train, y_train, groups_train, X_val, y_val, X_test, y_test,
        feature_cols, meta['class_weights'], reports_dir=reports_dir, models_dir=models_dir, smoke_test=False,
        val_mask=val_mask, test_mask=test_mask
    )
    all_results.append(res_xgb)
    git_commit("model: XGBoost")
    
    # --- MODEL 5: 1D CNN ---
    res_cnn = train_1d_cnn(
        seq_train['X'], seq_train['y'], seq_train['groups'],
        seq_val['X'], seq_val['y'],
        seq_test['X'], seq_test['y'],
        meta['class_weights'], reports_dir=reports_dir, models_dir=models_dir, smoke_test=False
    )
    all_results.append(res_cnn)
    git_commit("model: 1D CNN")
    
    # --- MODEL 6: LSTM ---
    res_lstm = train_lstm(
        seq_train['X'], seq_train['y'], seq_train['groups'],
        seq_val['X'], seq_val['y'],
        seq_test['X'], seq_test['y'],
        meta['class_weights'], reports_dir=reports_dir, models_dir=models_dir, smoke_test=False
    )
    all_results.append(res_lstm)
    git_commit("model: LSTM")
    
    # --- STEP 5: COMPARE AND SELECT ---
    print("\n--- STEP 5: COMPARING MODELS & SELECTING WINNER ---")
    winner = compare_models(all_results, reports_dir=reports_dir, models_dir=models_dir)
    git_commit("evaluation: comparison, benchmarking and best model selection")
    
    # Test predict.py on test set slice
    print("\n--- VERIFYING PREDICT.PY END-TO-END ---")
    sample_csv = "data/raw/sample_test_night.csv"
    pred_csv = "reports/sample_predictions.csv"
    sample_df = df_test_sample = pd.read_csv("data/raw/noctiband_sleep_data_full.csv", nrows=960)
    sample_df.to_csv(sample_csv, index=False)
    
    cmd = [sys.executable, "predict.py", "--input", sample_csv, "--output", pred_csv]
    subprocess.run(cmd, check=True)
    print("predict.py verified successfully!")
    
    print("\n" + "="*70)
    print(f"PIPELINE COMPLETE! BEST MODEL: {winner['name']}")
    print(f"Test Macro-F1: {winner['test_f1']:.4f} | Cohen's Kappa: {winner['test_kappa']:.4f} | Test Acc: {winner['test_acc']*100:.2f}%")
    print("="*70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true", help="Run smoke test only")
    parser.add_argument("--full-only", action="store_true", help="Run full pipeline only")
    args = parser.parse_args()
    
    if args.smoke_only:
        run_smoke_test()
    elif args.full_only:
        run_full_pipeline()
    else:
        run_smoke_test()
        run_full_pipeline()
