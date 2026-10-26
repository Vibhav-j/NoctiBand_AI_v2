"""
predict.py
Production inference script for NoctiBand AI.
Loads the selected best model from models/best_model/ and generates sleep stage
predictions for a new input CSV file.

Usage:
    python predict.py --input path/to/input.csv --output path/to/predictions.csv
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd

CLASSES = ['Wake', 'N1', 'N2', 'N3', 'REM']
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

BASE_NUMERIC_COLS = [
    'heart_rate_bpm', 'spo2_pct', 'skin_temp_c',
    'acc_x_g', 'acc_y_g', 'acc_z_g',
    'gyro_x_dps', 'gyro_y_dps', 'gyro_z_dps',
    'motion_intensity'
]

PHYSIOLOGICAL_BOUNDS = {
    'heart_rate_bpm': (35.0, 150.0),
    'spo2_pct': (80.0, 100.0),
    'skin_temp_c': (28.0, 40.0),
    'acc_x_g': (-5.0, 5.0),
    'acc_y_g': (-5.0, 5.0),
    'acc_z_g': (-5.0, 5.0),
    'gyro_x_dps': (-300.0, 300.0),
    'gyro_y_dps': (-300.0, 300.0),
    'gyro_z_dps': (-300.0, 300.0),
    'motion_intensity': (0.0, 10.0)
}


def load_best_model_package(best_model_dir="models/best_model"):
    """
    Loads model, scaler, and metadata from the best_model bundle.
    """
    meta_path = os.path.join(best_model_dir, "best_model_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing best_model metadata at {meta_path}. Run training pipeline first.")
        
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    scaler_path = os.path.join(best_model_dir, "scaler.joblib")
    scaler = joblib.load(scaler_path)
    
    with open(os.path.join(best_model_dir, "features_meta.json"), "r") as f:
        feat_meta = json.load(f)
        
    model_filename = meta['model_filename']
    model_path = os.path.join(best_model_dir, model_filename)
    
    if meta['is_neural_sequence']:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
    else:
        model = joblib.load(model_path)
        
    return model, scaler, meta, feat_meta


def preprocess_input(df_raw, scaler, feat_meta, is_neural_sequence=False):
    """
    Prepares input dataframe for model prediction.
    """
    df = df_raw.copy()
    
    # 1. Bounding
    for col, (low, high) in PHYSIOLOGICAL_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=low, upper=high)
            
    # 2. Magnitudes
    df['acc_mag'] = np.sqrt(df['acc_x_g']**2 + df['acc_y_g']**2 + df['acc_z_g']**2)
    df['gyro_mag'] = np.sqrt(df['gyro_x_dps']**2 + df['gyro_y_dps']**2 + df['gyro_z_dps']**2)
    
    base_signal_cols = feat_meta['base_signal_cols']
    feature_cols = feat_meta['feature_cols']
    
    # Ensure subject/night grouping columns exist or provide defaults
    if 'subject_id' not in df.columns:
        df['subject_id'] = 1
    if 'night' not in df.columns:
        df['night'] = 1

    if not is_neural_sequence:
        # Classical model: Compute rolling features
        feature_dfs = [df[base_signal_cols].copy()]
        grouped = df.groupby(['subject_id', 'night'])[base_signal_cols]
        for w in [3, 6]:
            rmean = grouped.rolling(window=w, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
            rmean.columns = [f"{col}_roll{w}_mean" for col in base_signal_cols]
            feature_dfs.append(rmean)
            
            rstd = grouped.rolling(window=w, min_periods=1).std().fillna(0.0).reset_index(level=[0, 1], drop=True)
            rstd.columns = [f"{col}_roll{w}_std" for col in base_signal_cols]
            feature_dfs.append(rstd)
            
            rmin = grouped.rolling(window=w, min_periods=1).min().reset_index(level=[0, 1], drop=True)
            rmin.columns = [f"{col}_roll{w}_min" for col in base_signal_cols]
            feature_dfs.append(rmin)
            
            rmax = grouped.rolling(window=w, min_periods=1).max().reset_index(level=[0, 1], drop=True)
            rmax.columns = [f"{col}_roll{w}_max" for col in base_signal_cols]
            feature_dfs.append(rmax)
            
        full_feat_df = pd.concat(feature_dfs, axis=1)
        X = scaler.transform(full_feat_df[feature_cols].values)
        return X, None
    else:
        # Neural sequence model: Scale base features and build sliding windows
        # Extract base scaled features
        # Note: Scaler was fitted on all feature_cols, so we project base columns
        base_indices = [feature_cols.index(c) for c in base_signal_cols]
        # To scale base features identically, construct full dummy array
        dummy = np.zeros((len(df), len(feature_cols)))
        dummy[:, :len(base_signal_cols)] = df[base_signal_cols].values
        # Or more cleanly: use base indices from mean_ and scale_
        mean_base = scaler.mean_[:len(base_signal_cols)]
        scale_base = scaler.scale_[:len(base_signal_cols)]
        scaled_base = (df[base_signal_cols].values - mean_base) / scale_base
        
        window_size = feat_meta.get('window_size', 10)
        X_seq_list = []
        valid_indices = []
        
        for (sub_id, night_id), session_df in df.groupby(['subject_id', 'night'], sort=False):
            session_indices = session_df.index.tolist()
            session_vals = scaled_base[session_indices]
            n_epochs = len(session_df)
            
            if n_epochs < window_size:
                continue
            for start in range(0, n_epochs - window_size + 1):
                end = start + window_size
                X_seq_list.append(session_vals[start:end])
                valid_indices.append(session_indices[end - 1])
                
        return np.array(X_seq_list, dtype=np.float32), valid_indices


def predict_csv(input_csv, output_csv, best_model_dir="models/best_model"):
    """
    Main prediction pipeline.
    """
    print(f"Reading input data from {input_csv}...")
    df_raw = pd.read_csv(input_csv)
    print(f"Input data shape: {df_raw.shape}")
    
    model, scaler, meta, feat_meta = load_best_model_package(best_model_dir)
    print(f"Loaded best model: {meta['winner_model_name']} from {best_model_dir}")
    
    is_neural = meta['is_neural_sequence']
    X, valid_indices = preprocess_input(df_raw, scaler, feat_meta, is_neural_sequence=is_neural)
    
    if not is_neural:
        preds = model.predict(X)
        probs = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        
        out_df = df_raw.copy()
        out_df['predicted_stage'] = [IDX_TO_CLASS[p] for p in preds]
        if probs is not None:
            for i, c in enumerate(CLASSES):
                out_df[f'prob_{c}'] = probs[:, i].round(4)
    else:
        probs = model.predict(X, batch_size=512, verbose=0)
        preds = np.argmax(probs, axis=1)
        
        out_df = df_raw.loc[valid_indices].copy().reset_index(drop=True)
        out_df['predicted_stage'] = [IDX_TO_CLASS[p] for p in preds]
        for i, c in enumerate(CLASSES):
            out_df[f'prob_{c}'] = probs[:, i].round(4)
            
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Predictions successfully saved to {output_csv} ({len(out_df)} rows).")
    print(f"Predicted class distribution:\n{out_df['predicted_stage'].value_counts()}")
    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NoctiBand AI Sleep Stage Predictor")
    parser.add_argument("--input", required=True, help="Path to input raw CSV file")
    parser.add_argument("--output", required=True, help="Path to destination predictions CSV file")
    parser.add_argument("--best_model_dir", default="models/best_model", help="Directory holding best model artifacts")
    
    args = parser.parse_args()
    predict_csv(args.input, args.output, args.best_model_dir)
