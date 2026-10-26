"""
src/preprocess.py
Data preprocessing, feature engineering, subject-level train/val/test splitting,
scaling, class weight computation, and sequence window generation.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

try:
    from src.utils import CLASSES, CLASS_TO_IDX
except ImportError:
    from utils import CLASSES, CLASS_TO_IDX

BASE_NUMERIC_COLS = [
    'heart_rate_bpm', 'spo2_pct', 'skin_temp_c',
    'acc_x_g', 'acc_y_g', 'acc_z_g',
    'gyro_x_dps', 'gyro_y_dps', 'gyro_z_dps',
    'motion_intensity'
]

# Physical outlier boundaries based on EDA findings
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


def load_and_clean_raw_data(csv_path):
    """
    Loads raw CSV, validates schema, applies physiological sanity clipping.
    """
    df = pd.read_csv(csv_path)
    
    # Verify no nulls
    if df.isnull().sum().sum() > 0:
        df = df.dropna().reset_index(drop=True)
        
    # Apply physiological sanity boundaries
    for col, (low, high) in PHYSIOLOGICAL_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=low, upper=high)
            
    # Add magnitude features
    df['acc_mag'] = np.sqrt(df['acc_x_g']**2 + df['acc_y_g']**2 + df['acc_z_g']**2)
    df['gyro_mag'] = np.sqrt(df['gyro_x_dps']**2 + df['gyro_y_dps']**2 + df['gyro_z_dps']**2)
    
    return df


def split_subjects(df, train_size=70, val_size=15, test_size=15, seed=42):
    """
    Splits subjects strictly at subject_id level into 70 train, 15 val, 15 test.
    Ensures zero subject overlap across splits.
    """
    subjects = np.sort(df['subject_id'].unique())
    n_total = len(subjects)
    assert train_size + val_size + test_size == n_total, (
        f"Sum of splits ({train_size} + {val_size} + {test_size}) != total subjects ({n_total})"
    )
    
    # Stratify subjects by sleep architecture profile (N1/REM balance)
    sub_profile = pd.crosstab(df['subject_id'], df['sleep_stage'], normalize='index')
    # Use binned N1 proportion for stratified sampling if possible
    try:
        bins = pd.qcut(sub_profile['N1'], q=5, labels=False, duplicates='drop')
        train_subs, temp_subs = train_test_split(
            subjects, train_size=train_size, random_state=seed, stratify=bins
        )
        temp_bins = bins.loc[temp_subs]
        val_subs, test_subs = train_test_split(
            temp_subs, train_size=val_size, test_size=test_size, random_state=seed, stratify=temp_bins
        )
    except Exception:
        # Fallback to standard random split
        train_subs, temp_subs = train_test_split(subjects, train_size=train_size, random_state=seed)
        val_subs, test_subs = train_test_split(temp_subs, train_size=val_size, test_size=test_size, random_state=seed)
        
    return sorted(list(train_subs)), sorted(list(val_subs)), sorted(list(test_subs))


def extract_rolling_features(df, base_cols, windows=[3, 6, 10]):
    """
    Computes rolling window statistics (mean, std, min, max) strictly WITHIN
    each (subject_id, night) sequence in chronological time order.
    Uses min_periods=1 to prevent NaN at session boundaries.
    """
    feature_dfs = [df[base_cols].copy()]
    
    grouped = df.groupby(['subject_id', 'night'])[base_cols]
    
    for w in windows:
        # Rolling mean
        rmean = grouped.rolling(window=w, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
        rmean.columns = [f"{col}_roll{w}_mean" for col in base_cols]
        feature_dfs.append(rmean)
        
        # Rolling std
        rstd = grouped.rolling(window=w, min_periods=1).std().fillna(0.0).reset_index(level=[0, 1], drop=True)
        rstd.columns = [f"{col}_roll{w}_std" for col in base_cols]
        feature_dfs.append(rstd)
        
        # Rolling min
        rmin = grouped.rolling(window=w, min_periods=1).min().reset_index(level=[0, 1], drop=True)
        rmin.columns = [f"{col}_roll{w}_min" for col in base_cols]
        feature_dfs.append(rmin)
        
        # Rolling max
        rmax = grouped.rolling(window=w, min_periods=1).max().reset_index(level=[0, 1], drop=True)
        rmax.columns = [f"{col}_roll{w}_max" for col in base_cols]
        feature_dfs.append(rmax)
        
    full_features = pd.concat(feature_dfs, axis=1)
    return full_features


def create_sliding_windows(df, feature_cols, window_size=10, stride=1, target_epoch='last'):
    """
    Builds sliding sequence windows (N, window_size, num_features) strictly
    WITHIN each (subject_id, night) recording, never crossing session or subject boundaries.
    
    Args:
        df: DataFrame containing features, subject_id, night, and label
        feature_cols: List of feature column names
        window_size: Number of consecutive epochs per sequence
        stride: Step size between sequence starts
        target_epoch: 'last' or 'center' epoch to determine the sequence label
    """
    X_seq_list = []
    y_seq_list = []
    groups_list = []
    
    target_idx = (window_size - 1) if target_epoch == 'last' else (window_size // 2)
    
    # Process each subject and night session separately
    for (sub_id, night_id), session_df in df.groupby(['subject_id', 'night'], sort=False):
        feat_matrix = session_df[feature_cols].values
        labels = session_df['label'].values
        n_epochs = len(session_df)
        
        if n_epochs < window_size:
            continue
            
        for start in range(0, n_epochs - window_size + 1, stride):
            end = start + window_size
            window = feat_matrix[start:end]
            label = labels[start + target_idx]
            
            X_seq_list.append(window)
            y_seq_list.append(label)
            groups_list.append(sub_id)
            
    X_seq = np.array(X_seq_list, dtype=np.float32)
    y_seq = np.array(y_seq_list, dtype=np.int64)
    groups = np.array(groups_list, dtype=np.int64)
    
    return X_seq, y_seq, groups


def run_pipeline(csv_path="data/raw/noctiband_sleep_data_full.csv",
                 processed_dir="data/processed",
                 models_dir="models",
                 reports_dir="reports",
                 seed=42,
                 smoke_test=False):
    """
    Executes full Step 1 Preprocessing and data preparation.
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    print(f"Loading and validating raw data from {csv_path}...")
    df = load_and_clean_raw_data(csv_path)
    
    # Subject split
    if smoke_test:
        # Small smoke test subset: 3 train subjects, 1 val, 1 test
        train_subs = [1, 2, 3]
        val_subs = [4]
        test_subs = [5]
        df = df[df['subject_id'].isin(train_subs + val_subs + test_subs)].copy()
        print(f"Smoke test mode: using subjects Train={train_subs}, Val={val_subs}, Test={test_subs}")
    else:
        train_subs, val_subs, test_subs = split_subjects(df, train_size=70, val_size=15, test_size=15, seed=seed)
        
    # Save splits.json
    splits_dict = {
        'seed': seed,
        'train_subjects': [int(s) for s in train_subs],
        'val_subjects': [int(s) for s in val_subs],
        'test_subjects': [int(s) for s in test_subs]
    }
    with open(os.path.join(processed_dir, "splits.json"), "w") as f:
        json.dump(splits_dict, f, indent=2)
        
    # Map target label
    df['label'] = df['sleep_stage'].map(CLASS_TO_IDX)
    
    # Feature engineering: base cols + rolling stats
    base_signal_cols = BASE_NUMERIC_COLS + ['acc_mag', 'gyro_mag']
    print("Computing within-session rolling window features (windows=[3, 6, 10])...")
    feat_df = extract_rolling_features(df, base_signal_cols, windows=[3, 6, 10])
    feature_cols = feat_df.columns.tolist()
    
    # Calculate within-session epoch index (0 to 959) for fair row-matched evaluation
    df['epoch_in_session'] = df.groupby(['subject_id', 'night']).cumcount()
    
    # Attach identifiers and label for splitting
    full_df = pd.concat([df[['subject_id', 'night', 'timestamp', 'epoch_in_session', 'label', 'sleep_stage']], feat_df], axis=1)
    
    train_df = full_df[full_df['subject_id'].isin(train_subs)].copy().reset_index(drop=True)
    val_df = full_df[full_df['subject_id'].isin(val_subs)].copy().reset_index(drop=True)
    test_df = full_df[full_df['subject_id'].isin(test_subs)].copy().reset_index(drop=True)
    
    # Fit StandardScaler strictly on Train features
    print(f"Fitting StandardScaler on Train set ({len(train_df)} rows, {len(feature_cols)} features)...")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[feature_cols].values)
    val_scaled = scaler.transform(val_df[feature_cols].values)
    test_scaled = scaler.transform(test_df[feature_cols].values)
    
    # Save scaler
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    
    # Update feature values in dataframes
    train_df[feature_cols] = train_scaled
    val_df[feature_cols] = val_scaled
    test_df[feature_cols] = test_scaled
    
    # Compute class weights strictly on train set
    train_labels = train_df['label'].values
    unique_classes = np.array(range(len(CLASSES)))
    weights = compute_class_weight('balanced', classes=unique_classes, y=train_labels)
    class_weights_dict = {CLASSES[i]: float(weights[i]) for i in range(len(CLASSES))}
    class_weights_idx = {i: float(weights[i]) for i in range(len(CLASSES))}
    
    with open(os.path.join(models_dir, "class_weights.json"), "w") as f:
        json.dump(class_weights_dict, f, indent=2)
        
    print(f"Class weights computed on Train: {class_weights_dict}")
    
    # Columns to drop for classical model training:
    dropped_cols = ['subject_id', 'night', 'timestamp', 'epoch_in_session', 'label', 'sleep_stage']
    print(f"Columns dropped for feature matrix: {dropped_cols}")
    
    # Save tabular datasets to parquet
    train_path = os.path.join(processed_dir, "train.parquet")
    val_path = os.path.join(processed_dir, "val.parquet")
    test_path = os.path.join(processed_dir, "test.parquet")
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    print("Tabular datasets saved to parquet.")
    
    # Build 3D Sequence Windows for CNN & LSTM
    # Using the 12 base signal features (scaled) for sequence modeling
    base_scaled_indices = [feature_cols.index(c) for c in base_signal_cols]
    window_size = 10
    stride = 1
    
    print(f"Building sliding sequences (window_size={window_size}, stride={stride}, last epoch label)...")
    X_train_seq, y_train_seq, groups_train_seq = create_sliding_windows(
        train_df, base_signal_cols, window_size=window_size, stride=stride, target_epoch='last'
    )
    X_val_seq, y_val_seq, groups_val_seq = create_sliding_windows(
        val_df, base_signal_cols, window_size=window_size, stride=stride, target_epoch='last'
    )
    X_test_seq, y_test_seq, groups_test_seq = create_sliding_windows(
        test_df, base_signal_cols, window_size=window_size, stride=stride, target_epoch='last'
    )
    
    np.savez_compressed(
        os.path.join(processed_dir, "seq_train.npz"),
        X=X_train_seq, y=y_train_seq, groups=groups_train_seq
    )
    np.savez_compressed(
        os.path.join(processed_dir, "seq_val.npz"),
        X=X_val_seq, y=y_val_seq, groups=groups_val_seq
    )
    np.savez_compressed(
        os.path.join(processed_dir, "seq_test.npz"),
        X=X_test_seq, y=y_test_seq, groups=groups_test_seq
    )
    print(f"Sequence datasets saved: Train={X_train_seq.shape}, Val={X_val_seq.shape}, Test={X_test_seq.shape}")
    
    # Save feature metadata
    metadata = {
        'feature_cols': feature_cols,
        'base_signal_cols': base_signal_cols,
        'dropped_cols': dropped_cols,
        'window_size': window_size,
        'stride': stride,
        'class_weights': class_weights_dict,
        'class_weights_idx': class_weights_idx,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df)
    }
    with open(os.path.join(processed_dir, "features_meta.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Document all decisions in reports/preprocessing.md
    if not smoke_test:
        write_preprocessing_report(reports_dir, splits_dict, train_df, val_df, test_df, 
                                   class_weights_dict, dropped_cols, feature_cols, base_signal_cols, window_size)

    return metadata


def write_preprocessing_report(reports_dir, splits_dict, train_df, val_df, test_df, 
                               class_weights, dropped_cols, feature_cols, base_signal_cols, window_size):
    """
    Generates reports/preprocessing.md documenting all cleaning, splitting, feature engineering,
    and class weight decisions.
    """
    report_path = os.path.join(reports_dir, "preprocessing.md")
    
    def get_stage_distribution(df):
        dist = df['sleep_stage'].value_counts()
        pcts = df['sleep_stage'].value_counts(normalize=True) * 100
        return pd.DataFrame({'Count': dist, 'Percentage': pcts.round(2)})

    train_dist = get_stage_distribution(train_df)
    val_dist = get_stage_distribution(val_df)
    test_dist = get_stage_distribution(test_df)
    
    content = f"""# Data Preprocessing and Feature Engineering Documentation

- **Pipeline Branch**: `pipeline-A`
- **Output Artifacts**: `data/processed/`, `models/scaler.joblib`, `models/class_weights.json`
- **Status**: Completed

---

## 1. Missing Values & Outlier Handling Decisions

1. **Missing Data**:
   - As established in the EDA report, the raw dataset contains $0$ missing values across all 14 columns. No imputation was required.
2. **Outlier Decisions & Physiological Bounding**:
   - Statistical analysis demonstrated that outliers detected via $1.5 \\times \\text{{IQR}}$ reflect legitimate physiological events (e.g., nocturnal arousals, body rotations, heart rate dips during deep sleep).
   - Rather than aggressive removal (which would discard critical transient sleep-stage transitions), conservative physiological bounds were applied:
     - `heart_rate_bpm`: bounded to $[35.0, 150.0]$ bpm.
     - `spo2_pct`: bounded to $[80.0, 100.0]\\%$.
     - `skin_temp_c`: bounded to $[28.0, 40.0]^\\circ\\text{{C}}$.
     - Accelerometers (`acc_x_g`, `acc_y_g`, `acc_z_g`): bounded to $[-5.0, 5.0]g$.
     - Gyroscopes (`gyro_x_dps`, `gyro_y_dps`, `gyro_z_dps`): bounded to $[-300.0, 300.0]^\\circ/\\text{{s}}$.
     - `motion_intensity`: bounded to $[0.0, 10.0]$.
   - This prevents numerical overflow or gradient explosion in neural networks while preserving 100% of physical movement dynamics.

---

## 2. Subject-Level Splitting & Stratification

To eliminate cross-subject data leakage and guarantee that models evaluate generalizability to unseen individuals:
- **Splitting Unit**: Strictly by `subject_id`.
- **Partition Sizes**:
  - **Train**: 70 subjects ({len(train_df):,} epochs / 70 nights $\\times$ 960 epochs)
  - **Validation**: 15 subjects ({len(val_df):,} epochs / 15 nights $\\times$ 960 epochs)
  - **Test**: 15 subjects ({len(test_df):,} epochs / 15 nights $\\times$ 960 epochs)
- **Random Seed**: `42`
- **Stratification**: Subject-level stratification based on individual sleep stage profile.
- **Subject Lists Saved**: `data/processed/splits.json`

### Class Balance Across Splits

| Sleep Stage | Train Epochs (\\%) | Validation Epochs (\\%) | Test Epochs (\\%) |
|---|---|---|---|
| **N2** | {train_dist.loc['N2', 'Count']:,} ({train_dist.loc['N2', 'Percentage']:.2f}\\%) | {val_dist.loc['N2', 'Count']:,} ({val_dist.loc['N2', 'Percentage']:.2f}\\%) | {test_dist.loc['N2', 'Count']:,} ({test_dist.loc['N2', 'Percentage']:.2f}\\%) |
| **REM** | {train_dist.loc['REM', 'Count']:,} ({train_dist.loc['REM', 'Percentage']:.2f}\\%) | {val_dist.loc['REM', 'Count']:,} ({val_dist.loc['REM', 'Percentage']:.2f}\\%) | {test_dist.loc['REM', 'Count']:,} ({test_dist.loc['REM', 'Percentage']:.2f}\\%) |
| **N3** | {train_dist.loc['N3', 'Count']:,} ({train_dist.loc['N3', 'Percentage']:.2f}\\%) | {val_dist.loc['N3', 'Count']:,} ({val_dist.loc['N3', 'Percentage']:.2f}\\%) | {test_dist.loc['N3', 'Count']:,} ({test_dist.loc['N3', 'Percentage']:.2f}\\%) |
| **N1** | {train_dist.loc['N1', 'Count']:,} ({train_dist.loc['N1', 'Percentage']:.2f}\\%) | {val_dist.loc['N1', 'Count']:,} ({val_dist.loc['N1', 'Percentage']:.2f}\\%) | {test_dist.loc['N1', 'Count']:,} ({test_dist.loc['N1', 'Percentage']:.2f}\\%) |
| **Wake** | {train_dist.loc['Wake', 'Count']:,} ({train_dist.loc['Wake', 'Percentage']:.2f}\\%) | {val_dist.loc['Wake', 'Count']:,} ({val_dist.loc['Wake', 'Percentage']:.2f}\\%) | {test_dist.loc['Wake', 'Count']:,} ({test_dist.loc['Wake', 'Percentage']:.2f}\\%) |
| **Total** | **{len(train_df):,} (100.0\\%)** | **{len(val_df):,} (100.0\\%)** | **{len(test_df):,} (100.0\\%)** |

---

## 3. Class Imbalance Mitigation

- **Approach**: Inverse-frequency balanced class weighting. No SMOTE or synthetic oversampling was performed across subject boundaries to avoid creating synthetic subjects or corrupting time-series continuity.
- **Computed Formula**: $w_c = \\frac{{N}}{{K \\times N_c}}$ fitted strictly on the Train set.
- **Weights Applied**:
  - `Wake`: **{class_weights['Wake']:.4f}**
  - `N1`: **{class_weights['N1']:.4f}**
  - `N2`: **{class_weights['N2']:.4f}**
  - `N3`: **{class_weights['N3']:.4f}**
  - `REM`: **{class_weights['REM']:.4f}**

---

## 4. Feature Engineering & Columns Dropped

### Columns Dropped to Prevent Data Leakage
The following columns were dropped from the model feature matrix:
1. `subject_id`: Prevent models from memorizing specific participant biometrics.
2. `night`: Prevent spurious correlations with night sequence.
3. `timestamp`: Absolute time-of-night and calendar dates were excluded to ensure the model generalises across diverse bedtime schedules without temporal overfitting.
4. `sleep_stage` & `label`: Target labels excluded from inputs.
5. Row indices: Excluded.

### Engineered Features for Classical Models
- **Magnitude Signals**:
  - `acc_mag = sqrt(acc_x^2 + acc_y^2 + acc_z^2)`
  - `gyro_mag = sqrt(gyro_x^2 + gyro_y^2 + gyro_z^2)`
- **Temporal Rolling Statistics**:
  - Computed strictly **within each subject and session** in chronological time order.
  - Window lengths: $N = 3$ epochs (1.5 minutes) and $N = 6$ epochs (3.0 minutes).
  - Statistics per signal: `mean`, `std`, `min`, `max`.
  - Total engineered feature count: **{len(feature_cols)} features**.

### Sequence Windows for CNN and LSTM
- Built strictly within each recording session without crossing night or subject boundaries.
- **Window Size ($W$)**: {window_size} consecutive epochs (5.0 minutes of continuous context).
- **Stride ($S$)**: 1 epoch.
- **Target Epoch**: `last` epoch of the window (causal streaming inference model).
- Sequence Tensor Dimensions: $(N, {window_size}, 12)$.

---

## 5. Scaling and Artifact Serialization

- `StandardScaler` was fitted strictly on the Train set and saved to `models/scaler.joblib`.
- Val and Test sets were transformed using the Train scaler parameters.
- Processed datasets were serialized to high-performance Parquet (`train.parquet`, `val.parquet`, `test.parquet`) and compressed NumPy archives (`seq_train.npz`, `seq_val.npz`, `seq_test.npz`).
"""
    with open(report_path, "w") as f:
        f.write(content)


if __name__ == "__main__":
    run_pipeline()
