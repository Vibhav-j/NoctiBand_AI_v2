# Data Preprocessing and Feature Engineering Documentation

- **Pipeline Branch**: `pipeline-A`
- **Output Artifacts**: `data/processed/`, `models/scaler.joblib`, `models/class_weights.json`
- **Status**: Completed

---

## 1. Missing Values & Outlier Handling Decisions

1. **Missing Data**:
   - As established in the EDA report, the raw dataset contains $0$ missing values across all 14 columns. No imputation was required.
2. **Outlier Decisions & Physiological Bounding**:
   - Statistical analysis demonstrated that outliers detected via $1.5 \times \text{IQR}$ reflect legitimate physiological events (e.g., nocturnal arousals, body rotations, heart rate dips during deep sleep).
   - Rather than aggressive removal (which would discard critical transient sleep-stage transitions), conservative physiological bounds were applied:
     - `heart_rate_bpm`: bounded to $[35.0, 150.0]$ bpm.
     - `spo2_pct`: bounded to $[80.0, 100.0]\%$.
     - `skin_temp_c`: bounded to $[28.0, 40.0]^\circ\text{C}$.
     - Accelerometers (`acc_x_g`, `acc_y_g`, `acc_z_g`): bounded to $[-5.0, 5.0]g$.
     - Gyroscopes (`gyro_x_dps`, `gyro_y_dps`, `gyro_z_dps`): bounded to $[-300.0, 300.0]^\circ/\text{s}$.
     - `motion_intensity`: bounded to $[0.0, 10.0]$.
   - This prevents numerical overflow or gradient explosion in neural networks while preserving 100% of physical movement dynamics.

---

## 2. Subject-Level Splitting & Stratification

To eliminate cross-subject data leakage and guarantee that models evaluate generalizability to unseen individuals:
- **Splitting Unit**: Strictly by `subject_id`.
- **Partition Sizes**:
  - **Train**: 70 subjects (470,400 epochs / 70 nights $\times$ 960 epochs)
  - **Validation**: 15 subjects (100,800 epochs / 15 nights $\times$ 960 epochs)
  - **Test**: 15 subjects (100,800 epochs / 15 nights $\times$ 960 epochs)
- **Random Seed**: `42`
- **Stratification**: Subject-level stratification based on individual sleep stage profile.
- **Subject Lists Saved**: `data/processed/splits.json`

### Class Balance Across Splits

| Sleep Stage | Train Epochs (\%) | Validation Epochs (\%) | Test Epochs (\%) |
|---|---|---|---|
| **N2** | 241,475 (51.33\%) | 51,983 (51.57\%) | 51,609 (51.20\%) |
| **REM** | 103,563 (22.02\%) | 22,183 (22.01\%) | 22,243 (22.07\%) |
| **N3** | 83,475 (17.75\%) | 17,664 (17.52\%) | 17,912 (17.77\%) |
| **N1** | 23,257 (4.94\%) | 4,997 (4.96\%) | 5,012 (4.97\%) |
| **Wake** | 18,630 (3.96\%) | 3,973 (3.94\%) | 4,024 (3.99\%) |
| **Total** | **470,400 (100.0\%)** | **100,800 (100.0\%)** | **100,800 (100.0\%)** |

---

## 3. Class Imbalance Mitigation

- **Approach**: Inverse-frequency balanced class weighting. No SMOTE or synthetic oversampling was performed across subject boundaries to avoid creating synthetic subjects or corrupting time-series continuity.
- **Computed Formula**: $w_c = \frac{N}{K \times N_c}$ fitted strictly on the Train set.
- **Weights Applied**:
  - `Wake`: **5.0499**
  - `N1`: **4.0452**
  - `N2`: **0.3896**
  - `N3`: **1.1270**
  - `REM`: **0.9084**

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
  - Total engineered feature count: **156 features**.

### Sequence Windows for CNN and LSTM
- Built strictly within each recording session without crossing night or subject boundaries.
- **Window Size ($W$)**: 10 consecutive epochs (5.0 minutes of continuous context).
- **Stride ($S$)**: 1 epoch.
- **Target Epoch**: `last` epoch of the window (causal streaming inference model).
- Sequence Tensor Dimensions: $(N, 10, 12)$.

---

## 5. Scaling and Artifact Serialization

- `StandardScaler` was fitted strictly on the Train set and saved to `models/scaler.joblib`.
- Val and Test sets were transformed using the Train scaler parameters.
- Processed datasets were serialized to high-performance Parquet (`train.parquet`, `val.parquet`, `test.parquet`) and compressed NumPy archives (`seq_train.npz`, `seq_val.npz`, `seq_test.npz`).
