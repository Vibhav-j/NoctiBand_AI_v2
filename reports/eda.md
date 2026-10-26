# Exploratory Data Analysis: NoctiBand Sleep Dataset

- **Dataset File**: `data/raw/noctiband_sleep_data_full.csv`
- **File Size**: ~53.87 MB (56,486,306 bytes)
- **Report Date**: 2026-10-26
- **Pipeline Stage**: Part B - Data Inspection (Read-Only)

---

## 1. Dataset Shape, Dtypes, Missing Values, and Duplicates

- **Total Rows**: 672,000
- **Total Columns**: 14
- **Full Duplicate Rows**: 0 (0.0%)
- **Total Missing / Null Values**: 0 across all columns (100% complete)

### Column Schema & Data Types

| # | Column Name | Pandas Dtype | Missing Count | Null % | Description / Semantic Role |
|---|---|---|---|---|---|
| 0 | `subject_id` | `int64` | 0 | 0.0% | Subject identifier (1 to 100) |
| 1 | `night` | `int64` | 0 | 0.0% | Recording night index per subject (1 to 7) |
| 2 | `timestamp` | `object` (string) | 0 | 0.0% | Epoch timestamp (`YYYY-MM-DD HH:MM:SS`) |
| 3 | `heart_rate_bpm` | `float64` | 0 | 0.0% | Heart rate in beats per minute |
| 4 | `spo2_pct` | `float64` | 0 | 0.0% | Blood oxygen saturation percentage |
| 5 | `skin_temp_c` | `float64` | 0 | 0.0% | Wrist skin temperature in °C |
| 6 | `acc_x_g` | `float64` | 0 | 0.0% | Triaxial accelerometer X-axis in units of g |
| 7 | `acc_y_g` | `float64` | 0 | 0.0% | Triaxial accelerometer Y-axis in units of g |
| 8 | `acc_z_g` | `float64` | 0 | 0.0% | Triaxial accelerometer Z-axis in units of g |
| 9 | `gyro_x_dps` | `float64` | 0 | 0.0% | Triaxial gyroscope X-axis in degrees/sec |
| 10 | `gyro_y_dps` | `float64` | 0 | 0.0% | Triaxial gyroscope Y-axis in degrees/sec |
| 11 | `gyro_z_dps` | `float64` | 0 | 0.0% | Triaxial gyroscope Z-axis in degrees/sec |
| 12 | `motion_intensity` | `float64` | 0 | 0.0% | Composite wrist movement intensity index |
| 13 | `sleep_stage` | `object` (string) | 0 | 0.0% | Polysomnography (PSG) sleep stage label |

---

## 2. First 5 Rows Preview

| `subject_id` | `night` | `timestamp` | `heart_rate_bpm` | `spo2_pct` | `skin_temp_c` | `acc_x_g` | `acc_y_g` | `acc_z_g` | `gyro_x_dps` | `gyro_y_dps` | `gyro_z_dps` | `motion_intensity` | `sleep_stage` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 2025-01-02 22:30:00 | 60.6 | 96.2 | 33.82 | 0.020 | 0.054 | 0.982 | -0.35 | -0.31 | -0.09 | 0.016 | Wake |
| 1 | 1 | 2025-01-02 22:30:30 | 60.5 | 96.3 | 33.78 | 0.038 | 0.036 | 0.987 | 0.16 | 0.47 | -0.17 | 0.012 | Wake |
| 1 | 1 | 2025-01-02 22:31:00 | 59.8 | 96.4 | 33.54 | 0.017 | 0.040 | 0.991 | -0.00 | -0.08 | -0.04 | 0.008 | Wake |
| 1 | 1 | 2025-01-02 22:31:30 | 55.5 | 96.4 | 33.72 | 0.364 | 0.167 | 1.021 | -26.63 | -17.12 | 23.92 | 0.097 | Wake |
| 1 | 1 | 2025-01-02 22:32:00 | 55.8 | 96.7 | 33.76 | 0.752 | 0.305 | 0.744 | -45.72 | -2.19 | -36.79 | 0.101 | Wake |

---

## 3. Label Column & Class Distribution

### Candidate Label Columns
- **Primary Candidate**: `sleep_stage`
  - Values correspond exactly to the 5 standard American Academy of Sleep Medicine (AASM) sleep stages: `Wake`, `N1` (Light sleep stage 1), `N2` (Light sleep stage 2), `N3` (Slow-wave / deep sleep), and `REM` (Rapid Eye Movement).
  - No other categorical target column exists in the dataset.

### Class Balance & Distribution

| Sleep Stage (Class) | Epoch Count | Percentage | Typical Clinical Proportion (Normal Adult) |
|---|---|---|---|
| **N2** (Light Sleep Stage 2) | 345,067 | 51.35% | ~45% - 55% |
| **REM** (Rapid Eye Movement) | 147,989 | 22.02% | ~20% - 25% |
| **N3** (Deep / Slow Wave) | 119,051 | 17.72% | ~15% - 20% |
| **N1** (Light Sleep Stage 1) | 33,266 | 4.95% | ~2% - 5% |
| **Wake** | 26,627 | 3.96% | ~5% |
| **Total** | **672,000** | **100.00%** | **100.00%** |

- **Class Imbalance Assessment**:
  - The classes are naturally imbalanced, reflecting standard clinical sleep architecture.
  - `N2` constitutes over half the dataset (51.35%), while `N1` (4.95%) and `Wake` (3.96%) are minority classes.
  - Appropriate evaluation metrics (Macro F1, Balanced Accuracy, per-class PR-AUC) and loss balancing (class weights or focal loss) will be necessary during modeling.

---

## 4. Subject, Session, and Night Identifiers

### Identifier Columns
1. **`subject_id`**:
   - Unique values: 100 individuals (integers `1` through `100`).
   - Every subject has complete data across all 7 recorded nights.
2. **`night`**:
   - Unique values: 7 nights (integers `1` through `7`).
3. **Session / Recording Entity**:
   - Unique `(subject_id, night)` sessions: **700 distinct sessions**.
   - Balanced structure: Every session contains exactly **960 rows** (epochs).
   - $100 \text{ subjects} \times 7 \text{ nights} \times 960 \text{ epochs} = 672,000 \text{ rows}$.

> [!IMPORTANT]
> **Cross-Validation Split Strategy**:
> Train/validation/test splits must be grouped strictly by `subject_id` (e.g., using `GroupKFold` or `GroupShuffleSplit`). Splitting randomly across rows or sessions would leak individual physiological baselines (heart rate, skin temperature) into validation/test sets.

---

## 5. Timestamp & Temporal Structure

- **Column**: `timestamp`
- **Range**:
  - Start: `2025-01-02 22:30:00`
  - End: `2026-12-03 06:29:30`
- **Sampling Interval**:
  - Exactly **30 seconds** (`00:00:30`) between consecutive rows within every recording session.
  - Exactly matches standard clinical polysomnography epoch duration ($30\text{ s}$).
  - $960 \text{ epochs} \times 30\text{ s} = 28,800\text{ s} = 480\text{ minutes} = 8.0\text{ hours}$ per night per subject.
- **Ordering**:
  - **Within Sessions**: Strictly monotonic increasing with zero timestamp jitter, missing timestamps, or duplicate timestamps.
  - **Overall Dataset**: Strictly monotonic increasing across the entire CSV file. Subject 1 runs nights 1-7 sequentially, followed by Subject 2 nights 1-7, through Subject 100.
  - **Inter-Session Gap**: 16 hours during daytime between each night session (from 06:29:30 to 22:30:00).

---

## 6. Value Ranges, Statistical Summary & Outlier Analysis

### Summary Statistics for Numeric Features

| Feature | Min | Q1 (25%) | Median (50%) | Mean | Q3 (75%) | Max | Std Dev | IQR Outliers Count (%) |
|---|---|---|---|---|---|---|---|---|
| `heart_rate_bpm` | 40.000 | 52.400 | 57.600 | 58.118 | 63.300 | 101.400 | 8.068 | 5,257 (0.78%) |
| `spo2_pct` | 90.400 | 96.100 | 96.600 | 96.611 | 97.200 | 99.700 | 0.812 | 6,265 (0.93%) |
| `skin_temp_c` | 32.490 | 34.080 | 34.450 | 34.453 | 34.830 | 36.390 | 0.555 | 4,144 (0.62%) |
| `acc_x_g` | -2.865 | -0.010 | 0.020 | 0.020 | 0.050 | 3.133 | 0.108 | 31,829 (4.74%) |
| `acc_y_g` | -3.014 | 0.020 | 0.050 | 0.050 | 0.080 | 2.752 | 0.106 | 31,788 (4.73%) |
| `acc_z_g` | -2.044 | 0.950 | 0.980 | 0.980 | 1.010 | 3.859 | 0.107 | 31,850 (4.74%) |
| `gyro_x_dps` | -179.400 | -0.210 | 0.000 | 0.009 | 0.220 | 169.250 | 6.382 | 33,210 (4.94%) |
| `gyro_y_dps` | -174.590 | -0.230 | 0.000 | -0.005 | 0.220 | 156.700 | 6.462 | 33,255 (4.95%) |
| `gyro_z_dps` | -142.350 | -0.220 | 0.000 | -0.009 | 0.210 | 161.240 | 6.412 | 33,143 (4.93%) |
| `motion_intensity` | 0.000 | 0.009 | 0.015 | 0.035 | 0.026 | 2.938 | 0.106 | 28,670 (4.27%) |

### Outlier and Physical Plausibility Assessment

1. **Vital Signs**:
   - **`heart_rate_bpm`**: Ranges from 40.0 to 101.4 bpm. Resting sleep heart rates normally dip into the 40s during deep sleep (bradycardia of sleep) and spike during REM or brief awakenings. No physiologically impossible negative or extreme values (>200 or <30 bpm) exist.
   - **`spo2_pct`**: Ranges from 90.4% to 99.7%. This represents normal to mild nocturnal desaturation ranges. No corrupt sensor artifacts (such as 0% or >100%) are detected.
   - **`skin_temp_c`**: Ranges from 32.49°C to 36.39°C. Consistent with normal circadian nocturnal peripheral skin temperature rhythms.

2. **Inertial Sensors (IMU)**:
   - **Accelerometers (`acc_x_g`, `acc_y_g`, `acc_z_g`)**:
     - The baseline is dominated by gravity along the vertical axis (`acc_z_g` median is 0.98g, exactly matching $1g$ resting acceleration).
     - Peaks reach $\pm 3g$, which correspond to normal nocturnal body turnings, limb movements, or posture shifts.
   - **Gyroscopes (`gyro_x_dps`, `gyro_y_dps`, `gyro_z_dps`)**:
     - Narrow interquartile range centered at 0.0 dps (reflecting stillness during sleep).
     - Transient excursions up to $\pm 179^\circ/\text{s}$ occurring in ~4.9% of epochs during physical repositioning or waking episodes.
   - **`motion_intensity`**:
     - Bounded between 0.0 and 2.938.
     - Strongly right-skewed and zero-inflated; 75% of epochs have intensity $\le 0.026$, aligning with quiescent sleep.

3. **Data Quality Verdict**:
   - The dataset is exceptionally clean with zero missing values, zero corrupted/unparseable timestamps, zero duplicate rows, and no physically impossible values.
   - The detected statistical "outliers" via the $1.5 \times \text{IQR}$ rule are legitimate physiological events (e.g., nocturnal awakenings, limb movements, brief tachycardic arousal episodes) rather than corrupted sensor artifacts.
