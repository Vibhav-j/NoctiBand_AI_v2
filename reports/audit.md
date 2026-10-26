# NoctiBand AI Pipeline Leakage and Integrity Audit

- **Audit Date**: 2026-10-27
- **Branch**: `pipeline-A`

## Check A: Subject Partition Disjointness
- **Train Subjects**: 70 (1 to 100) | **Row Count**: 470,400 epochs
- **Validation Subjects**: 15 | **Row Count**: 100,800 epochs
- **Test Subjects**: 15 | **Row Count**: 100,800 epochs
- **Total Rows Verified**: 672,000 (Matches raw dataset 672,000 exactly)
- **Disjointness Assertion**: PASSED (Train, Val, Test subject sets are 100% pairwise disjoint; 0 cross-subject contamination).

## Check B: Feature Columns & Leakage Scrutiny
- **Classical Models Feature Count**: 156 features
- **CNN / LSTM Input Channels**: 12 base signal features over sequence length W=10
- **Explicitly Dropped Columns**: `['subject_id', 'night', 'timestamp', 'epoch_in_session', 'label', 'sleep_stage']`

### Classical Feature List (Full 156 columns):
```text
  [001] heart_rate_bpm
  [002] spo2_pct
  [003] skin_temp_c
  [004] acc_x_g
  [005] acc_y_g
  [006] acc_z_g
  [007] gyro_x_dps
  [008] gyro_y_dps
  [009] gyro_z_dps
  [010] motion_intensity
  [011] acc_mag
  [012] gyro_mag
  [013] heart_rate_bpm_roll3_mean
  [014] spo2_pct_roll3_mean
  [015] skin_temp_c_roll3_mean
  [016] acc_x_g_roll3_mean
  [017] acc_y_g_roll3_mean
  [018] acc_z_g_roll3_mean
  [019] gyro_x_dps_roll3_mean
  [020] gyro_y_dps_roll3_mean
  [021] gyro_z_dps_roll3_mean
  [022] motion_intensity_roll3_mean
  [023] acc_mag_roll3_mean
  [024] gyro_mag_roll3_mean
  [025] heart_rate_bpm_roll3_std
  [026] spo2_pct_roll3_std
  [027] skin_temp_c_roll3_std
  [028] acc_x_g_roll3_std
  [029] acc_y_g_roll3_std
  [030] acc_z_g_roll3_std
  [031] gyro_x_dps_roll3_std
  [032] gyro_y_dps_roll3_std
  [033] gyro_z_dps_roll3_std
  [034] motion_intensity_roll3_std
  [035] acc_mag_roll3_std
  [036] gyro_mag_roll3_std
  [037] heart_rate_bpm_roll3_min
  [038] spo2_pct_roll3_min
  [039] skin_temp_c_roll3_min
  [040] acc_x_g_roll3_min
  [041] acc_y_g_roll3_min
  [042] acc_z_g_roll3_min
  [043] gyro_x_dps_roll3_min
  [044] gyro_y_dps_roll3_min
  [045] gyro_z_dps_roll3_min
  [046] motion_intensity_roll3_min
  [047] acc_mag_roll3_min
  [048] gyro_mag_roll3_min
  [049] heart_rate_bpm_roll3_max
  [050] spo2_pct_roll3_max
  [051] skin_temp_c_roll3_max
  [052] acc_x_g_roll3_max
  [053] acc_y_g_roll3_max
  [054] acc_z_g_roll3_max
  [055] gyro_x_dps_roll3_max
  [056] gyro_y_dps_roll3_max
  [057] gyro_z_dps_roll3_max
  [058] motion_intensity_roll3_max
  [059] acc_mag_roll3_max
  [060] gyro_mag_roll3_max
  [061] heart_rate_bpm_roll6_mean
  [062] spo2_pct_roll6_mean
  [063] skin_temp_c_roll6_mean
  [064] acc_x_g_roll6_mean
  [065] acc_y_g_roll6_mean
  [066] acc_z_g_roll6_mean
  [067] gyro_x_dps_roll6_mean
  [068] gyro_y_dps_roll6_mean
  [069] gyro_z_dps_roll6_mean
  [070] motion_intensity_roll6_mean
  [071] acc_mag_roll6_mean
  [072] gyro_mag_roll6_mean
  [073] heart_rate_bpm_roll6_std
  [074] spo2_pct_roll6_std
  [075] skin_temp_c_roll6_std
  [076] acc_x_g_roll6_std
  [077] acc_y_g_roll6_std
  [078] acc_z_g_roll6_std
  [079] gyro_x_dps_roll6_std
  [080] gyro_y_dps_roll6_std
  [081] gyro_z_dps_roll6_std
  [082] motion_intensity_roll6_std
  [083] acc_mag_roll6_std
  [084] gyro_mag_roll6_std
  [085] heart_rate_bpm_roll6_min
  [086] spo2_pct_roll6_min
  [087] skin_temp_c_roll6_min
  [088] acc_x_g_roll6_min
  [089] acc_y_g_roll6_min
  [090] acc_z_g_roll6_min
  [091] gyro_x_dps_roll6_min
  [092] gyro_y_dps_roll6_min
  [093] gyro_z_dps_roll6_min
  [094] motion_intensity_roll6_min
  [095] acc_mag_roll6_min
  [096] gyro_mag_roll6_min
  [097] heart_rate_bpm_roll6_max
  [098] spo2_pct_roll6_max
  [099] skin_temp_c_roll6_max
  [100] acc_x_g_roll6_max
  [101] acc_y_g_roll6_max
  [102] acc_z_g_roll6_max
  [103] gyro_x_dps_roll6_max
  [104] gyro_y_dps_roll6_max
  [105] gyro_z_dps_roll6_max
  [106] motion_intensity_roll6_max
  [107] acc_mag_roll6_max
  [108] gyro_mag_roll6_max
  [109] heart_rate_bpm_roll10_mean
  [110] spo2_pct_roll10_mean
  [111] skin_temp_c_roll10_mean
  [112] acc_x_g_roll10_mean
  [113] acc_y_g_roll10_mean
  [114] acc_z_g_roll10_mean
  [115] gyro_x_dps_roll10_mean
  [116] gyro_y_dps_roll10_mean
  [117] gyro_z_dps_roll10_mean
  [118] motion_intensity_roll10_mean
  [119] acc_mag_roll10_mean
  [120] gyro_mag_roll10_mean
  [121] heart_rate_bpm_roll10_std
  [122] spo2_pct_roll10_std
  [123] skin_temp_c_roll10_std
  [124] acc_x_g_roll10_std
  [125] acc_y_g_roll10_std
  [126] acc_z_g_roll10_std
  [127] gyro_x_dps_roll10_std
  [128] gyro_y_dps_roll10_std
  [129] gyro_z_dps_roll10_std
  [130] motion_intensity_roll10_std
  [131] acc_mag_roll10_std
  [132] gyro_mag_roll10_std
  [133] heart_rate_bpm_roll10_min
  [134] spo2_pct_roll10_min
  [135] skin_temp_c_roll10_min
  [136] acc_x_g_roll10_min
  [137] acc_y_g_roll10_min
  [138] acc_z_g_roll10_min
  [139] gyro_x_dps_roll10_min
  [140] gyro_y_dps_roll10_min
  [141] gyro_z_dps_roll10_min
  [142] motion_intensity_roll10_min
  [143] acc_mag_roll10_min
  [144] gyro_mag_roll10_min
  [145] heart_rate_bpm_roll10_max
  [146] spo2_pct_roll10_max
  [147] skin_temp_c_roll10_max
  [148] acc_x_g_roll10_max
  [149] acc_y_g_roll10_max
  [150] acc_z_g_roll10_max
  [151] gyro_x_dps_roll10_max
  [152] gyro_y_dps_roll10_max
  [153] gyro_z_dps_roll10_max
  [154] motion_intensity_roll10_max
  [155] acc_mag_roll10_max
  [156] gyro_mag_roll10_max
```

### Neural Sequence Base Feature Channels:
```text
  [01] heart_rate_bpm
  [02] spo2_pct
  [03] skin_temp_c
  [04] acc_x_g
  [05] acc_y_g
  [06] acc_z_g
  [07] gyro_x_dps
  [08] gyro_y_dps
  [09] gyro_z_dps
  [10] motion_intensity
  [11] acc_mag
  [12] gyro_mag
```

- **Feature Leakage Scrutiny**: PASSED. Zero columns encode subject IDs, night indices, timestamps, row positions, or sleep stages.

## Check C: Scaler Verification
- **Scaler Mean Vector Length**: 156
- **Scaler Scale Vector Length**: 156
- **Scaler Samples Seen Count (`n_samples_seen_`)**: 470400.0
- **Train Sample Match**: Exact match. The scaler was fit on exactly 470,400 training samples and 0 validation/test samples.
- **Scaler Fit Audit**: PASSED.

## Check D: Window Boundary Isolation Verification
- **Rolling Features Boundary Test**: PASSED. Session start epochs have unscaled std of exactly 0.0 (max abs diff < 1e-15), confirming strict within-session reset.
- **Sequence Windows Total (Train)**: 465,990 windows (Exact theoretical match: 70 subjects * 7 nights * 951 valid windows).
- **Sequence Boundary Test**: PASSED. Sequences are strictly constructed within each recording session and never cross subject or night borders.

## Check E: Class Distribution Across Splits
| Class   |   Train Count |   Train % |   Val Count |   Val % |   Test Count |   Test % |
|:--------|--------------:|----------:|------------:|--------:|-------------:|---------:|
| N2      |        241475 |     51.33 |       51983 |   51.57 |        51609 |    51.2  |
| REM     |        103563 |     22.02 |       22183 |   22.01 |        22243 |    22.07 |
| N3      |         83475 |     17.75 |       17664 |   17.52 |        17912 |    17.77 |
| N1      |         23257 |      4.94 |        4997 |    4.96 |         5012 |     4.97 |
| Wake    |         18630 |      3.96 |        3973 |    3.94 |         4024 |     3.99 |

- **Class Distribution Balance**: Confirmed identical proportions across Train, Val, and Test.

### Overall Leakage & Integrity Verdict: 100% CLEAN & VERIFIED.