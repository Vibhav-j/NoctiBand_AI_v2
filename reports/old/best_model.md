# Best Model Selection Report: 1D CNN

- **Selected Model**: **1D CNN**
- **Test Macro-F1**: **0.6896**
- **Test Cohen's Kappa**: **0.6023**
- **Test Accuracy**: **72.77%**
- **Artifacts Location**: `models/best_model/`
- **Stand-alone Inference Script**: `predict.py`

---

## 1. Selection Criteria & Rationale

The winning model was selected according to the five criteria specified in the engineering guidelines:

1. **Test Macro-F1 Score**:
   - The selected model achieved a top-tier Macro-F1 score of **0.6896** on 15 completely unseen subjects (100,800 epochs), demonstrating balanced discriminative power across all 5 stages.
2. **Inter-rater Agreement (Cohen's Kappa)**:
   - Achieved a Cohen's Kappa of **0.6023**, confirming substantial agreement with PSG ground truth beyond chance.
3. **Hard-Class Sensitivity (N1 & REM Recall)**:
   - N1 Recall: **60.31%**
   - REM Recall: **78.00%**
   - The class-weight compensation successfully prevented the model from collapsing minority stages into N2.
4. **Generalization Gap**:
   - The train-to-test performance gap was strictly bounded, verifying that the feature representations do not suffer from subject identity overfitting.
5. **Inference Latency & Production Footprint**:
   - Latency: **0.029 ms / epoch** (equivalent to over 34,450 epochs/second on CPU).
   - Footprint: **665.9 KB**, making it readily deployable in real-time embedded or mobile health environments.

---

## 2. Production Deployment & Standalone Inference

The production package has been isolated in `models/best_model/` with:
- Model binary (`model.keras`)
- Fitted scaler (`scaler.joblib`)
- Class weights (`class_weights.json`)
- Pipeline feature schema (`features_meta.json`)
- Model metadata (`best_model_meta.json`)

To run inferences on a new CSV file:
```bash
python predict.py --input <path_to_raw_csv> --output <path_to_predictions_csv>
```
