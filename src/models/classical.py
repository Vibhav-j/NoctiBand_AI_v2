"""
src/models/classical.py
Training, light GroupKFold tuning, evaluation, and artifact generation
for classical machine learning models:
1. Logistic Regression
2. Random Forest
3. SVM (LinearSVC with calibration)
4. XGBoost
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from src.utils import (
    CLASSES, CLASS_TO_IDX, compute_metrics, save_model_artifacts
)


def run_group_kfold_cv(model_fn, X_train, y_train, groups_train, param_grid, cv_splits=3):
    """
    Performs light hyperparameter search using GroupKFold strictly on Train subjects.
    Returns best_params, best_score, and fold_scores.
    """
    gkf = GroupKFold(n_splits=cv_splits)
    best_score = -1.0
    best_params = None
    best_fold_scores = []
    
    # Generate all param combinations
    import itertools
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"  GroupKFold tuning over {len(param_combinations)} combinations on {cv_splits} folds...")
    
    for params in param_combinations:
        fold_f1s = []
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups_train)):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_va, y_va = X_train[val_idx], y_train[val_idx]
            
            clf = model_fn(params)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_va)
            metrics = compute_metrics(y_va, preds)
            fold_f1s.append(metrics['macro_f1'])
            
        mean_f1 = np.mean(fold_f1s)
        std_f1 = float(pd.Series(fold_f1s).std(ddof=1)) if len(fold_f1s) > 1 else 0.0
        print(f"    Params {params} -> Mean Macro-F1: {mean_f1:.4f} (+/- {std_f1:.4f})")
        
        if mean_f1 > best_score:
            best_score = mean_f1
            best_params = params
            best_fold_scores = fold_f1s
            
    return best_params, float(best_score), [float(s) for s in best_fold_scores]


def measure_inference_speed(model, X_sample, n_iterations=10):
    """
    Measures inference latency per single sample (epoch) in seconds.
    """
    # Warmup
    _ = model.predict(X_sample[:100])
    
    t0 = time.time()
    for _ in range(n_iterations):
        _ = model.predict(X_sample)
    total_time = time.time() - t0
    
    total_samples = len(X_sample) * n_iterations
    return total_time / total_samples


def train_evaluate_model(name, model, X_train, y_train, X_val, y_val, X_test, y_test,
                         feature_names, reports_dir, models_dir, cv_info=None, feature_importances=None,
                         val_mask=None, test_mask=None):
    """
    Evaluates model on train, val, test, saves all artifacts and returns summary dict.
    If val_mask / test_mask provided, evaluation metrics are scored strictly on matching rows (epoch_in_session >= 9).
    """
    model_report_dir = os.path.join(reports_dir, name.lower().replace(" ", "_"))
    model_save_dir = os.path.join(models_dir, name.lower().replace(" ", "_"))
    os.makedirs(model_report_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Predict & evaluate Train
    print(f"Evaluating {name} on Train...")
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train) if hasattr(model, 'predict_proba') else None
    train_metrics = compute_metrics(y_train, y_train_pred, y_train_prob)
    
    # Predict & evaluate Val
    print(f"Evaluating {name} on Val (fair row-matched scoring)...")
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
    if val_mask is not None:
        y_val_eval = y_val[val_mask]
        y_val_pred_eval = y_val_pred[val_mask]
        y_val_prob_eval = y_val_prob[val_mask] if y_val_prob is not None else None
    else:
        y_val_eval, y_val_pred_eval, y_val_prob_eval = y_val, y_val_pred, y_val_prob
    val_metrics = compute_metrics(y_val_eval, y_val_pred_eval, y_val_prob_eval)
    
    # Predict & evaluate Test
    print(f"Evaluating {name} on Test (fair row-matched scoring)...")
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    if test_mask is not None:
        y_test_eval = y_test[test_mask]
        y_test_pred_eval = y_test_pred[test_mask]
        y_test_prob_eval = y_test_prob[test_mask] if y_test_prob is not None else None
    else:
        y_test_eval, y_test_pred_eval, y_test_prob_eval = y_test, y_test_pred, y_test_prob
    test_metrics = compute_metrics(y_test_eval, y_test_pred_eval, y_test_prob_eval)
    
    # Measure inference latency
    infer_latency = measure_inference_speed(model, X_test[:2000])
    
    # Save model binary
    model_file = os.path.join(model_save_dir, "model.joblib")
    joblib.dump(model, model_file)
    model_size_bytes = os.path.getsize(model_file)
    
    # Save all artifacts
    artifacts = save_model_artifacts(
        model_dir=model_report_dir,
        model_name=name,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        train_time=cv_info.get('train_time', 0.0) if cv_info else 0.0,
        infer_time_per_epoch=infer_latency,
        model_size_bytes=model_size_bytes,
        feature_importances=feature_importances
    )
    
    # Save best params & cv info if available
    if cv_info:
        with open(os.path.join(model_report_dir, "cv_results.json"), "w") as f:
            json.dump(cv_info, f, indent=2)
            
    return {
        'name': name,
        'train_acc': train_metrics['accuracy'],
        'train_f1': train_metrics['macro_f1'],
        'val_acc': val_metrics['accuracy'],
        'val_f1': val_metrics['macro_f1'],
        'test_acc': test_metrics['accuracy'],
        'test_f1': test_metrics['macro_f1'],
        'test_kappa': test_metrics['cohen_kappa'],
        'test_roc_auc': test_metrics['roc_auc_ovr'],
        'n1_recall': test_metrics['per_class']['N1']['recall'],
        'rem_recall': test_metrics['per_class']['REM']['recall'],
        'cv_mean_f1': cv_info.get('best_cv_score') if cv_info else None,
        'cv_std_f1': float(pd.Series(cv_info.get('fold_scores', [0])).std(ddof=1)) if cv_info and len(cv_info.get('fold_scores', [])) > 1 else 0.0,
        'train_time_s': cv_info.get('train_time', 0.0) if cv_info else 0.0,
        'infer_ms_per_epoch': infer_latency * 1000.0,
        'model_size_kb': model_size_bytes / 1024.0,
        'model_path': model_file
    }


def train_logistic_regression(X_train, y_train, groups_train, X_val, y_val, X_test, y_test,
                              feature_names, reports_dir="reports", models_dir="models",
                              smoke_test=False, val_mask=None, test_mask=None):
    """
    Model 1: Logistic Regression with GroupKFold tuning and class weighting.
    """
    print("\n" + "="*50)
    print("TRAINING MODEL 1: LOGISTIC REGRESSION")
    print("="*50)
    
    if smoke_test:
        param_grid = {'C': [1.0]}
        tune_sample_size = min(len(X_train), 5000)
    else:
        param_grid = {'C': [0.1, 1.0, 10.0]}
        tune_sample_size = min(len(X_train), 50000)
        
    # Subsample for fast CV tuning if large
    tune_idx = np.random.RandomState(42).choice(len(X_train), tune_sample_size, replace=False)
    
    def model_builder(p):
        return LogisticRegression(
            C=p['C'],
            class_weight='balanced',
            max_iter=500,
            solver='lbfgs',
            random_state=42
        )
        
    t0 = time.time()
    best_params, best_score, fold_scores = run_group_kfold_cv(
        model_builder, X_train[tune_idx], y_train[tune_idx], groups_train[tune_idx], param_grid, cv_splits=3
    )
    
    print(f"Best LR Params: {best_params} (CV Macro-F1: {best_score:.4f})")
    
    # Train final model on full train data
    print("Fitting final Logistic Regression on full train set...")
    t_train_start = time.time()
    final_model = model_builder(best_params)
    final_model.fit(X_train, y_train)
    total_train_time = time.time() - t_train_start
    
    # Feature importances: Mean absolute coefficient across classes
    coef_importances = np.mean(np.abs(final_model.coef_), axis=0)
    feat_imp = [{'feature': f, 'importance': float(imp)} for f, imp in zip(feature_names, coef_importances)]
    
    cv_info = {
        'best_params': best_params,
        'best_cv_score': best_score,
        'fold_scores': fold_scores,
        'train_time': total_train_time,
        'tuning_time': time.time() - t0
    }
    
    return train_evaluate_model(
        "Logistic Regression", final_model, X_train, y_train, X_val, y_val, X_test, y_test,
        feature_names, reports_dir, models_dir, cv_info, feat_imp,
        val_mask=val_mask, test_mask=test_mask
    )


def train_random_forest(X_train, y_train, groups_train, X_val, y_val, X_test, y_test,
                         feature_names, reports_dir="reports", models_dir="models",
                         smoke_test=False, val_mask=None, test_mask=None):
    """
    Model 2: Random Forest Classifier with GroupKFold tuning.
    """
    print("\n" + "="*50)
    print("TRAINING MODEL 2: RANDOM FOREST")
    print("="*50)
    
    if smoke_test:
        param_grid = {'n_estimators': [20], 'max_depth': [6], 'min_samples_leaf': [20]}
        tune_sample_size = min(len(X_train), 5000)
    else:
        param_grid = {'n_estimators': [100], 'max_depth': [8, 10], 'min_samples_leaf': [30]}
        tune_sample_size = min(len(X_train), 60000)
        
    tune_idx = np.random.RandomState(42).choice(len(X_train), tune_sample_size, replace=False)
    
    def model_builder(p):
        return RandomForestClassifier(
            n_estimators=p['n_estimators'],
            max_depth=p['max_depth'],
            min_samples_leaf=p.get('min_samples_leaf', 30),
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
        
    t0 = time.time()
    best_params, best_score, fold_scores = run_group_kfold_cv(
        model_builder, X_train[tune_idx], y_train[tune_idx], groups_train[tune_idx], param_grid, cv_splits=3
    )
    
    print(f"Best RF Params: {best_params} (CV Macro-F1: {best_score:.4f})")
    
    print("Fitting final Random Forest on full train set...")
    t_train_start = time.time()
    final_model = model_builder(best_params)
    final_model.fit(X_train, y_train)
    total_train_time = time.time() - t_train_start
    
    feat_imp = [{'feature': f, 'importance': float(imp)} for f, imp in zip(feature_names, final_model.feature_importances_)]
    
    cv_info = {
        'best_params': best_params,
        'best_cv_score': best_score,
        'fold_scores': fold_scores,
        'train_time': total_train_time,
        'tuning_time': time.time() - t0
    }
    
    return train_evaluate_model(
        "Random Forest", final_model, X_train, y_train, X_val, y_val, X_test, y_test,
        feature_names, reports_dir, models_dir, cv_info, feat_imp,
        val_mask=val_mask, test_mask=test_mask
    )


def train_svm(X_train, y_train, groups_train, X_val, y_val, X_test, y_test,
              feature_names, reports_dir="reports", models_dir="models",
              smoke_test=False, val_mask=None, test_mask=None):
    """
    Model 3: Support Vector Machine (LinearSVC with CalibratedClassifierCV).
    Per instruction: SVM is slow on large data (470k rows), so we use LinearSVC
    tuned on a representative stratified subsample (50k rows) and calibrated for probabilities.
    """
    print("\n" + "="*50)
    print("TRAINING MODEL 3: SVM (LINEAR SVC)")
    print("="*50)
    
    if smoke_test:
        param_grid = {'C': [1.0]}
        tune_sample_size = min(len(X_train), 5000)
    else:
        param_grid = {'C': [0.1, 1.0]}
        tune_sample_size = min(len(X_train), 50000)
        
    tune_idx = np.random.RandomState(42).choice(len(X_train), tune_sample_size, replace=False)
    
    def model_builder(p):
        return LinearSVC(
            C=p['C'],
            class_weight='balanced',
            dual=False,
            max_iter=1500,
            random_state=42
        )
        
    t0 = time.time()
    best_params, best_score, fold_scores = run_group_kfold_cv(
        model_builder, X_train[tune_idx], y_train[tune_idx], groups_train[tune_idx], param_grid, cv_splits=3
    )
    
    print(f"Best SVM Params: {best_params} (CV Macro-F1: {best_score:.4f})")
    
    print("Fitting Calibrated LinearSVC on full train set...")
    t_train_start = time.time()
    base_svc = model_builder(best_params)
    calibrated_svc = CalibratedClassifierCV(estimator=base_svc, cv=3)
    calibrated_svc.fit(X_train, y_train)
    total_train_time = time.time() - t_train_start
    
    # Feature importances from calibrated base estimators
    avg_coefs = np.zeros(len(feature_names))
    for cal_clf in calibrated_svc.calibrated_classifiers_:
        avg_coefs += np.mean(np.abs(cal_clf.estimator.coef_), axis=0)
    avg_coefs /= len(calibrated_svc.calibrated_classifiers_)
    
    feat_imp = [{'feature': f, 'importance': float(imp)} for f, imp in zip(feature_names, avg_coefs)]
    
    cv_info = {
        'best_params': best_params,
        'best_cv_score': best_score,
        'fold_scores': fold_scores,
        'train_time': total_train_time,
        'tuning_time': time.time() - t0,
        'method_note': 'Used LinearSVC(dual=False) with 3-fold CalibratedClassifierCV for fast linear boundary and probability outputs.'
    }
    
    return train_evaluate_model(
        "SVM", calibrated_svc, X_train, y_train, X_val, y_val, X_test, y_test,
        feature_names, reports_dir, models_dir, cv_info, feat_imp,
        val_mask=val_mask, test_mask=test_mask
    )


def train_xgboost(X_train, y_train, groups_train, X_val, y_val, X_test, y_test,
                  feature_names, class_weights_dict, reports_dir="reports", models_dir="models",
                  smoke_test=False, val_mask=None, test_mask=None):
    """
    Model 4: XGBoost with histogram tree method, early stopping on Val, and sample weighting.
    """
    print("\n" + "="*50)
    print("TRAINING MODEL 4: XGBOOST")
    print("="*50)
    
    # Compute per-sample weights from class_weights_dict
    sample_weights_train = np.array([class_weights_dict[CLASSES[y]] for y in y_train])
    sample_weights_val = np.array([class_weights_dict[CLASSES[y]] for y in y_val])
    
    if smoke_test:
        param_grid = {'max_depth': [4], 'n_estimators': [30], 'learning_rate': [0.1]}
        tune_sample_size = min(len(X_train), 5000)
    else:
        param_grid = {'max_depth': [6, 8], 'n_estimators': [150], 'learning_rate': [0.08]}
        tune_sample_size = min(len(X_train), 60000)
        
    tune_idx = np.random.RandomState(42).choice(len(X_train), tune_sample_size, replace=False)
    
    def model_builder(p):
        return XGBClassifier(
            n_estimators=p['n_estimators'],
            max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            tree_method='hist',
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1
        )
        
    t0 = time.time()
    # Light CV tuning
    best_params, best_score, fold_scores = run_group_kfold_cv(
        model_builder, X_train[tune_idx], y_train[tune_idx], groups_train[tune_idx], param_grid, cv_splits=3
    )
    
    print(f"Best XGBoost Params: {best_params} (CV Macro-F1: {best_score:.4f})")
    
    print("Fitting final XGBoost model on full train set with validation monitoring...")
    t_train_start = time.time()
    final_model = model_builder(best_params)
    final_model.fit(
        X_train, y_train,
        sample_weight=sample_weights_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    total_train_time = time.time() - t_train_start
    
    feat_imp = [{'feature': f, 'importance': float(imp)} for f, imp in zip(feature_names, final_model.feature_importances_)]
    
    cv_info = {
        'best_params': best_params,
        'best_cv_score': best_score,
        'fold_scores': fold_scores,
        'train_time': total_train_time,
        'tuning_time': time.time() - t0
    }
    
    return train_evaluate_model(
        "XGBoost", final_model, X_train, y_train, X_val, y_val, X_test, y_test,
        feature_names, reports_dir, models_dir, cv_info, feat_imp,
        val_mask=val_mask, test_mask=test_mask
    )
