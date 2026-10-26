"""
src/utils.py
Common evaluation, metrics computation, and artifact visualization utilities.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

CLASSES = ['Wake', 'N1', 'N2', 'N3', 'REM']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}


def compute_metrics(y_true, y_pred, y_prob=None, class_names=CLASSES):
    """
    Computes comprehensive multi-class classification metrics:
    - Accuracy
    - Macro precision, recall, F1
    - Per-class precision, recall, F1
    - Cohen's Kappa
    - One-vs-Rest ROC-AUC (if y_prob is provided)
    """
    acc = float(accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    
    # Macro metrics
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Per-class metrics
    prec_per_class, rec_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), average=None, zero_division=0
    )
    
    per_class_dict = {}
    for idx, cname in enumerate(class_names):
        per_class_dict[cname] = {
            'precision': float(prec_per_class[idx]),
            'recall': float(rec_per_class[idx]),
            'f1': float(f1_per_class[idx]),
            'support': int(support[idx])
        }
        
    # ROC-AUC (One-vs-Rest)
    roc_auc = None
    if y_prob is not None:
        try:
            # Check if all classes present in y_true
            present_classes = np.unique(y_true)
            if len(present_classes) == len(class_names):
                roc_auc = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
            else:
                # Handle subset in smoke testing
                roc_auc = float(roc_auc_score(y_true, y_prob[:, present_classes], multi_class='ovr', average='macro', labels=present_classes))
        except Exception as e:
            print(f"Warning: ROC-AUC calculation failed: {e}")
            roc_auc = None

    report_str = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))), normalize='true')

    return {
        'accuracy': acc,
        'cohen_kappa': kappa,
        'macro_precision': float(prec_macro),
        'macro_recall': float(rec_macro),
        'macro_f1': float(f1_macro),
        'roc_auc_ovr': roc_auc,
        'per_class': per_class_dict,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_norm.tolist(),
        'classification_report': report_str
    }


def plot_and_save_confusion_matrix(cm, cm_norm, class_names, out_path_png, out_path_csv, title="Confusion Matrix"):
    """
    Saves confusion matrix as a raw & normalized CSV, and renders a side-by-side PNG.
    """
    # Save CSVs
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_norm_df = pd.DataFrame(cm_norm, index=class_names, columns=class_names)
    
    cm_df.to_csv(out_path_csv.replace('.csv', '_raw.csv'))
    cm_norm_df.to_csv(out_path_csv.replace('.csv', '_normalized.csv'))
    cm_norm_df.to_csv(out_path_csv) # primary default

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0])
    axes[0].set_title(f"{title} (Counts)")
    axes[0].set_xlabel("Predicted Stage")
    axes[0].set_ylabel("True Stage")
    
    # Normalized
    sns.heatmap(cm_norm_df, annot=True, fmt='.2%', cmap='Blues', cbar=True, ax=axes[1])
    axes[1].set_title(f"{title} (Normalized)")
    axes[1].set_xlabel("Predicted Stage")
    axes[1].set_ylabel("True Stage")
    
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=200)
    plt.close()


def save_model_artifacts(model_dir, model_name, train_metrics, val_metrics, test_metrics, 
                         train_time, infer_time_per_epoch, model_size_bytes,
                         history=None, feature_importances=None):
    """
    Saves full artifact report for a single model:
    - metrics.json
    - confusion matrices (PNG + CSV)
    - classification report text
    - loss/accuracy curves (for NNs)
    - feature importance CSV & PNG (if available)
    - summary README/report
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Confusion Matrix plots and CSVs for test set
    cm = np.array(test_metrics['confusion_matrix'])
    cm_norm = np.array(test_metrics['confusion_matrix_normalized'])
    cm_png = os.path.join(model_dir, "confusion_matrix.png")
    cm_csv = os.path.join(model_dir, "confusion_matrix.csv")
    plot_and_save_confusion_matrix(cm, cm_norm, CLASSES, cm_png, cm_csv, title=f"{model_name} Test Confusion Matrix")
    
    # 2. Save Classification Report
    with open(os.path.join(model_dir, "classification_report.txt"), "w") as f:
        f.write(f"=== {model_name} Test Classification Report ===\n\n")
        f.write(test_metrics['classification_report'])
        
    # 3. Learning Curves (if neural net history provided)
    if history is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(history.get('loss', []), label='Train Loss')
        axes[0].plot(history.get('val_loss', []), label='Val Loss')
        axes[0].set_title('Loss Curve')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(history.get('accuracy', []), label='Train Accuracy')
        axes[1].plot(history.get('val_accuracy', []), label='Val Accuracy')
        axes[1].set_title('Accuracy Curve')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "loss_accuracy_curves.png"), dpi=200)
        plt.close()

    # 4. Feature importances (if available)
    if feature_importances is not None:
        feat_df = pd.DataFrame(feature_importances).sort_values(by='importance', ascending=False)
        feat_df.to_csv(os.path.join(model_dir, "feature_importance.csv"), index=False)
        
        plt.figure(figsize=(10, 8))
        top_feats = feat_df.head(20)
        sns.barplot(data=top_feats, x='importance', y='feature', hue='feature', palette='viridis', legend=False)
        plt.title(f"{model_name} Top 20 Feature Importances / Coefficients")
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "feature_importance.png"), dpi=200)
        plt.close()

    # 5. Combined metrics.json
    all_metrics = {
        'model_name': model_name,
        'training_time_seconds': float(train_time),
        'inference_time_ms_per_epoch': float(infer_time_per_epoch * 1000.0),
        'model_size_kb': float(model_size_bytes / 1024.0),
        'train': {
            'accuracy': train_metrics['accuracy'],
            'macro_f1': train_metrics['macro_f1'],
            'cohen_kappa': train_metrics['cohen_kappa'],
            'roc_auc_ovr': train_metrics['roc_auc_ovr']
        },
        'val': {
            'accuracy': val_metrics['accuracy'],
            'macro_f1': val_metrics['macro_f1'],
            'cohen_kappa': val_metrics['cohen_kappa'],
            'roc_auc_ovr': val_metrics['roc_auc_ovr']
        },
        'test': {
            'accuracy': test_metrics['accuracy'],
            'macro_f1': test_metrics['macro_f1'],
            'cohen_kappa': test_metrics['cohen_kappa'],
            'roc_auc_ovr': test_metrics['roc_auc_ovr'],
            'macro_precision': test_metrics['macro_precision'],
            'macro_recall': test_metrics['macro_recall'],
            'per_class': test_metrics['per_class']
        }
    }
    
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
        
    return all_metrics
