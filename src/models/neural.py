"""
src/models/neural.py
Deep learning sequence architectures (1D CNN and LSTM) implemented with Keras.
Features:
- Sliding window sequences strictly within subject/session
- Early stopping on validation loss with best weights restore
- Learning rate scheduling via ReduceLROnPlateau
- Class weight compensation
- Light GroupKFold tuning on Train subjects
- Artifact generation (loss/acc curves, confusion matrix, metrics.json)
"""

import os
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import GroupKFold

from src.utils import (
    CLASSES, CLASS_TO_IDX, compute_metrics, save_model_artifacts
)


def build_1d_cnn(input_shape, num_classes=5, filters=32, lr=0.001):
    """
    Constructs a 1D Convolutional Neural Network for multi-epoch temporal context.
    """
    inputs = keras.Input(shape=input_shape)
    
    # Block 1
    x = layers.Conv1D(filters=filters, kernel_size=3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Block 2
    x = layers.Conv1D(filters=filters * 2, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)
    
    # Block 3: Global Pooling
    x = layers.Conv1D(filters=filters * 2, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense head
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="1D_CNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_lstm(input_shape, num_classes=5, hidden_units=32, lr=0.001):
    """
    Constructs a Bidirectional / stacked LSTM for temporal sleep state transitions.
    """
    inputs = keras.Input(shape=input_shape)
    
    # Bidirectional LSTM layer
    x = layers.Bidirectional(layers.LSTM(hidden_units, return_sequences=True, dropout=0.2))(inputs)
    x = layers.LSTM(hidden_units, dropout=0.2)(x)
    
    # Dense head
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="LSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_cnn_lstm(input_shape, num_classes=5, lr=0.001):
    """
    Constructs a CNN-LSTM hybrid architecture:
    - Conv1D 32 filters, kernel 3, padding same, BatchNorm, ReLU
    - Conv1D 64 filters, kernel 3, padding same, BatchNorm, ReLU
    - Dropout 0.2
    - LSTM 64 units (return_sequences=False, recurrent_dropout 0.2)
    - Dense 64, ReLU, Dropout 0.3
    - Dense 5, softmax
    No pooling before LSTM to preserve full 10-step sequence temporal resolution.
    """
    inputs = keras.Input(shape=input_shape)
    
    # Conv Block 1
    x = layers.Conv1D(filters=32, kernel_size=3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Conv Block 2
    x = layers.Conv1D(filters=64, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Recurrent sequence layer (no pooling)
    x = layers.LSTM(64, return_sequences=False, recurrent_dropout=0.2)(x)
    
    # Dense classification head
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="CNN_LSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def measure_nn_inference_speed(model, X_sample, n_iterations=5):
    """
    Measures inference latency per epoch sample in seconds.
    """
    # Warmup
    _ = model.predict(X_sample[:100], verbose=0)
    
    t0 = time.time()
    for _ in range(n_iterations):
        _ = model.predict(X_sample, batch_size=512, verbose=0)
    total_time = time.time() - t0
    
    total_samples = len(X_sample) * n_iterations
    return total_time / total_samples


def train_evaluate_nn_model(name, model_fn, best_params, X_train, y_train, groups_train,
                            X_val, y_val, X_test, y_test, class_weights_dict,
                            reports_dir, models_dir, cv_info, max_epochs=25, batch_size=512):
    """
    Trains neural network with early stopping, evaluates on all splits, and saves artifacts.
    """
    model_report_dir = os.path.join(reports_dir, name.lower().replace(" ", "_").replace("-", "_"))
    model_save_dir = os.path.join(models_dir, name.lower().replace(" ", "_").replace("-", "_"))
    os.makedirs(model_report_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Map class weights to integer dict
    class_weights_idx = {i: class_weights_dict[CLASSES[i]] for i in range(len(CLASSES))}
    
    # Callbacks
    model_keras_file = os.path.join(model_save_dir, "model.keras")
    cb_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        )
    ]
    
    print(f"Building {name} with params: {best_params}...")
    model = model_fn(best_params)
    
    print(f"Fitting {name} on Train ({len(X_train)} sequences) with early stopping on Val...")
    t_train_start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        class_weight=class_weights_idx,
        callbacks=cb_list,
        verbose=1
    )
    total_train_time = time.time() - t_train_start
    print(f"Training completed in {total_train_time:.1f}s.")
    
    # Save model
    model.save(model_keras_file)
    model_size_bytes = os.path.getsize(model_keras_file)
    
    # Predictions
    print(f"Evaluating {name} across splits...")
    y_train_prob = model.predict(X_train, batch_size=batch_size, verbose=0)
    y_train_pred = np.argmax(y_train_prob, axis=1)
    train_metrics = compute_metrics(y_train, y_train_pred, y_train_prob)
    
    y_val_prob = model.predict(X_val, batch_size=batch_size, verbose=0)
    y_val_pred = np.argmax(y_val_prob, axis=1)
    val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob)
    
    y_test_prob = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_test_pred = np.argmax(y_test_prob, axis=1)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob)
    
    # Latency
    infer_latency = measure_nn_inference_speed(model, X_test[:2000])
    
    # Save artifacts
    artifacts = save_model_artifacts(
        model_dir=model_report_dir,
        model_name=name,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        train_time=total_train_time,
        infer_time_per_epoch=infer_latency,
        model_size_bytes=model_size_bytes,
        history=history.history
    )
    
    cv_info['train_time'] = total_train_time
    with open(os.path.join(model_report_dir, "cv_results.json"), "w") as f:
        json.dump(cv_info, f, indent=2)
        
    return {
        'name': name,
        'train_acc': train_metrics['accuracy'],
        'train_f1': train_metrics['macro_f1'],
        'val_acc': val_metrics['accuracy'],
        'val_f1': val_metrics['macro_f1'],
        'val_kappa': val_metrics['cohen_kappa'],
        'test_acc': test_metrics['accuracy'],
        'test_f1': test_metrics['macro_f1'],
        'test_kappa': test_metrics['cohen_kappa'],
        'test_roc_auc': test_metrics['roc_auc_ovr'],
        'n1_recall': test_metrics['per_class']['N1']['recall'],
        'rem_recall': test_metrics['per_class']['REM']['recall'],
        'cv_mean_f1': cv_info.get('best_cv_score'),
        'cv_std_f1': float(np.std(cv_info['fold_scores'])) if cv_info.get('fold_scores') else 0.0,
        'train_time_s': total_train_time,
        'infer_ms_per_epoch': infer_latency * 1000.0,
        'model_size_kb': model_size_bytes / 1024.0,
        'model_path': model_keras_file
    }


def train_1d_cnn(X_train, y_train, groups_train, X_val, y_val, X_test, y_test,
                 class_weights_dict, reports_dir="reports", models_dir="models",
                 smoke_test=False):
    """
    Model 5: 1D CNN for multi-epoch sleep feature sequences.
    """
    print("\n" + "="*50)
    print("TRAINING MODEL 5: 1D CNN")
    print("="*50)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    if smoke_test:
        param_grid = [{'filters': 16, 'lr': 0.005}]
        epochs = 2
        batch_size = 128
        tune_sample_size = min(len(X_train), 2000)
    else:
        param_grid = [{'filters': 32, 'lr': 0.001}, {'filters': 48, 'lr': 0.001}]
        epochs = 20
        batch_size = 512
        tune_sample_size = min(len(X_train), 40000)
        
    # Light CV tuning strictly on Train subjects
    tune_idx = np.random.RandomState(42).choice(len(X_train), tune_sample_size, replace=False)
    X_tune, y_tune, groups_tune = X_train[tune_idx], y_train[tune_idx], groups_train[tune_idx]
    
    gkf = GroupKFold(n_splits=2)
    best_score = -1.0
    best_params = param_grid[0]
    best_fold_scores = []
    
    print("  GroupKFold tuning for 1D CNN...")
    for p in param_grid:
        fold_scores = []
        for tr_idx, val_idx in gkf.split(X_tune, y_tune, groups=groups_tune):
            m = build_1d_cnn(input_shape, filters=p['filters'], lr=p['lr'])
            m.fit(X_tune[tr_idx], y_tune[tr_idx], epochs=2, batch_size=batch_size, verbose=0)
            preds = np.argmax(m.predict(X_tune[val_idx], batch_size=batch_size, verbose=0), axis=1)
            metrics = compute_metrics(y_tune[val_idx], preds)
            fold_scores.append(metrics['macro_f1'])
        mean_score = np.mean(fold_scores)
        print(f"    Params {p} -> CV Macro-F1: {mean_score:.4f}")
        if mean_score > best_score:
            best_score = mean_score
            best_params = p
            best_fold_scores = fold_scores
            
    cv_info = {
        'best_params': best_params,
        'best_cv_score': float(best_score),
        'fold_scores': [float(s) for s in best_fold_scores]
    }
    
    def final_builder(p):
        return build_1d_cnn(input_shape, filters=p['filters'], lr=p['lr'])
        
    return train_evaluate_nn_model(
        "1D CNN", final_builder, best_params, X_train, y_train, groups_train,
        X_val, y_val, X_test, y_test, class_weights_dict,
        reports_dir, models_dir, cv_info, max_epochs=epochs, batch_size=batch_size
    )


def train_lstm(X_train, y_train, groups_train, X_val, y_val, X_test, y_test,
               class_weights_dict, reports_dir="reports", models_dir="models",
               smoke_test=False):
    """
    Model 6: LSTM for recurrent sequence modeling across sleep epochs.
    """
    print("\n" + "="*50)
    print("TRAINING MODEL 6: LSTM")
    print("="*50)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    if smoke_test:
        param_grid = [{'hidden_units': 16, 'lr': 0.005}]
        epochs = 2
        batch_size = 128
        tune_sample_size = min(len(X_train), 2000)
    else:
        param_grid = [{'hidden_units': 32, 'lr': 0.001}, {'hidden_units': 48, 'lr': 0.001}]
        epochs = 18
        batch_size = 512
        tune_sample_size = min(len(X_train), 30000)
        
    tune_idx = np.random.RandomState(42).choice(len(X_train), tune_sample_size, replace=False)
    X_tune, y_tune, groups_tune = X_train[tune_idx], y_train[tune_idx], groups_train[tune_idx]
    
    gkf = GroupKFold(n_splits=2)
    best_score = -1.0
    best_params = param_grid[0]
    best_fold_scores = []
    
    print("  GroupKFold tuning for LSTM...")
    for p in param_grid:
        fold_scores = []
        for tr_idx, val_idx in gkf.split(X_tune, y_tune, groups=groups_tune):
            m = build_lstm(input_shape, hidden_units=p['hidden_units'], lr=p['lr'])
            m.fit(X_tune[tr_idx], y_tune[tr_idx], epochs=2, batch_size=batch_size, verbose=0)
            preds = np.argmax(m.predict(X_tune[val_idx], batch_size=batch_size, verbose=0), axis=1)
            metrics = compute_metrics(y_tune[val_idx], preds)
            fold_scores.append(metrics['macro_f1'])
        mean_score = np.mean(fold_scores)
        print(f"    Params {p} -> CV Macro-F1: {mean_score:.4f}")
        if mean_score > best_score:
            best_score = mean_score
            best_params = p
            best_fold_scores = fold_scores
            
    cv_info = {
        'best_params': best_params,
        'best_cv_score': float(best_score),
        'fold_scores': [float(s) for s in best_fold_scores]
    }
    
    def final_builder(p):
        return build_lstm(input_shape, hidden_units=p['hidden_units'], lr=p['lr'])
        
    return train_evaluate_nn_model(
        "LSTM", final_builder, best_params, X_train, y_train, groups_train,
        X_val, y_val, X_test, y_test, class_weights_dict,
        reports_dir, models_dir, cv_info, max_epochs=epochs, batch_size=batch_size
    )


def train_cnn_lstm(X_train, y_train, groups_train, X_val, y_val, X_test, y_test,
                   class_weights_dict, reports_dir="reports", models_dir="models",
                   smoke_test=False, seed=42):
    """
    Model 7: CNN-LSTM hybrid for multi-epoch sleep feature sequences.
    Preserves full 10-step sequence with no pooling before LSTM.
    """
    print("\n" + "="*50)
    print("TRAINING MODEL 7: CNN-LSTM")
    print("="*50)
    
    tf.keras.utils.set_random_seed(seed)
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    epochs = 2 if smoke_test else 20
    batch_size = 128 if smoke_test else 512
    
    best_params = {'filters': [32, 64], 'lstm_units': 64, 'lr': 0.001, 'seed': seed}
    cv_info = {
        'best_params': best_params,
        'best_cv_score': None,
        'fold_scores': []
    }
    
    def final_builder(p):
        return build_cnn_lstm(input_shape, lr=p.get('lr', 0.001))
        
    return train_evaluate_nn_model(
        "CNN-LSTM", final_builder, best_params, X_train, y_train, groups_train,
        X_val, y_val, X_test, y_test, class_weights_dict,
        reports_dir, models_dir, cv_info, max_epochs=epochs, batch_size=batch_size
    )

