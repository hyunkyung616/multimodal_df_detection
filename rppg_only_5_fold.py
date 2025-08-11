# This code is an original work authored by Hyunkyung Lee.
# It was developed as part of the MSc CyberSecurity individual project at King's College London.
# This code is written for the Google Colab environment.

# This code applies 5-fold cross validation to the existing model.
# Except for the full data loading, 5-fold parts, and error handling part,
# all other parts are identical to the existing model code.

# Install according to your environment using the command below.

# !pip install tensorflow keras-tuner scikit-learn scipy

import os, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Bidirectional, Dense, Dropout, BatchNormalization, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import kerastuner as kt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

SEED = 42
tf.keras.utils.set_random_seed(SEED)

# # Set after loading all data
raw_shape = None
feat_shape = None

# Data load
def load_data(split_dir):
    X_raw = [] 
    X_feat = []
    y = []

    # Original data (label 0)
    orig_raw_dir = os.path.join(split_dir, 'original', 'processed_signals')
    orig_feat_dir = os.path.join(split_dir, 'original', 'derived_features')

    if not os.path.exists(orig_raw_dir):
        print(f"Error: Could not find original Raw PPG directory: {orig_raw_dir}")
        return np.array([]), np.array([]), np.array([])
    if not os.path.exists(orig_feat_dir):
        print(f"Error: Could not find source features directory: {orig_feat_dir}")
        return np.array([]), np.array([]), np.array([])

    raw_files = sorted([f for f in os.listdir(orig_raw_dir) if f.endswith('.csv')])
    if not raw_files:
        print(f"Warning: There is no CSV file in '{orig_raw_dir}'.")

    for file in raw_files:
        raw_path = os.path.join(orig_raw_dir, file)
        feat_path = os.path.join(orig_feat_dir, file)

        if not os.path.exists(feat_path):
            print(f"Warning: The feature file for '{file}' is missing at '{feat_path}'. Skipping sample.")
            continue

        try:
            raw_df = pd.read_csv(raw_path)
            X_raw.append(raw_df.filter(like='_filtered').values)
        except Exception as e:
            print(f"Error: Failed to read raw PPG file '{raw_path}': {e}. Skipping sample.")
            continue 

        try:
            feat_df = pd.read_csv(feat_path)
            X_feat.append(feat_df.drop(columns='frame').values)
        except Exception as e:
            print(f"Error: Failed to read feature file '{feat_path}': {e}. Skipping sample.")
            if X_raw: X_raw.pop()
            continue

        y.append(0)

    # Deepfake data (label 1)
    df_raw_dir = os.path.join(split_dir, 'deepfakes', 'processed_signals')
    df_feat_dir = os.path.join(split_dir, 'deepfakes', 'derived_features')

    if not os.path.exists(df_raw_dir):
        print(f"Error: Could not find Deepfake Raw PPG directory: {df_raw_dir}")
        return np.array([]), np.array([]), np.array([])
    if not os.path.exists(df_feat_dir):
        print(f"Error: Could not find Deepfake features directory: {df_feat_dir}")
        return np.array([]), np.array([]), np.array([])

    df_raw_files = sorted([f for f in os.listdir(df_raw_dir) if f.endswith('.csv')])
    if not df_raw_files:
        print(f"Warning: No CSV file in '{df_raw_dir}'.")

    for file in df_raw_files:
        raw_path = os.path.join(df_raw_dir, file)
        feat_path = os.path.join(df_feat_dir, file)

        if not os.path.exists(feat_path):
            print(f"Warning: The feature file for '{feat_path}' is missing. Skipping sample.")
            continue

        try:
            raw_df = pd.read_csv(raw_path)
            X_raw.append(raw_df.filter(like='_filtered').values)
        except Exception as e:
            print(f"Error: Failed to read raw PPG file '{raw_path}': {e}. Skipping sample.")
            continue

        try:
            feat_df = pd.read_csv(feat_path)
            X_feat.append(feat_df.drop(columns='frame').values)
        except Exception as e:
            print(f"Error: Failed to read feature file '{feat_path}': {e}. Skipping sample.")
            if X_raw: X_raw.pop()
            continue

        y.append(1)

    max_raw_len = max(arr.shape[0] for arr in X_raw) if X_raw else 1
    max_feat_len = max(arr.shape[0] for arr in X_feat) if X_feat else 1

    if not X_raw or not X_feat: 
        print("It returns an empty array because no data was loaded successfully.")
        return np.array([]), np.array([]), np.array([])

    X_raw_padded = [np.pad(arr, ((0, max_raw_len - arr.shape[0]), (0,0) if arr.ndim > 1 else (0,0)), 'constant') for arr in X_raw]
    X_feat_padded = [np.pad(arr, ((0, max_feat_len - arr.shape[0]), (0,0) if arr.ndim > 1 else (0,0)), 'constant') for arr in X_feat]

    return np.array(X_raw_padded), np.array(X_feat_padded), np.array(y)


# rPPG-only model build (same as the rPPG-only model)
def build_ppg_feat_model(hp):

    raw_input = Input(shape=raw_shape, name='raw_ppg_input')

    conv_filters = hp.Int('conv_filters', min_value=32, max_value=128, step=32)
    x_ppg = Conv1D(conv_filters, 3, activation='relu', padding='same')(raw_input)
    x_ppg = BatchNormalization()(x_ppg)
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    x_ppg = Dropout(dropout_rate)(x_ppg)

    lstm_units1 = hp.Int('lstm_units1', min_value=64, max_value=256, step=64)
    x_ppg = Bidirectional(LSTM(lstm_units1, return_sequences=True))(x_ppg)
    x_ppg = Dropout(dropout_rate)(x_ppg)

    lstm_units2 = hp.Int('lstm_units2', min_value=32, max_value=128, step=32)
    x_ppg = Bidirectional(LSTM(lstm_units2))(x_ppg)
    ppg_output = Dense(64, activation='relu')(x_ppg)

    feat_input = Input(shape=feat_shape, name='csv_feat_input')

    x_feat_dense = Dense(128, activation='relu')(feat_input)
    x_feat_dense = BatchNormalization()(x_feat_dense)
    x_feat_dense = Dropout(dropout_rate)(x_feat_dense)
    feat_output = GlobalAveragePooling1D()(x_feat_dense)
    feat_output = Dense(64, activation='relu')(feat_output)

    merged = Concatenate()([ppg_output, feat_output])

    fusion_units = hp.Int('fusion_units', min_value=64, max_value=256, step=64)
    final_dropout = hp.Float('final_dropout', 0.2, 0.5, step=0.1)

    z = Dense(fusion_units, activation='relu')(merged)
    z = BatchNormalization()(z)
    z = Dropout(final_dropout)(z)
    output = Dense(1, activation='sigmoid')(z)

    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model = Model(inputs=[raw_input, feat_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model

# Main execution
all_data_dir = '/content/data_final'

# Load all data (load all data at once for K-fold)
print("Loading full data for K-fold cross validation...")
X_raw_all, X_feat_all, y_all = load_data(all_data_dir)

if X_raw_all.size == 0 or X_feat_all.size == 0:
    print("\nError: No data loaded.")
    exit()

global raw_shape, feat_shape
raw_shape = X_raw_all.shape[1:]
feat_shape = X_feat_all.shape[1:] 

print(f"\nLoaded all data shapes:")
print(f"  Raw PPG: {X_raw_all.shape}")
print(f"  CSV Features: {X_feat_all.shape}")
print(f"  Label: {y_all.shape}")

# K-fold
N_SPLITS = 5         
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
FIXED_EPOCHS = 100  

fold_results = {
    'loss': [],
    'accuracy': [],
    'auc': [],
    'precision': [],
    'recall': [],
    'f1_score': []
}
best_auc_overall = -1 
best_model_path_overall = ""

# Start the K-fold cross-validation loop
for fold, (train_val_idx, test_idx) in enumerate(skf.split(X_raw_all, y_all)):
    print(f"\n\n{'='*50}")
    print(f"--- Starting Fold #{fold+1}/{N_SPLITS} ---")
    print(f"{'='*50}")

    X_raw_train_val, X_raw_test = X_raw_all[train_val_idx], X_raw_all[test_idx]
    X_feat_train_val, X_feat_test = X_feat_all[train_val_idx], X_feat_all[test_idx]
    y_train_val, y_test = y_all[train_val_idx], y_all[test_idx]

    # Split the training and validation sets again from the training+validation set
    # Separate the validation set for hyperparameter tuning
    inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx_inner, val_idx_inner = next(inner_skf.split(X_raw_train_val, y_train_val))

    X_raw_train, X_raw_val = X_raw_train_val[train_idx_inner], X_raw_train_val[val_idx_inner]
    X_feat_train, X_feat_val = X_feat_train_val[train_idx_inner], X_feat_train_val[val_idx_inner]
    y_train, y_val = y_train_val[train_idx_inner], y_train_val[val_idx_inner]

    print(f"\nFold {fold+1} Data Shapes:")
    print(f"  Train Raw PPG: {X_raw_train.shape}, Feat: {X_feat_train.shape}, Label: {y_train.shape}")
    print(f"  Val Raw PPG: {X_raw_val.shape}, Feat: {X_feat_val.shape}, Label: {y_val.shape}")
    print(f"  Test Raw PPG: {X_raw_test.shape}, Feat: {X_feat_test.shape}, Label: {y_test.shape}")

    # Hyperparameter tuning
    tuner = kt.RandomSearch(
        build_ppg_feat_model, 
        objective=kt.Objective("val_auc", direction="max"),
        max_trials=20,
        executions_per_trial=1,
        directory=f'tuner_results/ppg_feat_fold_{fold}', 
        project_name='ppg_and_derived_features_kfold_tuning'
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1
    )

    print(f"\nFold {fold+1}: Start hyperparameter tuning...")

    tuner.search(
        [X_raw_train, X_feat_train], y_train,
        validation_data=([X_raw_val, X_feat_val], y_val),
        epochs=50, batch_size=32, callbacks=[reduce_lr], verbose=1
    )

    # Print best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"\nFold #{fold+1} Best hyperparameters:")
    print(f"  conv_filters: {best_hps.get('conv_filters')}")
    print(f"  lstm_units1: {best_hps.get('lstm_units1')}")
    print(f"  lstm_units2: {best_hps.get('lstm_units2')}")
    print(f"  dropout_rate: {best_hps.get('dropout_rate')}")
    print(f"  fusion_units: {best_hps.get('fusion_units')}")
    print(f"  final_dropout: {best_hps.get('final_dropout')}")
    print(f"  learning_rate: {best_hps.get('learning_rate')}")

    # Build the optimal model (build anew for each fold)
    model = tuner.hypermodel.build(best_hps)

    # Checkpoint
    fold_model_filepath = f'best_ppg_feat_kfold_model_fold_{fold+1}.h5'
    checkpoint = ModelCheckpoint(
        filepath=fold_model_filepath, monitor='val_auc', save_best_only=True, mode='max', verbose=1
    )

    print(f"\nFold {fold+1}: Start learning the best model...")

    history = model.fit(
        [X_raw_train, X_feat_train], y_train,
        validation_data=([X_raw_val, X_feat_val], y_val),
        epochs=FIXED_EPOCHS, batch_size=32, callbacks=[checkpoint, reduce_lr], verbose=1
    )

    # # After training is complete, load the best model from that fold.
    best_fold_model = load_model(fold_model_filepath)

    # Final evaluation with test data
    print(f"\nFold {fold+1}: Evaluating the test set...")

    test_loss, test_accuracy, test_auc = best_fold_model.evaluate([X_raw_test, X_feat_test], y_test, verbose=0)

    y_pred_prob = best_fold_model.predict([X_raw_test, X_feat_test]).flatten()
    y_pred_label = (y_pred_prob > 0.5).astype(int)

    test_precision = precision_score(y_test, y_pred_label)
    test_recall = recall_score(y_test, y_pred_label)
    test_f1_score = f1_score(y_test, y_pred_label)

    print(f"Fold #{fold+1} Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, AUC: {test_auc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1 Score: {test_f1_score:.4f}")

    fold_results['loss'].append(test_loss)
    fold_results['accuracy'].append(test_accuracy)
    fold_results['auc'].append(test_auc)
    fold_results['precision'].append(test_precision)
    fold_results['recall'].append(test_recall)
    fold_results['f1_score'].append(test_f1_score)

    if test_auc > best_auc_overall:
        best_auc_overall = test_auc
        best_model_path_overall = fold_model_filepath

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title(f'Fold #{fold+1} AUC (PPG + Derived Features)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Fold #{fold+1} Loss (PPG + Derived Features)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'training_history_ppg_feat_fold_{fold+1}.png')
    plt.show()

    tf.keras.backend.clear_session()

print("\n\n" + "="*50)
print(f"{N_SPLITS}-Fold Cross Validation Final Results (PPG + Derived Features)")
print("="*50)

for metric, values in fold_results.items():
    print(f"Average {metric.capitalize()}: {np.mean(values):.4f} (Std: {np.std(values):.4f})")
    print(f"Individual {metric.capitalize()}: {[f'{v:.4f}' for v in values]}")

plt.figure(figsize=(8, 6))
plt.bar(range(1, N_SPLITS + 1), fold_results['auc'])
plt.axhline(np.mean(fold_results['auc']), color='r', linestyle='--', label=f'Mean AUC: {np.mean(fold_results["auc"]):.4f}')
plt.title(f'{N_SPLITS}-Fold cross validation test AUC results')
plt.xlabel('Fold number')
plt.ylabel('Test AUC')
plt.legend()
plt.grid(True)
plt.savefig('kfold_test_auc_results_ppg_feat.png', dpi=300)
plt.show()

print(f"\nPath where all top models are stored: {best_model_path_overall}, AUC: {best_auc_overall:.4f}")

if best_model_path_overall and os.path.exists(best_model_path_overall):
    print(f"\nDetailed evaluation of the overall best model ({best_model_path_overall}:")
    best_model_for_analysis = load_model(best_model_path_overall)

    print("\n(Note: This detailed evaluation is performed on the test set of the last fold processed.)")

    final_eval_X_raw = X_raw_test 
    final_eval_X_feat = X_feat_test
    final_eval_y = y_test

    y_pred_prob_final = best_model_for_analysis.predict([final_eval_X_raw, final_eval_X_feat], verbose=0).flatten()
    y_pred_label_final = (y_pred_prob_final > 0.5).astype(int)

    final_test_loss, final_test_acc, final_test_auc = best_model_for_analysis.evaluate([final_eval_X_raw, final_eval_X_feat], final_eval_y, verbose=0)
    final_precision = precision_score(final_eval_y, y_pred_label_final)
    final_recall = recall_score(final_eval_y, y_pred_label_final)
    final_f1 = f1_score(final_eval_y, y_pred_label_final)
    final_conf_matrix = confusion_matrix(final_eval_y, y_pred_label_final)

    print(f"Test Loss    : {final_test_loss:.4f}")
    print(f"Test Accuracy: {final_test_acc:.4f}")
    print(f"Test AUC     : {final_test_auc:.4f}")
    print(f"Precision         : {final_precision:.4f}")
    print(f"Recall         : {final_recall:.4f}")
    print(f"F1 Score      : {final_f1:.4f}")
    print("Confusion Matrix:")
    print(final_conf_matrix)
else:
    print(f"\nCould not find the best performing model file ({best_model_path_overall}).")