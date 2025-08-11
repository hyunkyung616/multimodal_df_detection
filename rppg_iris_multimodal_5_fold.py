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
from tensorflow.keras.layers import (Input, Conv1D, LSTM, Dense, Dropout, BatchNormalization,
                                     MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D,
                                     Concatenate, AdditiveAttention, Lambda, Reshape)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import kerastuner as kt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, accuracy_score


SEED = 42
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

raw_shape = None
feat_shape = None
iris_shape = None
iris_initial_feat_dim = None

# Data load
def load_multimodal_data(data_dir, iris_expected_feat_dim=None):
    """Raw PPG signals, CSV features, and Iris patterns simultaneously load"""
    X_raw, X_feat, X_iris, y = [], [], [], []

    orig_raw_dir = os.path.join(data_dir, 'original', 'processed_signals')
    orig_feat_dir = os.path.join(data_dir, 'original', 'derived_features')
    orig_iris_dir = os.path.join(data_dir, 'original', 'processed_iris')
    df_raw_dir = os.path.join(data_dir, 'deepfakes', 'processed_signals')
    df_feat_dir = os.path.join(data_dir, 'deepfakes', 'derived_features')
    df_iris_dir = os.path.join(data_dir, 'deepfakes', 'processed_iris')

    dirs_to_check = [orig_raw_dir, orig_feat_dir, orig_iris_dir, df_raw_dir, df_feat_dir, df_iris_dir]
    for dir_path in dirs_to_check:
        if not os.path.exists(dir_path):
            print(f"Error: Directory not found at: {dir_path}")
            return np.array([]), np.array([]), np.array([]), np.array([])

    # Original data (label 0)
    raw_files = sorted([f for f in os.listdir(orig_raw_dir) if f.endswith('.csv')])
    for file in raw_files:
        raw_path = os.path.join(orig_raw_dir, file)
        feat_path = os.path.join(orig_feat_dir, file)
        iris_path = os.path.join(orig_iris_dir, file)

        if not os.path.exists(feat_path) or not os.path.exists(iris_path):
            print(f"Warning: Missing feature or iris file for '{file}'. Skipping sample.")
            continue

        try:
            raw_df = pd.read_csv(raw_path)
            X_raw.append(raw_df.filter(like='_filtered').values)
            feat_df = pd.read_csv(feat_path)
            X_feat.append(feat_df.drop(columns='frame').values)
            iris_df = pd.read_csv(iris_path)
            X_iris.append(iris_df.drop(columns=['frame']).values)
            y.append(0)
        except Exception as e:
            print(f"Error reading files for '{file}': {e}. Skipping sample.")
            continue

    # Deepfake data (label 1)
    df_raw_files = sorted([f for f in os.listdir(df_raw_dir) if f.endswith('.csv')])
    for file in df_raw_files:
        raw_path = os.path.join(df_raw_dir, file)
        feat_path = os.path.join(df_feat_dir, file)
        iris_path = os.path.join(df_iris_dir, file)

        if not os.path.exists(feat_path) or not os.path.exists(iris_path):
            print(f"Warning: Missing feature or iris file for '{file}'. Skipping sample.")
            continue

        try:
            raw_df = pd.read_csv(raw_path)
            X_raw.append(raw_df.filter(like='_filtered').values)
            feat_df = pd.read_csv(feat_path)
            X_feat.append(feat_df.drop(columns='frame').values)
            iris_df = pd.read_csv(iris_path)
            X_iris.append(iris_df.drop(columns=['frame']).values)
            y.append(1)
        except Exception as e:
            print(f"Error reading files for '{file}': {e}. Skipping sample.")
            continue

    max_raw_len = max(arr.shape[0] for arr in X_raw) if X_raw else 1
    max_feat_len = max(arr.shape[0] for arr in X_feat) if X_feat else 1
    max_iris_len = max(arr.shape[0] for arr in X_iris) if X_iris else 1

    X_raw_padded = [np.pad(arr, ((0, max_raw_len - arr.shape[0]), (0,0)), 'constant') for arr in X_raw]
    X_feat_padded = [np.pad(arr, ((0, max_feat_len - arr.shape[0]), (0,0)), 'constant') for arr in X_feat]
    X_iris_padded = [np.pad(arr, ((0, max_iris_len - arr.shape[0]), (0,0)), 'constant') for arr in X_iris]

    return (np.array(X_raw_padded),
            np.array(X_feat_padded),
            np.array(X_iris_padded),
            np.array(y))

# Positional Embedding
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, output_dim, *args, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        self.position_embedding = None

    def build(self, input_shape):
        if self.position_embedding is None:
            self.position_embedding = tf.keras.layers.Embedding(
                input_dim=self.sequence_length,
                output_dim=self.output_dim
            )
        super().build(input_shape)

    def call(self, inputs):
        length = tf.shape(inputs)[-2]
        positions = tf.range(start=0, limit=length, dtype=tf.int32)
        embedded_positions = self.position_embedding(positions)
        return inputs + embedded_positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "output_dim": self.output_dim,
        })
        return config

# Build multimodal model
def build_multimodal_model(hp):
    raw_input = Input(shape=raw_shape, name='raw_ppg_input')
    feat_input = Input(shape=feat_shape, name='csv_feat_input')
    iris_input = Input(shape=iris_shape, name='iris_input')

    # rPPG signal
    conv_filters = hp.Int('conv_filters', min_value=32, max_value=128, step=32)
    x_ppg = Conv1D(conv_filters, 3, activation='relu', padding='same')(raw_input)
    x_ppg = BatchNormalization()(x_ppg)
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    x_ppg = Dropout(dropout_rate)(x_ppg)

    lstm_units1 = hp.Int('lstm_units1', min_value=64, max_value=256, step=64)
    x_ppg = tf.keras.layers.Bidirectional(LSTM(lstm_units1, return_sequences=True))(x_ppg)
    x_ppg = Dropout(dropout_rate)(x_ppg)

    lstm_units2 = hp.Int('lstm_units2', min_value=32, max_value=128, step=32)
    x_ppg = tf.keras.layers.Bidirectional(LSTM(lstm_units2))(x_ppg)
    ppg_output = Dense(64, activation='relu')(x_ppg)

    # FFT features
    x_feat_dense = Dense(128, activation='relu')(feat_input)
    x_feat_dense = BatchNormalization()(x_feat_dense)
    x_feat_dense = Dropout(dropout_rate)(x_feat_dense)
    feat_output = GlobalAveragePooling1D()(x_feat_dense)
    feat_output = Dense(64, activation='relu')(feat_output)

    # Iris pattern features (Transformer)
    num_heads = hp.Choice('num_heads', values=[2, 4, 8])
    key_dim = hp.Choice('key_dim', values=[32, 64, 128])
    ff_dim = hp.Int('ff_dim', min_value=64, max_value=256, step=64)
    attn_dropout = hp.Float('attn_dropout', 0.1, 0.3, step=0.1)
    ffn_dropout = hp.Float('ffn_dropout', 0.1, 0.3, step=0.1)

    x_iris = PositionalEmbedding(sequence_length=iris_shape[0], output_dim=iris_shape[1])(iris_input)
    x_iris = LayerNormalization(epsilon=1e-6)(x_iris)

    attn_output1 = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=attn_dropout)(x_iris, x_iris)
    x_iris_norm1 = LayerNormalization(epsilon=1e-6)(x_iris + attn_output1)

    ffn1 = Dense(ff_dim, activation='relu')(x_iris_norm1)
    ffn1 = Dropout(ffn_dropout)(ffn1)
    ffn1 = Dense(iris_shape[-1])(ffn1)
    x_iris_encoded1 = LayerNormalization(epsilon=1e-6)(x_iris_norm1 + ffn1)

    attn_output2 = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=attn_dropout)(x_iris_encoded1, x_iris_encoded1)
    x_iris_norm2 = LayerNormalization(epsilon=1e-6)(x_iris_encoded1 + attn_output2)

    ffn2 = Dense(ff_dim, activation='relu')(x_iris_norm2)
    ffn2 = Dropout(ffn_dropout)(ffn2)
    ffn2 = Dense(iris_shape[-1])(ffn2)
    x_iris_encoded2 = LayerNormalization(epsilon=1e-6)(x_iris_norm2 + ffn2)

    iris_output = GlobalAveragePooling1D()(x_iris_encoded2)
    iris_output = Dense(64, activation='relu')(iris_output)

    # Feature Fusion (Attention)
    ppg_vec = Reshape((1, 64))(ppg_output)
    feat_vec = Reshape((1, 64))(feat_output)
    iris_vec = Reshape((1, 64))(iris_output)
    modalities_combined = Concatenate(axis=1)([ppg_vec, feat_vec, iris_vec])
    attention_output = AdditiveAttention()([modalities_combined, modalities_combined])
    fused_features = GlobalAveragePooling1D()(attention_output)

    # Classifier
    fusion_units = hp.Int('fusion_units', min_value=64, max_value=256, step=64)
    final_dropout = hp.Float('final_dropout', 0.2, 0.5, step=0.1)
    z = Dense(fusion_units, activation='relu')(fused_features)
    z = BatchNormalization()(z)
    z = Dropout(final_dropout)(z)
    output = Dense(1, activation='sigmoid')(z)

    # Compile Model
    learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])
    model = Model(inputs=[raw_input, feat_input, iris_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model

# Main Execution (K-fold applied)
# Load all data (load all data at once for K-fold)
all_data_dir = '/content/data_final'

print("Attempting to determine iris feature dimension from all data directory...")
_temp_iris_dim_check_list = []
for sub_dir in ['original', 'deepfakes']:
    iris_sample_dir = os.path.join(all_data_dir, sub_dir, 'processed_iris')
    if os.path.exists(iris_sample_dir):
        sample_files = sorted([f for f in os.listdir(iris_sample_dir) if f.endswith('.csv')])
        if sample_files:
            sample_path = os.path.join(iris_sample_dir, sample_files[0])
            try:
                sample_df = pd.read_csv(sample_path)
                _temp_iris_dim_check_list.append(sample_df.drop(columns=['frame']).shape[-1])
            except Exception as e:
                print(f"Warning: Could not read sample iris file {sample_path}. Error: {e}")

if _temp_iris_dim_check_list:
    iris_initial_feat_dim = max(_temp_iris_dim_check_list)
    print(f"Determined iris feature dimension: {iris_initial_feat_dim}")
else:
    print("Error: Could not determine iris feature dimension. Please check data directory. Exiting.")
    exit()

print("Loading all data for K-fold cross-validation...")
X_raw_all, X_feat_all, X_iris_all, y_all = load_multimodal_data(all_data_dir, iris_expected_feat_dim=iris_initial_feat_dim)

if X_raw_all.size == 0 or X_feat_all.size == 0 or X_iris_all.size == 0:
    print("\nError: No data loaded. Please check data_dir path and structure. Exiting.")
    exit()

raw_shape = X_raw_all.shape[1:]
feat_shape = X_feat_all.shape[1:]
iris_shape = X_iris_all.shape[1:]

print(f"\nLoaded all data shapes:")
print(f"  Raw PPG: {X_raw_all.shape}")
print(f"  CSV Features: {X_feat_all.shape}")
print(f"  Iris Patterns: {X_iris_all.shape}")
print(f"  Labels: {y_all.shape}")

# K-fold
N_SPLITS = 5
FIXED_EPOCHS = 100
MAX_TRIALS = 20
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

fold_results = {
    'loss': [], 'accuracy': [], 'auc': [], 'precision': [], 'recall': [], 'f1_score': []
}

all_predictions_list = []
all_labels_list = []

for fold, (train_val_idx, test_idx) in enumerate(skf.split(X_raw_all, y_all)):
    print(f"\n\n{'='*50}")
    print(f"--- Starting Fold #{fold+1}/{N_SPLITS} ---")
    print(f"{'='*50}")

    X_raw_train_val, X_raw_test = X_raw_all[train_val_idx], X_raw_all[test_idx]
    X_feat_train_val, X_feat_test = X_feat_all[train_val_idx], X_feat_all[test_idx]
    X_iris_train_val, X_iris_test = X_iris_all[train_val_idx], X_iris_all[test_idx]
    y_train_val, y_test = y_all[train_val_idx], y_all[test_idx]

    inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx_inner, val_idx_inner = next(inner_skf.split(X_raw_train_val, y_train_val))

    X_raw_train, X_raw_val = X_raw_train_val[train_idx_inner], X_raw_train_val[val_idx_inner]
    X_feat_train, X_feat_val = X_feat_train_val[train_idx_inner], X_feat_train_val[val_idx_inner]
    X_iris_train, X_iris_val = X_iris_train_val[train_idx_inner], X_iris_train_val[val_idx_inner]
    y_train, y_val = y_train_val[train_idx_inner], y_train_val[val_idx_inner]

    print(f"\nFold {fold+1} Data Shapes:")
    print(f"  Train Raw PPG: {X_raw_train.shape}, Feat: {X_feat_train.shape}, Iris: {X_iris_train.shape}, Label: {y_train.shape}")
    print(f"  Val Raw PPG: {X_raw_val.shape}, Feat: {X_feat_val.shape}, Iris: {X_iris_val.shape}, Label: {y_val.shape}")
    print(f"  Test Raw PPG: {X_raw_test.shape}, Feat: {X_feat_test.shape}, Iris: {X_iris_test.shape}, Label: {y_test.shape}")

    tuner = kt.RandomSearch(
        build_multimodal_model,
        objective=kt.Objective("val_auc", direction="max"),
        max_trials=MAX_TRIALS,
        executions_per_trial=1,
        directory=f'tuner_results/multimodal_fold_{fold}',
        project_name='multimodal_kfold_tuning'
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1
    )

    print(f"\nFold {fold+1}: Starting hyperparameter tuning...")
    tuner.search(
        [X_raw_train, X_feat_train, X_iris_train], y_train,
        validation_data=([X_raw_val, X_feat_val, X_iris_val], y_val),
        epochs=50,
        batch_size=32,
        callbacks=[reduce_lr],
        verbose=1
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"\nFold #{fold+1} Best hyperparameters:")
    print(f"  conv_filters: {best_hps.get('conv_filters')}")
    print(f"  lstm_units1: {best_hps.get('lstm_units1')}")
    print(f"  lstm_units2: {best_hps.get('lstm_units2')}")
    print(f"  num_heads: {best_hps.get('num_heads')}")
    print(f"  key_dim: {best_hps.get('key_dim')}")
    print(f"  fusion_units: {best_hps.get('fusion_units')}")
    print(f"  learning_rate: {best_hps.get('learning_rate')}")
    print(f"  dropout_rate: {best_hps.get('dropout_rate')}")

    model = tuner.hypermodel.build(best_hps)

    fold_model_filepath = f'best_multimodal_model_fold_{fold+1}.keras'
    checkpoint = ModelCheckpoint(
        filepath=fold_model_filepath, monitor='val_auc', save_best_only=True, mode='max', verbose=1
    )

    print(f"\nFold {fold+1}: Starting learning for the best model...")
    history = model.fit(
        [X_raw_train, X_feat_train, X_iris_train], y_train,
        validation_data=([X_raw_val, X_feat_val, X_iris_val], y_val),
        epochs=FIXED_EPOCHS,
        batch_size=32,
        callbacks=[checkpoint, reduce_lr],
        verbose=1
    )

    best_fold_model = load_model(fold_model_filepath, custom_objects={'PositionalEmbedding': PositionalEmbedding})

    print(f"\nFold {fold+1}: Evaluating on test set...")
    test_loss, test_accuracy, test_auc = best_fold_model.evaluate([X_raw_test, X_feat_test, X_iris_test], y_test, verbose=0)

    y_pred_prob = best_fold_model.predict([X_raw_test, X_feat_test, X_iris_test]).flatten()
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

    all_predictions_list.extend(y_pred_prob)
    all_labels_list.extend(y_test)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title(f'Fold #{fold+1} AUC (Multimodal Model)')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Fold #{fold+1} Loss (Multimodal Model)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'multimodal_model_history_fold_{fold+1}.png')
    plt.show()

    tf.keras.backend.clear_session()

print("\n\n" + "="*50)
print(f"{N_SPLITS}-Fold Cross-Validation Final Results")
print("="*50)
for metric, values in fold_results.items():
    print(f"Average {metric.capitalize()}: {np.mean(values):.4f} (Std: {np.std(values):.4f})")
    print(f"Individual {metric.capitalize()}s: {[f'{v:.4f}' for v in values]}")

plt.figure(figsize=(8, 6))
plt.bar(range(1, N_SPLITS + 1), fold_results['auc'])
plt.axhline(np.mean(fold_results['auc']), color='r', linestyle='--', label=f'Mean AUC: {np.mean(fold_results["auc"]):.4f}')
plt.title(f'{N_SPLITS}-Fold Cross-Validation Test AUC Results')
plt.xlabel('Fold number')
plt.ylabel('Test AUC')
plt.legend()
plt.grid(True)
plt.savefig('kfold_multimodal_test_auc_results.png', dpi=300)
plt.show()

# Ensemble Learning (Soft Voting)
print("\n\n" + "="*50)
print("Ensemble Model (Soft Voting) Detailed Evaluation")
print("="*50)

final_avg_predictions_prob = np.array(all_predictions_list)
final_pred_labels = (final_avg_predictions_prob > 0.5).astype(int)

ensemble_accuracy = accuracy_score(all_labels_list, final_pred_labels)
ensemble_auc = roc_auc_score(all_labels_list, final_avg_predictions_prob)
ensemble_precision = precision_score(all_labels_list, final_pred_labels)
ensemble_recall = recall_score(all_labels_list, final_pred_labels)
ensemble_f1 = f1_score(all_labels_list, final_pred_labels)
ensemble_conf_matrix = confusion_matrix(all_labels_list, final_pred_labels)

print(f"Ensemble Test Loss    : N/A (Loss cannot be directly computed this way)")
print(f"Ensemble Test Accuracy: {ensemble_accuracy:.4f}")
print(f"Ensemble Test AUC     : {ensemble_auc:.4f}")
print(f"Ensemble Precision    : {ensemble_precision:.4f}")
print(f"Ensemble Recall       : {ensemble_recall:.4f}")
print(f"Ensemble F1 Score     : {ensemble_f1:.4f}")
print("Ensemble Confusion Matrix:")
print(ensemble_conf_matrix)