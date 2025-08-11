# This code is an original work authored by Hyunkyung Lee.
# It was developed as part of the MSc CyberSecurity individual project at King's College London.
# This code is written for the Google Colab environment.

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

SEED = 42
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

# Data load
def load_multimodal_data(split_dir, iris_expected_feat_dim=None):
    """Raw PPG signals, CSV features, and Iris patterns simultaneously load"""
    X_raw, X_feat, X_iris, y = [], [], [], []

    # Original data (label 0)
    orig_raw_dir = os.path.join(split_dir, 'original', 'processed_signals')
    orig_feat_dir = os.path.join(split_dir, 'original', 'derived_features')
    orig_iris_dir = os.path.join(split_dir, 'original', 'processed_iris')

    raw_files = sorted([f for f in os.listdir(orig_raw_dir) if f.endswith('.csv')])

    for file in raw_files:
        base_name = os.path.splitext(file)[0]

        # rPPG signal
        raw_path = os.path.join(orig_raw_dir, file)
        raw_df = pd.read_csv(raw_path)
        X_raw.append(raw_df.filter(like='_filtered').values)

        # fft features
        feat_path = os.path.join(orig_feat_dir, file)
        feat_df = pd.read_csv(feat_path)
        X_feat.append(feat_df.drop(columns='frame').values)

        # iris pattern features
        iris_path = os.path.join(orig_iris_dir, file)
        if os.path.exists(iris_path):
            iris_df = pd.read_csv(iris_path)
            X_iris.append(iris_df.drop(columns=['frame']).values)
        else:
            if iris_expected_feat_dim is None:
                raise ValueError("Iris data not found and 'iris_expected_feat_dim' is not specified. Please determine the actual iris feature dimension first.")
            X_iris.append(np.zeros((0, iris_expected_feat_dim)))
        y.append(0)

    # Deepfake data (label 1)
    df_raw_dir = os.path.join(split_dir, 'deepfakes', 'processed_signals')
    df_feat_dir = os.path.join(split_dir, 'deepfakes', 'derived_features')
    df_iris_dir = os.path.join(split_dir, 'deepfakes', 'processed_iris')

    df_raw_files = sorted([f for f in os.listdir(df_raw_dir) if f.endswith('.csv')])

    for file in df_raw_files:
        base_name = os.path.splitext(file)[0]

        # rPPG signal
        raw_path = os.path.join(df_raw_dir, file)
        raw_df = pd.read_csv(raw_path)
        X_raw.append(raw_df.filter(like='_filtered').values)

        # fft features
        feat_path = os.path.join(df_feat_dir, file)
        feat_df = pd.read_csv(feat_path)
        X_feat.append(feat_df.drop(columns='frame').values)

        # iris pattern features
        iris_path = os.path.join(df_iris_dir, file)
        if os.path.exists(iris_path):
            iris_df = pd.read_csv(iris_path)
            X_iris.append(iris_df.drop(columns=['frame']).values)
        else:
            if iris_expected_feat_dim is None:
                raise ValueError("Iris data not found and 'iris_expected_feat_dim' is not specified. Please determine the actual iris feature dimension first.")
            X_iris.append(np.zeros((0, iris_expected_feat_dim)))

        y.append(1)
    

    max_raw_len = max(arr.shape[0] for arr in X_raw) if X_raw else 1 
    max_feat_len = max(arr.shape[0] for arr in X_feat) if X_feat else 1 
    max_iris_len = max(arr.shape[0] for arr in X_iris if arr.shape[0] > 0) if any(arr.shape[0] > 0 for arr in X_iris) else 1 

    X_raw_padded = [np.pad(arr, ((0, max_raw_len - arr.shape[0]), (0,0) if arr.ndim > 1 else (0,0)), 'constant') for arr in X_raw]
    X_feat_padded = [np.pad(arr, ((0, max_feat_len - arr.shape[0]), (0,0) if arr.ndim > 1 else (0,0)), 'constant') for arr in X_feat]
    

    X_iris_padded = []
    for arr in X_iris:
        if arr.shape[0] == 0: 
            padded_arr = np.zeros((max_iris_len, iris_expected_feat_dim))
        else:
            padded_arr = np.pad(arr, ((0, max_iris_len - arr.shape[0]), (0,0) if arr.ndim > 1 else (0,0)), 'constant')
        X_iris_padded.append(padded_arr)

    return (np.array(X_raw_padded),
            np.array(X_feat_padded),
            np.array(X_iris_padded),
            np.array(y))

# Positional Embedding
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        self.position_embedding = None

    def build(self, input_shape):
        if self.position_embedding is None:
            self.position_embedding = tf.keras.layers.Embedding(
                input_dim=input_shape[-2],
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
    # 입력 정의
    raw_input = Input(shape=raw_shape, name='raw_ppg_input')
    feat_input = Input(shape=feat_shape, name='csv_feat_input')
    iris_input = Input(shape=iris_shape, name='iris_input')

    # CNN-Bidirectional LSTM
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

    # MLP
    x_feat_dense = Dense(128, activation='relu')(feat_input)
    x_feat_dense = BatchNormalization()(x_feat_dense)
    x_feat_dense = Dropout(dropout_rate)(x_feat_dense)
    feat_output = GlobalAveragePooling1D()(x_feat_dense)
    feat_output = Dense(64, activation='relu')(feat_output)

    # Transformer with Positional Encoding
    num_heads = hp.Choice('num_heads', values=[2, 4, 8])
    key_dim = hp.Choice('key_dim', values=[32, 64, 128])
    ff_dim = hp.Int('ff_dim', min_value=64, max_value=256, step=64)
    attn_dropout = hp.Float('attn_dropout', 0.1, 0.3, step=0.1)
    ffn_dropout = hp.Float('ffn_dropout', 0.1, 0.3, step=0.1)

    # Positional Embedding
    x_iris = PositionalEmbedding(sequence_length=iris_shape[0], output_dim=iris_shape[1])(iris_input)
    x_iris = LayerNormalization(epsilon=1e-6)(x_iris)

    # Transformer Encoder Block 1
    attn_output1 = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=attn_dropout
    )(x_iris, x_iris)
    x_iris_norm1 = LayerNormalization(epsilon=1e-6)(x_iris + attn_output1)

    ffn1 = Dense(ff_dim, activation='relu')(x_iris_norm1)
    ffn1 = Dropout(ffn_dropout)(ffn1)
    ffn1 = Dense(iris_shape[-1])(ffn1)
    x_iris_encoded1 = LayerNormalization(epsilon=1e-6)(x_iris_norm1 + ffn1)

    # Transformer Encoder Block 2
    attn_output2 = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=attn_dropout
    )(x_iris_encoded1, x_iris_encoded1)
    x_iris_norm2 = LayerNormalization(epsilon=1e-6)(x_iris_encoded1 + attn_output2)

    ffn2 = Dense(ff_dim, activation='relu')(x_iris_norm2)
    ffn2 = Dropout(ffn_dropout)(ffn2)
    ffn2 = Dense(iris_shape[-1])(ffn2)
    x_iris_encoded2 = LayerNormalization(epsilon=1e-6)(x_iris_norm2 + ffn2)

    iris_output = GlobalAveragePooling1D()(x_iris_encoded2)
    iris_output = Dense(64, activation='relu')(iris_output)


    # Feature fusion (with cross-modality attention) - using Reshape layers
    # Each modality output is converted into a sequence and input into AdditiveAttention.
    # (batch_size, 64) -> (batch_size, 1, 64)
    ppg_vec = Reshape((1, 64))(ppg_output)
    feat_vec = Reshape((1, 64))(feat_output)
    iris_vec = Reshape((1, 64))(iris_output)

    # Combine all modality vectors into a single sequence (for attention input)
    # (batch_size, 3, 64)
    modalities_combined = Concatenate(axis=1)([ppg_vec, feat_vec, iris_vec])

    # AdditiveAttention
    attention_output = AdditiveAttention()([modalities_combined, modalities_combined])

    fused_features = GlobalAveragePooling1D()(attention_output)

    fusion_units = hp.Int('fusion_units', min_value=64, max_value=256, step=64)
    final_dropout = hp.Float('final_dropout', 0.2, 0.5, step=0.1)

    z = Dense(fusion_units, activation='relu')(fused_features)
    z = BatchNormalization()(z)
    z = Dropout(final_dropout)(z)
    output = Dense(1, activation='sigmoid')(z)

    # Compile
    learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])
    model = Model(inputs=[raw_input, feat_input, iris_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model

# Main execution
from google.colab import drive
drive.mount('/content/drive')

# After splitting data with split.json, data is saved in the split_json folder.
train_dir = '/content/split_data/train'
val_dir = '/content/split_data/val'
test_dir = '/content/split_data/test'

print("Attempting to determine iris feature dimension from training data...")
_temp_iris_dim_check_list = []

for sub_dir in ['original', 'deepfakes']:
    iris_sample_dir = os.path.join(train_dir, sub_dir, 'processed_iris')
    if os.path.exists(iris_sample_dir):
        sample_files = sorted([f for f in os.listdir(iris_sample_dir) if f.endswith('.csv')])
        if sample_files:
            sample_path = os.path.join(iris_sample_dir, sample_files[0])
            try:
                sample_df = pd.read_csv(sample_path)
                _temp_iris_dim_check_list.append(sample_df.drop(columns=['frame']).shape[-1])
            except Exception as e:
                print(f"Warning: Could not read sample iris file {sample_path} to determine dimension. Error: {e}")

if _temp_iris_dim_check_list:
    iris_initial_feat_dim = max(_temp_iris_dim_check_list)
    print(f"Determined iris feature dimension: {iris_initial_feat_dim}")
else:
    print("Warning: Could not determine iris feature dimension from sample files. Using default value 64. Please adjust 'iris_initial_feat_dim' if this is incorrect.")
    iris_initial_feat_dim = 64


# Multimodal data loading
print("Loading training data...")
X_raw_train, X_feat_train, X_iris_train, y_train = load_multimodal_data(train_dir, iris_initial_feat_dim)
print("Loading validation data...")
X_raw_val, X_feat_val, X_iris_val, y_val = load_multimodal_data(val_dir, iris_initial_feat_dim)
print("Loading test data...")
X_raw_test, X_feat_test, X_iris_test, y_test = load_multimodal_data(test_dir, iris_initial_feat_dim)

raw_shape = X_raw_train.shape[1:]
feat_shape = X_feat_train.shape[1:]
iris_shape = X_iris_train.shape[1:] 

print(f"\nTraining data shape:")
print(f"  Raw PPG: {X_raw_train.shape}")
print(f"  CSV Features: {X_feat_train.shape}")
print(f"  Iris Patterns: {X_iris_train.shape}")
print(f"  Labels: {y_train.shape}")

# Experimental setup (same as rPPG-only)
N_RUNS = 5                # Number of multiple runs for ensemble
FIXED_EPOCHS = 100        
MAX_TRIALS = 20           # Number of hyperparameter tuning trials
test_auc_results = []         

for run in range(N_RUNS):
    print(f"\n\n{'='*50}")
    print(f"Run #{run+1}/{N_RUNS} - Multimodal Model (Attention Applied)")
    print(f"{'='*50}")

    # Hyperparameter Tuning
    tuner = kt.RandomSearch(
        build_multimodal_model,
        objective=kt.Objective("val_auc", direction="max"),
        max_trials=MAX_TRIALS,
        executions_per_trial=1,
        directory=f'multimodal_attention_tuner_run_{run}',
        project_name='multimodal_attention'
    )

    # Learning rate scheduling
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    print("\nStarting hyperparameter tuning...")
    tuner.search(
        [X_raw_train, X_feat_train, X_iris_train], y_train,
        validation_data=([X_raw_val, X_feat_val, X_iris_val], y_val),
        epochs=50, # Epochs for tuning
        batch_size=32,
        callbacks=[reduce_lr],
        verbose=1
    )

    # Print optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"\nBest hyperparameters (run #{run+1}):")
    print(f"  conv_filters: {best_hps.get('conv_filters')}")
    print(f"  lstm_units1: {best_hps.get('lstm_units1')}")
    print(f"  lstm_units2: {best_hps.get('lstm_units2')}")
    print(f"  num_heads: {best_hps.get('num_heads')}")
    print(f"  key_dim: {best_hps.get('key_dim')}")
    print(f"  fusion_units: {best_hps.get('fusion_units')}")
    print(f"  learning_rate: {best_hps.get('learning_rate')}")
    print(f"  dropout_rate: {best_hps.get('dropout_rate')}")
    print(f"  attn_dropout: {best_hps.get('attn_dropout')}")
    print(f"  ffn_dropout: {best_hps.get('ffn_dropout')}")
    print(f"  final_dropout: {best_hps.get('final_dropout')}")


    # Building and training the optimal model
    model = tuner.hypermodel.build(best_hps)

    # Checkpoint
    checkpoint = ModelCheckpoint(
        f'best_multimodal_model_run_{run}.keras',
        monitor='val_auc',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )

    print("\nStarting learning for the best model...")
    history = model.fit(
        [X_raw_train, X_feat_train, X_iris_train], y_train,
        validation_data=([X_raw_val, X_feat_val, X_iris_val], y_val),
        epochs=FIXED_EPOCHS,
        batch_size=32,
        callbacks=[checkpoint, reduce_lr],
        verbose=1
    )

    # Load the best model (the best performing model saved during training)
    model = load_model(f'best_multimodal_model_run_{run}.keras',
                       custom_objects={'PositionalEmbedding': PositionalEmbedding})

    # Evaluation
    test_loss, test_acc, test_auc = model.evaluate(
        [X_raw_test, X_feat_test, X_iris_test], y_test, verbose=0
    )
    test_auc_results.append(test_auc)
    print(f"\nRun #{run+1} Test Performance: AUC={test_auc:.4f}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title(f'Multimodal Model Run #{run+1} AUC')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Multimodal Model Run #{run+1} Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'multimodal_attention_training_history_run_{run}.png')
    plt.show()

# Individual Run Results Analysis
print("\n\n" + "="*50)
print("Multimodal Model (Attention Applied) - Individual Run Results")
print("="*50)
print(f"Test AUC mean: {np.mean(test_auc_results):.4f}")
print(f"Test AUC standard deviation: {np.std(test_auc_results):.4f}")
print(f"Individual run test AUCs: {[f'{x:.4f}' for x in test_auc_results]}")

plt.figure(figsize=(8, 6))
plt.bar(range(1, N_RUNS+1), test_auc_results)
plt.axhline(np.mean(test_auc_results), color='r', linestyle='--', label='Mean')
plt.title('Multimodal Model (Attention Applied) - Multi-run Test AUC Results')
plt.xlabel('Run number')
plt.ylabel('Test AUC')
plt.legend()
plt.grid(True)
plt.savefig('multimodal_attention_multi_run_results.png', dpi=300)
plt.show()

# Save model in Google Drive
for run in range(N_RUNS):
    model_path = f'/content/drive/MyDrive/multimodal_attention_model_run_{run}.keras'

    model = load_model(f'best_multimodal_model_run_{run}.keras',
                       custom_objects={'PositionalEmbedding': PositionalEmbedding})
    model.save(model_path)
    print(f"Multimodal Attention Model Run #{run+1} saved: {model_path}")

# Ensemble learning and detailed performance analysis
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, accuracy_score

print("\n\n" + "="*50)
print("Ensemble Model (Soft Voting) Detailed Evaluation")
print("="*50)

ensemble_predictions_prob = []

# Load and predict all top-performing models trained over N_RUNS
for run in range(N_RUNS):
    model_path = f'best_multimodal_model_run_{run}.keras'

    model = load_model(model_path, custom_objects={'PositionalEmbedding': PositionalEmbedding})
    
    y_pred_prob_current_model = model.predict(
        [X_raw_test, X_feat_test, X_iris_test], verbose=0
    ).flatten()
    ensemble_predictions_prob.append(y_pred_prob_current_model)

# Average of predicted probabilities across all models (Soft Voting)
ensemble_predictions_prob = np.array(ensemble_predictions_prob)
final_avg_predictions_prob = np.mean(ensemble_predictions_prob, axis=0)

final_pred_labels = (final_avg_predictions_prob > 0.5).astype(int)

# Calculating ensemble performance metrics
ensemble_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_test, final_avg_predictions_prob)).numpy()
ensemble_accuracy = accuracy_score(y_test, final_pred_labels)
ensemble_auc = roc_auc_score(y_test, final_avg_predictions_prob)
ensemble_precision = precision_score(y_test, final_pred_labels)
ensemble_recall = recall_score(y_test, final_pred_labels)
ensemble_f1 = f1_score(y_test, final_pred_labels)
ensemble_conf_matrix = confusion_matrix(y_test, final_pred_labels)

print(f"Ensemble Test Loss   : {ensemble_loss:.4f}")
print(f"Ensemble Test Accuracy: {ensemble_accuracy:.4f}")
print(f"Ensemble Test AUC     : {ensemble_auc:.4f}")
print(f"Ensemble Precision    : {ensemble_precision:.4f}")
print(f"Ensemble Recall       : {ensemble_recall:.4f}")
print(f"Ensemble F1 Score     : {ensemble_f1:.4f}")
print("Ensemble Confusion Matrix:")
print(ensemble_conf_matrix)