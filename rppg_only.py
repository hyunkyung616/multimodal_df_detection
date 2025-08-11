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
from tensorflow.keras.layers import Input, Conv1D, LSTM, Bidirectional, Dense, Dropout, BatchNormalization, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import kerastuner as kt

SEED = 42
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

# Data load
def load_data(split_dir):

    X_raw = []  # processed_signals
    X_feat = []  # derived_features
    y = []      # label

    # Orignal data (label 0)
    orig_raw_dir = os.path.join(split_dir, 'original', 'processed_signals')
    orig_feat_dir = os.path.join(split_dir, 'original', 'derived_features')

    raw_files = sorted([f for f in os.listdir(orig_raw_dir) if f.endswith('.csv')])

    for file in raw_files:
        # rPPG signal
        raw_path = os.path.join(orig_raw_dir, file)
        raw_df = pd.read_csv(raw_path)
        X_raw.append(raw_df.filter(like='_filtered').values)

        # fft features
        feat_path = os.path.join(orig_feat_dir, file)
        feat_df = pd.read_csv(feat_path)
        X_feat.append(feat_df.drop(columns='frame').values)

        y.append(0)

    # Deepfake data (label 1)
    df_raw_dir = os.path.join(split_dir, 'deepfakes', 'processed_signals')
    df_feat_dir = os.path.join(split_dir, 'deepfakes', 'derived_features')

    df_raw_files = sorted([f for f in os.listdir(df_raw_dir) if f.endswith('.csv')])

    for file in df_raw_files:
        # rPPG signal
        raw_path = os.path.join(df_raw_dir, file)
        raw_df = pd.read_csv(raw_path)
        X_raw.append(raw_df.filter(like='_filtered').values)

        # fft features
        feat_path = os.path.join(df_feat_dir, file)
        feat_df = pd.read_csv(feat_path)
        X_feat.append(feat_df.drop(columns='frame').values)

        y.append(1)

    max_raw_len = max(arr.shape[0] for arr in X_raw) if X_raw else 1
    max_feat_len = max(arr.shape[0] for arr in X_feat) if X_feat else 1

    X_raw_padded = [np.pad(arr, ((0, max_raw_len - arr.shape[0]), (0,0) if arr.ndim > 1 else (0,0)), 'constant') for arr in X_raw]
    X_feat_padded = [np.pad(arr, ((0, max_feat_len - arr.shape[0]), (0,0) if arr.ndim > 1 else (0,0)), 'constant') for arr in X_feat]

    return np.array(X_raw_padded), np.array(X_feat_padded), np.array(y)


# rPPG-only model build
def build_ppg_feat_model(hp):

    raw_input = Input(shape=raw_shape, name='raw_ppg_input')

    # CNN-Bidirectional LSTM
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

    # FFT features
    feat_input = Input(shape=feat_shape, name='csv_feat_input')

    # MLP
    x_feat_dense = Dense(128, activation='relu')(feat_input)
    x_feat_dense = BatchNormalization()(x_feat_dense)
    x_feat_dense = Dropout(dropout_rate)(x_feat_dense)
    feat_output = GlobalAveragePooling1D()(x_feat_dense)
    feat_output = Dense(64, activation='relu')(feat_output)

    # Concatenate
    merged = Concatenate()([ppg_output, feat_output])

    # Final classifier
    fusion_units = hp.Int('fusion_units', min_value=64, max_value=256, step=64)
    final_dropout = hp.Float('final_dropout', 0.2, 0.5, step=0.1)

    z = Dense(fusion_units, activation='relu')(merged)
    z = BatchNormalization()(z)
    z = Dropout(final_dropout)(z)
    output = Dense(1, activation='sigmoid')(z)

    # Compile
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model = Model(inputs=[raw_input, feat_input], outputs=output)
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

print("Loading train data...")
X_raw_train, X_feat_train, y_train = load_data(train_dir)
print("Loading val data...")
X_raw_val, X_feat_val, y_val = load_data(val_dir)
print("Loading test data...")
X_raw_test, X_feat_test, y_test = load_data(test_dir)

raw_shape = X_raw_train.shape[1:]
feat_shape = X_feat_train.shape[1:]
print(f"\nTraining data format:")
print(f"  Raw PPG: {X_raw_train.shape}")
print(f"  CSV Features: {X_feat_train.shape}")
print(f"  Label: {y_train.shape}")

N_RUNS = 5          # Number of multiple executions
FIXED_EPOCHS = 100  
test_results = []   

for run in range(N_RUNS):
    print(f"\n\n{'='*50}")
    print(f"Run #{run+1}/{N_RUNS}")
    print(f"{'='*50}")

    # Hyperparameter tuning
    tuner = kt.RandomSearch(
        build_ppg_feat_model,
        objective=kt.Objective("val_auc", direction="max"),
        max_trials=20,
        executions_per_trial=1,
        directory=f'tuner_run_ppg_feat_bidirectional_{run}',
        project_name='ppg_and_derived_features_bidirectional'
    )

    # Learning rate scheduling
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    print("\nStart hyperparameter tuning...")
    tuner.search(
        [X_raw_train, X_feat_train], y_train,
        validation_data=([X_raw_val, X_feat_val], y_val),
        epochs=50,
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
    print(f"  dropout_rate: {best_hps.get('dropout_rate')}")
    print(f"  fusion_units: {best_hps.get('fusion_units')}")
    print(f"  final_dropout: {best_hps.get('final_dropout')}")
    print(f"  learning_rate: {best_hps.get('learning_rate')}")

    # Build best model
    model = tuner.hypermodel.build(best_hps)

    # Checkpoint
    checkpoint = ModelCheckpoint(
        f'best_model_ppg_feat_bidirectional_run_{run}.h5',
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    print("\nStart learning the best model...")
    history = model.fit(
        [X_raw_train, X_feat_train], y_train,
        validation_data=([X_raw_val, X_feat_val], y_val),
        epochs=FIXED_EPOCHS,
        batch_size=32,
        callbacks=[checkpoint, reduce_lr],
        verbose=1
    )

    # Load and evaluate the best model
    model = tf.keras.models.load_model(f'best_model_ppg_feat_bidirectional_run_{run}.h5')
    test_loss, test_acc, test_auc = model.evaluate([X_raw_test, X_feat_test], y_test)
    test_results.append(test_auc)
    print(f"\nRun #{run+1} Test Performance: AUC={test_auc:.4f}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title(f'Run #{run+1} AUC (PPG + Derived Features - Bidirectional LSTM)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Run #{run+1} Loss (PPG + Derived Features - Bidirectional LSTM)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'training_history_ppg_feat_bidirectional_run_{run}.png')
    plt.show()

# Result Evaluation
print("\n\n" + "="*50)
print("Multi-run average results (PPG + Derived Features - Bidirectional LSTM)")
print("="*50)
print(f"Test AUC mean: {np.mean(test_results):.4f}")
print(f"Test AUC standard deviation: {np.std(test_results):.4f}")
print(f"Individual execution results: {[f'{x:.4f}' for x in test_results]}")

plt.figure(figsize=(8, 6))
plt.bar(range(1, N_RUNS+1), test_results)
plt.axhline(np.mean(test_results), color='r', linestyle='--', label='mean')
plt.title('Multi-run test AUC results (PPG + Derived Features - Bidirectional LSTM)')
plt.xlabel('Run number')
plt.ylabel('Test AUC')
plt.legend()
plt.grid(True)
plt.savefig('multi_run_results_ppg_feat_bidirectional.png', dpi=300)
plt.show()

# Save the result in Google Drive
for run in range(N_RUNS):
    model_path = f'/content/drive/MyDrive/ppg_feat_bidirectional_model_run_{run}.h5' 
    tf.keras.models.save_model(
        tf.keras.models.load_model(f'best_model_ppg_feat_bidirectional_run_{run}.h5'), 
        model_path
    )
    print(f"Model run #{run+1} save complete: {model_path}")


# Detailed evaluation
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

# Load the highest performing model
best_run = np.argmax(test_results)
model = load_model(f'best_model_ppg_feat_bidirectional_run_{best_run}.h5') 

print(f"\nBest Performance Model (Run #{best_run+1}) Detailed Evaluation (PPG + Derived Features - Bidirectional LSTM):") 

X_test_raw = X_raw_test
X_test_feat = X_feat_test
y_true = y_test

y_pred_prob = model.predict([X_test_raw, X_test_feat], verbose=0).flatten()
y_pred_label = (y_pred_prob > 0.5).astype(int)

test_loss, test_acc, test_auc = model.evaluate([X_test_raw, X_test_feat], y_true, verbose=0)
precision = precision_score(y_true, y_pred_label)
recall = recall_score(y_true, y_pred_label)
f1 = f1_score(y_true, y_pred_label)
conf_matrix = confusion_matrix(y_true, y_pred_label)

print(f"Test Loss    : {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test AUC     : {test_auc:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1 Score     : {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)