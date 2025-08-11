# This code is an original work authored by Hyunkyung Lee.
# It was developed as part of the MSc CyberSecurity individual project at King's College London.

import os
import numpy as np
import pandas as pd
from scipy.fft import fft, ifft

def fft_lmn_resample(signal, orig_len, target_len=300):

    # FFT-LMN resampling function (rPPG signal)
    spectrum = fft(signal)
    if target_len < orig_len:
        mid = target_len // 2
        new_spectrum = np.concatenate([spectrum[:mid], spectrum[-mid:]])
    else:
        new_spectrum = np.zeros(target_len, dtype=complex)
        mid = orig_len // 2
        new_spectrum[:mid] = spectrum[:mid]
        new_spectrum[-mid:] = spectrum[-mid:]
    resampled = np.real(ifft(new_spectrum))
    return resampled * np.sqrt(orig_len / target_len)

def process_csv(input_path, output_path):
    df = pd.read_csv(input_path)
    orig_len = len(df)
    
    # rPPG
    ts_cols = [col for col in df.columns if '_filtered' in col]
    ts_data = {}
    for col in ts_cols:
        ts_data[col] = fft_lmn_resample(df[col].values, orig_len)
    
    # features
    non_ts_cols = [col for col in df.columns 
                   if col != 'frame' and col not in ts_cols]
    non_ts_data = {}
    for col in non_ts_cols:
        non_ts_data[col] = np.interp(
            np.linspace(0, orig_len-1, 300),
            np.arange(orig_len),
            df[col].values
        )
    
    new_df = pd.DataFrame({
        **ts_data,
        **non_ts_data,
        'frame': range(1, 301)
    })
    
    column_order = ['frame'] + ts_cols + non_ts_cols
    new_df = new_df[column_order]
    new_df.to_csv(output_path, index=False)

def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file)
            process_csv(input_path, output_path)
            print(f"{file}: {len(pd.read_csv(input_path))} → 300 Frame")

# !Important!
# The current path is an absolute path and is set to match my experimental environment.
# Modify it as appropriate for your experimental environment.
# Basically, when you run common_frame.py, the files are saved in the processed_common_frames folder.

if __name__ == "__main__":
    base_path = r"C:\Users\rinah\Desktop\Project\Data\interpolated_data\processed_common_frames"
    output_base = os.path.join(base_path, "resampled_300")
    
    folders = [
        "original/processed_iris",
        "original/processed_signals",
        "deepfakes/processed_iris",
        "deepfakes/processed_signals"
    ]
    
    for folder in folders:
        input_dir = os.path.join(base_path, folder)
        output_dir = os.path.join(output_base, folder)
        process_directory(input_dir, output_dir)
    
    print("\n" + "=" * 50)
    print("Frame count unified to 300.")
    print(f"Path: {output_base}")
    print("=" * 50)
