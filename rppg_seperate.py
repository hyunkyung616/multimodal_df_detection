# This code is an original work authored by Hyunkyung Lee.
# It was developed as part of the MSc CyberSecurity individual project at King's College London.

# A function that separates and stores the PPG signal and its derivative features.
# To separate the rPPG signal of time series features and the FFT data of independent features.

import os
import pandas as pd
import shutil

def separate_features(input_folder, output_base):
    
    derived_output = os.path.join(output_base, 'derived_features', 'original', 'processed_signals')
    os.makedirs(derived_output, exist_ok=True)
    
    signal_output = os.path.join(output_base, 'original_signals', 'original', 'processed_signals')
    os.makedirs(signal_output, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            input_path = os.path.join(input_folder, filename)
            
            df = pd.read_csv(input_path)
            
            # Separate derived features
            derived_cols = [col for col in df.columns 
                           if not col.endswith('_filtered') and col != 'frame']
            derived_df = df[['frame'] + derived_cols]
            derived_path = os.path.join(derived_output, filename)
            derived_df.to_csv(derived_path, index=False)
            
            signal_cols = ['frame'] + [col for col in df.columns if col.endswith('_filtered')]
            signal_df = df[signal_cols]
            signal_path = os.path.join(signal_output, filename)
            signal_df.to_csv(signal_path, index=False)
            
            print(f"Processed: {filename}")

# !Important!
# The current path is an absolute path and is set to match my experimental environment.
# Modify it as appropriate for your experimental environment.
# Basically, when you run common_frame.py, the files are saved in the resampled_300 folder.

input_folder = r"C:\Users\rinah\Desktop\Project\Data\faceforensics\interpolated_data\resampled_300\original\processed_signals"
output_base = r"C:\Users\rinah\Desktop\Project\Data\data_final"

separate_features(input_folder, output_base)
print("\n" + "="*50)
print("Complete")
print(f"FFT Path: {os.path.join(output_base, 'derived_features')}")
print(f"rPPG Path: {os.path.join(output_base, 'original_signals')}")
print("="*50)
