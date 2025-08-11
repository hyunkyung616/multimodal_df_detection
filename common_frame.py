# This code is an original work authored by Hyunkyung Lee.
# It was developed as part of the MSc CyberSecurity individual project at King's College London.

import os
import pandas as pd

# !Important!
# The current path is an absolute path and is set to match my experimental environment.
# Modify it as appropriate for your experimental environment.
# Basically, when you run spline.py, the interpolated files are saved in the interpolated_data folder.

base_path = r'C:\Users\rinah\Desktop\Project\Data\faceforensics\interpolated_data'

original_path = os.path.join(base_path, 'original')
deepfakes_path = os.path.join(base_path, 'deepfakes')

original_iris_path = os.path.join(original_path, 'processed_iris')
original_signal_path = os.path.join(original_path, 'processed_signals')
deepfakes_iris_path = os.path.join(deepfakes_path, 'processed_iris')
deepfakes_signal_path = os.path.join(deepfakes_path, 'processed_signals')

# Common frame processing result folder
new_base_path = os.path.join(base_path, 'processed_common_frames')
os.makedirs(new_base_path, exist_ok=True)
for category in ['original', 'deepfakes']:
    for folder_type in ['processed_iris', 'processed_signals']:
        os.makedirs(os.path.join(new_base_path, category, folder_type), exist_ok=True)

log_entries = []

original_iris_files = []
if os.path.exists(original_iris_path):
    original_iris_files = [f for f in os.listdir(original_iris_path) if f.endswith('.csv')]

for orig_file in original_iris_files:
    base = orig_file.replace('.csv', '')
    orig_iris_file_path = os.path.join(original_iris_path, orig_file)
    orig_signal_file_path = os.path.join(original_signal_path, orig_file)

    deepfakes_iris_files = []
    if os.path.exists(deepfakes_iris_path):
        deepfakes_iris_files = [f for f in os.listdir(deepfakes_iris_path)
                                if f.startswith(f"{base}_") and f.endswith('.csv')]
    for df_iris_file in deepfakes_iris_files:
        df_signal_file = df_iris_file
        df_iris_file_path = os.path.join(deepfakes_iris_path, df_iris_file)
        df_signal_file_path = os.path.join(deepfakes_signal_path, df_signal_file)

        all_files_exist = all(os.path.exists(p) for p in [
            orig_iris_file_path, orig_signal_file_path, df_iris_file_path, df_signal_file_path
        ])
        if not all_files_exist:
            missing_files = [p for p in [
                orig_iris_file_path, orig_signal_file_path, df_iris_file_path, df_signal_file_path
            ] if not os.path.exists(p)]
            print(f"File Missing: {', '.join(missing_files)}")
            continue

        try:
            orig_iris = pd.read_csv(orig_iris_file_path)
            orig_signal = pd.read_csv(orig_signal_file_path)
            df_iris = pd.read_csv(df_iris_file_path)
            df_signal = pd.read_csv(df_signal_file_path)

            # Calculating common frames for four files
            frames = [
                set(orig_iris['frame']),
                set(orig_signal['frame']),
                set(df_iris['frame']),
                set(df_signal['frame'])
            ]
            common_frames = set.intersection(*frames)
            if not common_frames:
                log_entries.append(f"{orig_file} <-> {df_iris_file}: No common frame (skip)")
                print(f"{orig_file} <-> {df_iris_file}: No common frame")
                continue

            # Common frame filtering
            orig_iris_common = orig_iris[orig_iris['frame'].isin(common_frames)].copy()
            orig_signal_common = orig_signal[orig_signal['frame'].isin(common_frames)].copy()
            df_iris_common = df_iris[df_iris['frame'].isin(common_frames)].copy()
            df_signal_common = df_signal[df_signal['frame'].isin(common_frames)].copy()

            # Reset frame number (starting from 1)
            for df in [orig_iris_common, orig_signal_common, df_iris_common, df_signal_common]:
                df['frame'] = range(1, len(df) + 1)

            orig_iris_common.to_csv(os.path.join(new_base_path, 'original', 'processed_iris', orig_file), index=False)
            orig_signal_common.to_csv(os.path.join(new_base_path, 'original', 'processed_signals', orig_file), index=False)
            df_iris_common.to_csv(os.path.join(new_base_path, 'deepfakes', 'processed_iris', df_iris_file), index=False)
            df_signal_common.to_csv(os.path.join(new_base_path, 'deepfakes', 'processed_signals', df_signal_file), index=False)

            min_frame = min(common_frames)
            max_frame = max(common_frames)
            frame_count = len(common_frames)
            log_entry = f"{orig_file} <-> {df_iris_file}: Frame {min_frame}~{max_frame} (Total {frame_count})"
            log_entries.append(log_entry)
            print(f"{log_entry}")

        except Exception as e:
            error_msg = f"{orig_file} <-> {df_iris_file}: Error - {str(e)}"
            log_entries.append(error_msg)
            print(f"{error_msg}")

# Save log file
log_path = os.path.join(new_base_path, 'common_frame_log.txt')
with open(log_path, 'w', encoding='utf-8') as f:
    f.write("=== Common frame processing ===\n")
    f.write(f"Processed file pairs: {len(log_entries)}\n")
    f.write("=" * 50 + "\n")
    for entry in log_entries:
        f.write(entry + "\n")

print(f"Total {len(log_entries)} file pairs processed.")
print(f"Log: {log_path}")
print(f"Path: {new_base_path}")
