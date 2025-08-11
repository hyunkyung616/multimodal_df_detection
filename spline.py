# This code is an original work authored by Hyunkyung Lee.
# It was developed as part of the MSc CyberSecurity individual project at King's College London.

import os
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline

def spline_interpolation_with_log(df, filename):

    df = df.sort_values('frame').reset_index(drop=True)
    
    frame_min = int(df['frame'].min())
    frame_max = int(df['frame'].max())
    full_frames = pd.DataFrame({'frame': list(range(frame_min, frame_max + 1))})
    
    merged = pd.merge(full_frames, df, on='frame', how='left')
    
    # Select numeric columns (excluding frame columns)
    value_cols = [col for col in merged.columns if col != 'frame' 
                 and pd.api.types.is_numeric_dtype(merged[col])]
    
    interpolated_frames = set()
    
    # Apply interpolation to each column
    for col in value_cols:

        valid_idx = merged[col].notna()
        
        # Identify missing sections
        missing_idxs = merged.index[~valid_idx]
        if len(missing_idxs) == 0:
            continue

        group_list = np.split(missing_idxs, np.where(np.diff(missing_idxs) != 1)[0] + 1)
        for group in group_list:
            if 1 <= len(group) <= 11:
                try:
                    cs = CubicSpline(
                        merged.loc[valid_idx, 'frame'].astype(int),
                        merged.loc[valid_idx, col]
                    )

                    # Applie spline interpolation
                    group_frames = merged.loc[group, 'frame'].astype(int)
                    merged.loc[group, col] = cs(group_frames)
                    interpolated_frames.update(group_frames.tolist())
                except Exception as e:
                    print(f"  Error ({filename}, {col}): {str(e)}")

                    merged[col] = merged[col].interpolate(method='linear')
                    merged[col] = merged[col].ffill().bfill()
    
    for col in value_cols:
        merged[col] = merged[col].interpolate(method='linear')
        merged[col] = merged[col].ffill().bfill()
    
    return merged, sorted([int(f) for f in interpolated_frames])

def process_directory(input_dir, output_dir):

    if not os.path.exists(input_dir):
        print(f"The input path does not exist: {input_dir}")
        return [], 0
    
    os.makedirs(output_dir, exist_ok=True)
    log_entries = []
    processed_count = 0
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            try:
                file_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)
                df = pd.read_csv(file_path)
                processed_df, interpolated_frames = spline_interpolation_with_log(df, filename)
                processed_df.to_csv(output_path, index=False)
                if interpolated_frames:
                    log_entry = f"{filename}: {len(interpolated_frames)} frame interpolated - {interpolated_frames}"
                else:
                    log_entry = f"{filename}: No interpolated frames"
                log_entries.append(log_entry)
                processed_count += 1
                print(f"{log_entry}")
            except Exception as e:
                error_entry = f"{filename}: Error - {str(e)}"
                log_entries.append(error_entry)
                print(f"⚠️ {error_entry}")
    
    return log_entries, processed_count

def main():
    # !Important!
    # The current path is an absolute path and is set to match my experimental environment.
    # Modify it as appropriate for your experimental environment.

    base_path = r'C:\Users\rinah\Desktop\Project\Data\faceforensics'
    input_dirs = [
        os.path.join(base_path, 'original', 'processed_signals'),
        os.path.join(base_path, 'deepfakes', 'processed_signals')
    ]
    output_base = os.path.join(base_path, 'interpolated_data')
    all_log_entries = []
    total_processed = 0
    
    for input_dir in input_dirs:
        rel_path = os.path.relpath(input_dir, base_path)
        output_dir = os.path.join(output_base, rel_path)
        print(f"\n🔁 처리 시작: {input_dir}")
        log_entries, processed_count = process_directory(input_dir, output_dir)
        if processed_count > 0:
            print(f"   └ {processed_count} file processed")
            all_log_entries.extend(log_entries)
            total_processed += processed_count
    
    log_path = os.path.join(output_base, 'interpolation_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=== Spline Interpolation ===\n")
        f.write(f"Total number of files processed: {total_processed}\n")
        f.write("=" * 50 + "\n")
        for entry in all_log_entries:
            f.write(entry + "\n")
    
    print("\n" + "=" * 50)
    print(f"Total {total_processed} files interpolated.")
    print(f"Path: {output_base}")
    print(f"Log: {log_path}")
    print("=" * 50)

if __name__ == "__main__":
    main()
