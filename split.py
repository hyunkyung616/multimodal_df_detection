# This code is an original work authored by Hyunkyung Lee.
# It was developed as part of the MSc CyberSecurity individual project at King's College London.
# This code is written for the Google Colab environment.

from google.colab import files
import os
import json
import shutil

# 1. JSON file upload (train.json, val.json, test.json)"
json_uploaded = files.upload()

# !Important!
# The current path is an absolute path and is set to match my experimental environment.
# Modify it as appropriate for your experimental environment.

data_base = '/content/data_final'
split_base = '/content/split_data'

for split_name in ['train', 'val', 'test']:
    for category in ['original', 'deepfakes']:
        for data_type in ['processed_signals', 'derived_features', 'processed_iris']:
            dir_path = os.path.join(split_base, split_name, category, data_type)
            os.makedirs(dir_path, exist_ok=True)

print("Split folder structure creation")

# Data split function
def split_data_by_json(json_file, split_name):

    with open(json_file, 'r') as f:
        pairs = json.load(f)

    copied_files = 0
    missing_files = []

    for a, b in pairs:
        for data_type in ['processed_signals', 'derived_features', 'processed_iris']:

            src_orig = os.path.join(data_base, 'original', data_type, f'{a}.csv')
            dst_orig = os.path.join(split_base, split_name, 'original', data_type, f'{a}.csv')

            if os.path.exists(src_orig):
                shutil.copy(src_orig, dst_orig)
                copied_files += 1
            else:
                missing_files.append(f'original/{data_type}/{a}.csv')

            src_df = os.path.join(data_base, 'deepfakes', data_type, f'{a}_{b}.csv')
            dst_df = os.path.join(split_base, split_name, 'deepfakes', data_type, f'{a}_{b}.csv')

            if os.path.exists(src_df):
                shutil.copy(src_df, dst_df)
                copied_files += 1
            else:
                missing_files.append(f'deepfakes/{data_type}/{a}_{b}.csv')

    print(f"{split_name} Split completed: {copied_files} files copied")
    if missing_files:
        print(f"Missing files: {len(missing_files)}")
        for missing in missing_files[:5]:
            print(f"  - {missing}")

if 'train.json' in os.listdir('/content'):
    split_data_by_json('train.json', 'train')

if 'val.json' in os.listdir('/content'):
    split_data_by_json('val.json', 'val')

if 'test.json' in os.listdir('/content'):
    split_data_by_json('test.json', 'test')

print("\nFull split complete.")
