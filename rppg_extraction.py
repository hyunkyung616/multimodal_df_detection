# This code is an original work authored by Hyunkyung Lee.
# It was developed as part of the MSc CyberSecurity individual project at King's College London.

import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from scipy.stats import entropy
import mediapipe as mp

# ROI landmarks indexed (Mediapipe Facemash)
FOREHEAD = [9, 108, 10, 337]
LEFT_CHEEK = [123, 117, 101, 205]
RIGHT_CHEEK = [352, 346, 330, 425]
NOSE = [4, 275, 248, 195, 3, 51, 45]

ROI_DICT = {
    'Forehead': FOREHEAD,
    'Left_Cheek': LEFT_CHEEK,
    'Right_Cheek': RIGHT_CHEEK,
    'Nose': NOSE
}

mp_face_mesh = mp.solutions.face_mesh

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):

    # Apply the band-pass filtering to the input signal
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    padlen = 3 * max(len(a), len(b))
    if len(data) <= padlen:
        print(f"[Warning] Signal too short for filtering (length={len(data)}, padlen={padlen}). Returning NaN array.")
        return np.full_like(data, np.nan)
    return filtfilt(b, a, data)

def extract_fft_features(signal, fs):

    #Extract FFT-based features from the filtered signal
    n = len(signal)
    fft_vals = np.abs(fft(signal))
    freqs = fftfreq(n, 1/fs)[:n//2]
    fft_vals = fft_vals[:n//2]

    if len(fft_vals) == 0 or np.all(np.isnan(signal)):
        return {k: np.nan for k in ['peak_freq', 'spectral_entropy', 'power_ratio', 'harmonic_std', 'phase_variance']}

    features = {
        'peak_freq': freqs[np.argmax(fft_vals)],
        'spectral_entropy': entropy(fft_vals + 1e-10),
        'power_ratio': np.sum(fft_vals[1:5])/np.sum(fft_vals) if np.sum(fft_vals) > 0 else 0,
        'harmonic_std': np.std(fft_vals[::5]),
        'phase_variance': np.var(np.angle(fft(signal)))
    }
    return features

def plot_filtered_signals(signals, video_name, output_folder):
    plt.figure(figsize=(15, 10))
    for i, (roi, values) in enumerate(signals.items(), 1):
        plt.subplot(len(signals), 1, i)
        plt.plot(values, label=f'{roi} Filtered')
        plt.xlabel('Frame Number')
        plt.ylabel('Amplitude')
        plt.title(f'{roi} Filtered PPG Signal')
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plot_dir = os.path.join(output_folder, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f'{video_name}_signals.png')
    plt.savefig(plot_path)
    plt.close()

def extract_and_save_signals(video_path, output_folder, lowcut=0.7, highcut=4.0):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    fs = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fs == 0:
        print(f"[Error] Invalid frame rate for {video_path}. Skipping.")
        return None, []

    raw_signals = {roi: [] for roi in ROI_DICT.keys()}
    skipped_frames = []  # Store skipped frame numbers
    success_frame_indices = []

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True) as face_mesh:

        cap = cv2.VideoCapture(video_path)
        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if not results.multi_face_landmarks:
                skipped_frames.append(frame_index)
                frame_index += 1
                continue

            landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]

            for roi_name, indices in ROI_DICT.items():
                points = [(int(lm.x * w), int(lm.y * h))
                          for lm in [landmarks[i] for i in indices]]
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(points)], 255)
                green = frame[:, :, 1]
                mean_val = cv2.mean(green, mask=mask)[0]
                raw_signals[roi_name].append(mean_val)
                
            success_frame_indices.append(frame_index)
            frame_index += 1

        cap.release()

    filtered_signals = {}
    for roi, signal in raw_signals.items():
        filtered_signals[roi] = bandpass_filter(signal, lowcut, highcut, fs)

    fft_features = {}
    for roi, signal in filtered_signals.items():
        if np.all(np.isnan(signal)):
            fft_features[roi] = {k: np.nan for k in ['peak_freq', 'spectral_entropy', 'power_ratio', 'harmonic_std', 'phase_variance']}
        else:
            fft_features[roi] = extract_fft_features(signal, fs)

    os.makedirs(output_folder, exist_ok=True)
    plot_filtered_signals(filtered_signals, video_name, output_folder)

    df = pd.DataFrame()
    df['frame'] = success_frame_indices

    for roi in ROI_DICT.keys():
        df[f'{roi}_filtered'] = filtered_signals[roi]

    for roi in ROI_DICT.keys():
        for feat_name, feat_val in fft_features[roi].items():
            df[f'{roi}_{feat_name}'] = [feat_val] * len(df)

    csv_path = os.path.join(output_folder, f"{video_name}.csv")
    df.to_csv(csv_path, index=False)
    return csv_path, skipped_frames

# !Important!
# The current path is an absolute path and is set to match my experimental environment.
# Modify it as appropriate for your experimental environment.

video_folder = r"D:\[KCL]IndividualProject\faceforensics\original_sequences\youtube\raw\videos"
output_folder = os.path.join(video_folder, "processed_signals")
video_exts = ('.mp4', '.avi', '.mov', '.mkv')

# For saving the entire skip frame log
total_skipped_frames_log = []

for file in os.listdir(video_folder):
    if file.lower().endswith(video_exts):
        video_path = os.path.join(video_folder, file)
        print(f"Processing: {file}")
        result, skipped_frames = extract_and_save_signals(video_path, output_folder)
        
        if result:
            print(f"Saved: {result}")
            # Add to global log
            video_name = os.path.splitext(file)[0]
            total_skipped_frames_log.extend(
                [(video_name, frame_num) for frame_num in skipped_frames]
            )
        else:
            print(f"Skipped: {file}")

# After all video processing, save as a skip frame log file.
if total_skipped_frames_log:
    log_path = os.path.join(output_folder, "skipped_frames_log.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        for video_name, frame_num in total_skipped_frames_log:
            f.write(f"{video_name}, frame {frame_num}\n")
    print(f"Skipped frame log saved to: {log_path}")
else:
    print("No skipped frames detected.")
