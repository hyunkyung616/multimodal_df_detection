# This code is an original work authored by Hyunkyung Lee.
# It was developed as part of the MSc CyberSecurity individual project at King's College London.

# This code was not directly used in the experiment.
# It was used as a demonstration for the final presentation.
# The logic and pipeline, including processing logic, signal extraction, and CSV storage,
# are identical to the code used in the experiment.

import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from scipy.stats import entropy
import mediapipe as mp

#ROI indexes
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
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    padlen = 3 * max(len(a), len(b))
    if len(data) <= padlen:
        print(f"[Warning] Signal too short for filtering (length={len(data)}, padlen={padlen}). Returning NaN array.")
        return np.full_like(data, np.nan)
    return filtfilt(b, a, data)

def extract_fft_features(signal, fs):
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

def plot_filtered_signals(signals, video_name):

    plt.figure(figsize=(8, 5))
    for i, (roi, values) in enumerate(signals.items(), 1):
        plt.subplot(len(signals), 1, i)
        plt.plot(values, label=f'{roi} Filtered')
        plt.xlabel('Frame Number')
        plt.ylabel('Amplitude')
        plt.title(f'{roi} Filtered PPG Signal from {video_name}')
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.show()

def extract_and_save_signals_with_visualization(video_path, output_folder, lowcut=0.7, highcut=4.0):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    cap = cv2.VideoCapture(video_path)
    fs = cap.get(cv2.CAP_PROP_FPS)

    if fs == 0:
        print(f"[Error] Invalid frame rate for {video_path}. Skipping.")
        cap.release()
        return None, []

    raw_signals = {roi: [] for roi in ROI_DICT.keys()}
    skipped_frames = []
    success_frame_indices = []

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True) as face_mesh:

        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            vis_frame = frame.copy()

            if not results.multi_face_landmarks:
                skipped_frames.append(frame_index)
            else:
                landmarks = results.multi_face_landmarks[0].landmark
                h, w = frame.shape[:2]

                for roi_name, indices in ROI_DICT.items():
                    points = [(int(lm.x * w), int(lm.y * h))
                              for lm in [landmarks[i] for i in indices]]
                    
                    cv2.polylines(vis_frame, [np.array(points)], isClosed=True, color=(0, 0, 255), thickness=2)

                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask, [np.array(points)], 255)
                    green = frame[:, :, 1]
                    mean_val = cv2.mean(green, mask=mask)[0]
                    raw_signals[roi_name].append(mean_val)

                success_frame_indices.append(frame_index)

            cv2.imshow('ROI Visualization', cv2.resize(vis_frame, (640, 360)))
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()

    filtered_signals = {}
    for roi, signal in raw_signals.items():
        filtered_signals[roi] = bandpass_filter(np.array(signal), lowcut, highcut, fs)

    fft_features = {}
    for roi, signal in filtered_signals.items():
        if np.all(np.isnan(signal)):
            fft_features[roi] = {k: np.nan for k in ['peak_freq', 'spectral_entropy', 'power_ratio', 'harmonic_std', 'phase_variance']}
        else:
            fft_features[roi] = extract_fft_features(signal, fs)

    if filtered_signals['Forehead'] is not None and len(filtered_signals['Forehead']) > 0 and not np.all(np.isnan(filtered_signals['Forehead'])):
        plot_filtered_signals(filtered_signals, video_name)
    else:
        print("Warning: No valid signals were extracted. Cannot plot.")
    
    os.makedirs(output_folder, exist_ok=True)
    
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

video_path = r"D:\[KCL]IndividualProject\faceforensics\original_sequences\youtube\raw\videos\555.mp4"
output_folder = r"D:\[KCL]IndividualProject\faceforensics\original_sequences\youtube\raw\videos\processed_signals"

print(f"Processing single video: {video_path}")
result_csv_path, skipped_frames_list = extract_and_save_signals_with_visualization(video_path, output_folder)

if result_csv_path:
    print(f"Processing complete. Data saved to: {result_csv_path}")
    if skipped_frames_list:
        print(f"Skipped frames: {skipped_frames_list}")
    else:
        print("No frames were skipped.")
else:
    print(f"Failed to process video: {video_path}")