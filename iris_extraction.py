# This code is an original work authored by Hyunkyung Lee.
# It was developed as part of the MSc CyberSecurity individual project at King's College London.

import os
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd

skipped_frames_log = []

def build_gabor_bank(scales=4, orientations=6, ksize=31):

    # Build a bank of Gabor filters with different scales and orientations.
    filters = []
    for theta in np.linspace(0, np.pi, orientations, endpoint=False):
        for freq in np.linspace(0.1, 0.4, scales):
            sigma = 0.56 / freq
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta,
                                        1.0/freq, gamma=0.5, psi=0)
            filters.append(kernel)
    return filters

def apply_gabor(img_gray, gabor_bank):

    # Apply the Gabor filter bank to a grayscale image and return the max response for each filter.
    responses = []
    for kern in gabor_bank:
        resp = cv2.filter2D(img_gray, cv2.CV_32F, kern)
        responses.append(np.max(np.abs(resp)))
    return responses

def extract_iris_roi(frame, face_mesh):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    h, w = frame.shape[:2]
    if not results.multi_face_landmarks:
        return None
    
    lm = results.multi_face_landmarks[0].landmark
    pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in [474,475,476,477]]
    cx = sum(p[0] for p in pts) // 4
    cy = sum(p[1] for p in pts) // 4
    r = int(np.hypot(pts[0][0]-pts[2][0], pts[0][1]-pts[2][1]) / 2)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    iris = cv2.bitwise_and(frame, frame, mask=mask)[cy-r:cy+r, cx-r:cx+r]

    return iris

def extract_and_save_signals(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    gabor_bank = build_gabor_bank()

    columns = ['frame'] + [f'scale{si}_ori{oi}' for si in range(1,5) for oi in range(1,7)]
    df = pd.DataFrame(columns=columns)

    frame_idx = 0
    base_name = os.path.basename(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        iris_roi = extract_iris_roi(frame, mp_face_mesh)
        if iris_roi is not None and iris_roi.size > 0:
            gray = cv2.cvtColor(iris_roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            feats = apply_gabor(gray, gabor_bank)
            df.loc[frame_idx] = [frame_idx] + feats
        else:
            print(f"Frame {frame_idx}: iris ROI not found, skipping.")

            # Log the skipped frame's video name and frame index.
            skipped_frames_log.append((base_name, frame_idx))
        frame_idx += 1

    cap.release()
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_folder, exist_ok=True)
    csv_path = os.path.join(output_folder, f"{base_name}.csv")
    df.to_csv(csv_path, index=False)
    return csv_path

def batch_process_videos(video_folder, output_folder, video_exts=('.mp4','.avi','.mov','.mkv')):
    os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(video_folder):
        if file_name.lower().endswith(video_exts):
            video_path = os.path.join(video_folder, file_name)
            print(f"Processing: {file_name}")
            result = extract_and_save_signals(video_path, output_folder)
            if result:
                print(f"Saved: {result}")
            else:
                print(f"Skipped: {file_name}")

    # Save skipped frames log
    if skipped_frames_log:
        log_path = os.path.join(output_folder, "skipped_frames_log.txt")
        with open(log_path, 'w', encoding='utf-8') as f:
            for video_name, frame_num in skipped_frames_log:
                f.write(f"{video_name}, frame {frame_num}\n")
        print(f"Skipped frame log saved to: {log_path}")
    else:
        print("No skipped frames detected.")

# !Important!
# The current path is an absolute path and is set to match my experimental environment.
# Modify it as appropriate for your experimental environment.

if __name__ == '__main__':
    video_folder = r"D:\[KCL]IndividualProject\faceforensics\original_sequences\youtube\raw\videos"
    output_folder = os.path.join(video_folder, "processed_iris")
    batch_process_videos(video_folder, output_folder)
