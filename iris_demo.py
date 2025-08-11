# This code is an original work authored by Hyunkyung Lee.
# It was developed as part of the MSc CyberSecurity individual project at King's College London.

# This code was not directly used in the experiment.
# It was used as a demonstration for the final presentation.
# The logic and pipeline, including processing logic, signal extraction, and CSV storage,
# are identical to the code used in the experiment.

import os
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd

skipped_frames_log = []

def build_gabor_bank(scales=4, orientations=6, ksize=31):
    filters = []
    for theta in np.linspace(0, np.pi, orientations, endpoint=False):
        for freq in np.linspace(0.1, 0.4, scales):
            sigma = 0.56 / freq
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta,
                                         1.0/freq, gamma=0.5, psi=0)
            filters.append(kernel)
    return filters

def apply_gabor(img_gray, gabor_bank):
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
        return None, None
    lm = results.multi_face_landmarks[0].landmark

    pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in [474,475,476,477]]
    cx = sum(p[0] for p in pts) // 4
    cy = sum(p[1] for p in pts) // 4
    r = int(np.hypot(pts[0][0]-pts[2][0], pts[0][1]-pts[2][1]) / 2)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    iris = cv2.bitwise_and(frame, frame, mask=mask)[cy-r:cy+r, cx-r:cx+r]
    return iris, (cx, cy, r)


def process_single_video_with_visualization(video_path, output_folder):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                      max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5)
    gabor_bank = build_gabor_bank()

    columns = ['frame'] + [f'gabor_feat_{i}' for i in range(len(gabor_bank))]
    df = pd.DataFrame(columns=columns)

    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Iris ROI', cv2.WINDOW_NORMAL)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vis_frame = frame.copy()

        iris_roi, circle_info = extract_iris_roi(frame, face_mesh)
        
        if iris_roi is not None and iris_roi.size > 0:

            gray_iris = cv2.cvtColor(iris_roi, cv2.COLOR_BGR2GRAY)
            gray_iris = cv2.equalizeHist(gray_iris)

            gabor_responses = apply_gabor(gray_iris, gabor_bank)

            df.loc[frame_idx] = [frame_idx] + gabor_responses

            cx, cy, r = circle_info
            cv2.circle(vis_frame, (cx, cy), r, (0, 255, 0), 2)

            gabor_visual = np.zeros_like(gray_iris, dtype=np.float32)
            for kern in gabor_bank:
                resp = cv2.filter2D(gray_iris, cv2.CV_32F, kern)
                gabor_visual = np.maximum(gabor_visual, np.abs(resp))
            
            gabor_norm = cv2.normalize(gabor_visual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            cv2.imshow('Iris ROI', cv2.resize(iris_roi, (300, 300)))
            cv2.imshow('Gabor Response', cv2.resize(gabor_norm, (300, 300)))

        else:
            print(f"Frame {frame_idx}: Iris ROI not found, skipping.")
            
        cv2.imshow('Frame', vis_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
            
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    os.makedirs(output_folder, exist_ok=True)
    csv_path = os.path.join(output_folder, f"{base_name}.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"Processing complete. Data saved to: {csv_path}")
    print(f"Total frames processed: {frame_idx}")
    return csv_path

if __name__ == '__main__':

    video_path = r"D:\[KCL]IndividualProject\faceforensics\original_sequences\youtube\raw\videos\555.mp4"
    output_folder = r"D:\[KCL]IndividualProject\faceforensics\original_sequences\youtube\raw\videos\processed_iris"
    
    process_single_video_with_visualization(video_path, output_folder)