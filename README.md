# I verify that I am the sole author of the programs contained in this archive, except where explicitly stated to the contrary.
# Hyunkyung Lee, 06 August 2025

# 1. faceforensics_download.py

2. Signal extraction
	1) rppg_extraction.py	: Code for extracting rppg signals and FFT features.
	2) iris_extraction.py	: Code for extracting iris pattern features.

3. Preprocessing
	1) spline.py 			: Handles interpolation of missing frames.
	2) common_frame.py 		: Extracts common frame from file pairs.
	3) resampling.py 		: Normalizes length to 300 frame.
	4) rppg_seperate 		: Separates rPPG signals and FFT features.

	*All scripts must be run after modifying the path.
	*Folders are created at each step, so adjust the paths accordingly to reflect this in your environment.

4. split.py
	Split data structure according to train.json, val.json, and test.json in the same folder.
	If the split result is train: 710, test: 137, val: 136, the split is successful.
	Use the following files within the folder:
		- train.json
		- test.json
		- val.json

5. Model training
	1) rppg_only.py				: Code for the rPPG-only model.
	2) rppg_iris_multimodal.py	: Code for the rPPG + Iris multimodal model.
	
6. Cross validation
	1) rppg_only_5_fold.py				: Code for the 5-fold cross validation for rPPG-only model.
	2) rppg_iris_multimodal_5_fold.py	: Code for the 5-fold cross validation for rPPG + Iris multimodal model.
	
* Demo script
	rppg_demo.py, iris_demo.py
	* These codes were not directly used in the experiments; they were written for demonstration purposes in the final presentation.
	* The logic and pipeline, including ROI setting and signal extraction, are identical to rppg_extraction.py and iris_extraction.py, respectively.