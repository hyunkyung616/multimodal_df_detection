# PPG and Iris Pattern Feature Analysis for Deepfake Video Detection

## 📃 Author's Declaration

''' I verify that I am the sole author of the programs contained in this archive, except where explicitly stated to the contrary. Hyunkyung Lee, 06 August 2025 '''

## 📝 Project Script Descriptions

### **Signal Extraction**
* `rppg_extraction.py`: Code for extracting rPPG signals and FFT features.
* `iris_extraction.py`: Code for extracting iris pattern features.

### **Preprocessing**
* `spline.py`: Handles interpolation of missing frames.
* `common_frame.py`: Extracts common frames from file pairs.
* `resampling.py`: Normalizes video length to 300 frames.
* `rppg_seperate.py`: Separates rPPG signals and FFT features.

### **Model Training**
* `rppg_only.py`: Code for the rPPG-only model.
* `rppg_iris_multimodal.py`: Code for the rPPG + Iris multimodal model.

### **Cross-validation**
* `rppg_only_5_fold.py`: Code for 5-fold cross-validation for the rPPG-only model.
* `rppg_iris_multimodal_5_fold.py`: Code for 5-fold cross-validation for the rPPG + Iris multimodal model.

### **Demo Scripts**
* `rppg_demo.py`, `iris_demo.py`: These codes were not used in the main experiments; they were written for demonstration purposes in the final presentation. The logic and pipeline, including ROI setting and signal extraction, are identical to `rppg_extraction.py` and `iris_extraction.py`, respectively.

---

### **⚠️ Important Notes Before Execution**

* **All scripts must be run after modifying the file paths.**
* As folders are created at each step, you must adjust the paths accordingly in your environment.
* To run `split.py`, `train.json`, `test.json`, and `val.json` must be in the same folder. The split is successful if the result is train: 710, test: 137, val: 136.