# Action Recognition via Unsupervised Keypoint Discovery (FCV Project)

Welcome to the FCV (Foundations of Computer Vision) project! 

This project implements a 2-Stage architecture for recognizing human actions in videos using **unsupervised keypoints**:
- **Stage 1 (Keypoint Discovery):** An unsupervised PyTorch Lightning module that learns to track moving body parts (keypoints) across video frames by attempting to reconstruct a target frame from a reference frame.
- **Stage 2 (Action Recognition):** A lightweight Transformer-based classifier that learns to predict the sporting action purely by looking at the sequential movements of the keypoints discovered in Stage 1.

## 🛠️ 1. Project Setup

Follow these steps to set up the project on your local machine.

### Prerequisites
- Python 3.10+
- A machine with an NVIDIA GPU (recommended for faster training) or CPU.

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Lala-Mumtaz-Ali/FCV-Project.git
   cd FCV-Project
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   
   # Windows:
   .\venv\Scripts\activate
   
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📂 2. Dataset Structure

This project uses the **Penn Action Dataset**. 
Before running the code, make sure your dataset is placed in the project directory exactly like this:

```text
FCV-Project/
└── Penn_Action/
    ├── frames/
    │   ├── 0001/
    │   │   ├── 000001.jpg
    │   │   ├── 000002.jpg
    │   ├── 0002/
    │   ...
    └── labels/
        ├── 0001.mat
        ├── 0002.mat
        ...
```
*(You can modify the `data_root` path in `config.py` if your dataset is located elsewhere).*

## 🚀 3. How to Run the Pipeline

The entire pipeline is controlled via the `main.py` script.

### Stage 1: Train the Keypoint Detector
Run the following command to train the unsupervised keypoint discovery model for 50 epochs.
```bash
python main.py --stage 1
```
*Checkpoints will be saved in the `./checkpoints/` directory.*

### Stage 2: Train the Action Classifier
Once Stage 1 is complete, run the following command to freeze the Keypoint Detector and train the Transformer Action Classifier for 30 epochs.
```bash
python main.py --stage 2
```

## 🧪 4. Testing & Evaluation

### Evaluate Overall Metrics
To evaluate the Final Mean Squared Error (MSE) of Stage 1 and the Top-1 Accuracy of Stage 2 on the validation set, run:
```bash
python evaluate.py
```

### Visual Testing
Want to physically *see* the discovered keypoints? 
- **Test Stage 1 (Frame Reconstruction):** 
  ```bash
  python test.py
  ```
  *(Saves `test_results.png` showing the source frame, target frame, and reconstructed frame)*

- **Test Stage 2 (Action Recognition):** 
  ```bash
  python test_stage2.py
  ```
  *(Saves `stage2_test_results.png` showing the overlaid keypoints across a video sequence and the final predicted action)*. You can open `test_stage2.py` and modify the `TARGET_ACTION` variable to test different sporting actions!