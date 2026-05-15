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

- **Test Stage 2 (Action Recognition — Visual):** 
  ```bash
  python test_stage2.py
  ```
  This picks 3 validation clips of `tennis_serve` by default and saves two output files:
  - `stage2_test_tennis_serve.png` — a grid showing 8 evenly-spaced frames per clip with the 40 discovered keypoints overlaid. Each clip row is bordered in **green** if the model predicted correctly, **red** if wrong, and includes a Top-3 confidence bar chart on the right.
  - `stage2_anim_tennis_serve.gif` — an animated GIF cycling through all 32 frames so you can see the keypoints moving in real time.

  **To test a different action**, pass the `--action` flag with any of the 15 Penn Action class names:
  ```bash
  python test_stage2.py --action golf_swing
  python test_stage2.py --action pushup
  python test_stage2.py --action jumping_jacks
  ```

  You can also control how many clips are shown (default is 3):
  ```bash
  python test_stage2.py --action squat --clips 5
  ```

  Valid action names are: `baseball_pitch`, `baseball_swing`, `bench_press`, `bowl`, `clean_and_jerk`, `golf_swing`, `jump_rope`, `jumping_jacks`, `pullup`, `pushup`, `situp`, `squat`, `strum_guitar`, `tennis_forehand`, `tennis_serve`.

### Per-Class Accuracy Breakdown
Want to see exactly which actions the model is good or bad at?

```bash
python eval_per_class.py
```

This runs through the entire validation set and prints a table like:

```
  Class                     Top-1   Top-3   Total
  -------------------------------------------------------
  baseball_pitch            83.3%   95.8%      24
  tennis_serve              91.7%  100.0%      12
  pushup                    60.0%   80.0%      15
  ...
```

It also saves `eval_per_class.png` — a horizontal bar chart showing Top-1 and Top-3 accuracy for every class side by side, colour-coded green (≥70%), orange (≥40%), or red (<40%) so you can spot weak classes at a glance.

**Options:**

- Use a specific checkpoint instead of the latest one:
  ```bash
  python eval_per_class.py --ckpt checkpoints/stage2-epoch79.ckpt
  ```

- Skip saving the chart (just print the table):
  ```bash
  python eval_per_class.py --no-plot
  ```