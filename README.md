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

## 📂 2. Dataset Setup

This project uses the **Penn Action Dataset**.

### Download
Download the dataset from the official source:

**[https://www.cis.upenn.edu/~kostas/Penn_Action.tar.gz](https://www.cis.upenn.edu/~kostas/Penn_Action.tar.gz)**

Then extract it:
```bash
# Linux / macOS
tar -xzf Penn_Action.tar.gz

# Windows (PowerShell)
tar -xzf Penn_Action.tar.gz
```

### Folder Structure
Once extracted, place the dataset inside the project directory exactly like this:

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

## 💾 3. Pre-Trained Checkpoints (Skip Training)

Training from scratch takes several hours. If you just want to run evaluation or the visual tests, you can download our pre-trained checkpoints directly from Google Drive:

**[Download Checkpoints from Google Drive](https://drive.google.com/drive/folders/1XbhTptYVgSAu54H5Byvk9pgO7QAUuand?usp=sharing)**

The folder contains all saved checkpoints from both stages. You only need **two files** — one for each stage:

| Stage | File to download | What it is |
|-------|-----------------|------------|
| Stage 1 | `stage1-epoch49-loss0.4976.ckpt` | Best keypoint detector (lowest reconstruction loss) |
| Stage 2 | `stage2-epoch49-val_top10.4639.ckpt` | Best action classifier (~46% Top-1 accuracy) |

**Setup steps:**
1. Create a `checkpoints/` folder inside the project if it does not already exist:
   ```bash
   mkdir checkpoints
   ```
2. Download the two `.ckpt` files above from the Drive folder and place them inside `checkpoints/`:
   ```text
   FCV-Project/
   └── checkpoints/
       ├── stage1-epoch49-loss0.4976.ckpt
       └── stage2-epoch49-val_top10.4639.ckpt
   ```
3. That's it — all evaluation and testing scripts will automatically detect and use these checkpoints.

> You do **not** need to download the `last.ckpt` or any other files. Only the two files listed above are required.

---

## 🚀 4. How to Run the Pipeline

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

## 🧪 5. Testing & Evaluation

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