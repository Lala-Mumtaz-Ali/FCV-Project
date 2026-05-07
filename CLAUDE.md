# FCV Project — CLAUDE.md

## Project Goal

Two-stage unsupervised keypoint-based action recognition on the **Penn Action** dataset (15 classes, ~2093 videos).

- **Stage 1** — Discover K=40 keypoints by training a detector + image translator end-to-end with a GAN loss. No pose annotations are used.
- **Stage 2** — Freeze the detector, train a Transformer classifier on keypoint sequences to recognise actions.

The pipeline is inspired by transporter-style unsupervised keypoint discovery: the detector must find keypoints that are actually useful for reconstructing a target frame from a reference frame, forcing them to be semantically meaningful.

---

## Repository Layout

```
FCV-Project/
├── config.py              # All hyperparameters — single source of truth
├── main.py                # Entry point: python main.py --stage 1|2
├── train.py               # Stage1Module, Stage2Module (PyTorch Lightning)
├── data.py                # VideoFramePairDataset, VideoSequenceDataset
├── loss.py                # CombinedLoss, PatchDiscriminator
├── utils.py               # keypoints_to_gaussian_maps
├── evaluate.py            # Full evaluation: python evaluate.py --stage 1|2
├── test.py                # Quick visual sanity check for Stage 1
├── models/
│   ├── detector.py        # KeypointDetector (ViT + heatmap head)
│   ├── translator.py      # KeypointGuidedTranslator (diffusion UNet)
│   ├── classifier.py      # ActionClassifier (Transformer)
│   └── generator.py       # TransformerCVAE (future use — not in current training)
├── checkpoints/           # Saved .ckpt files (auto-managed by Lightning)
├── logs/                  # TensorBoard logs
└── dataset/Penn_Action/
    ├── frames/<vid_id>/   # JPEG frames per video
    └── labels/<vid_id>.mat
```

---

## Architecture

### Stage 1 Models

#### KeypointDetector (`models/detector.py`)
- **Backbone**: `vit_small_patch16_224` from timm, pretrained. Frozen for the first 5 epochs, then fully unfrozen.
- **Heatmap head**: `LayerNorm → Linear(feat_dim→256) → GELU → Linear(256→K)`
- **Spatial softmax**: patch tokens reshaped to a spatial grid, upsampled to full image size, then softmax over spatial dims gives a probability map per keypoint.
- **Expected coordinates** (equation 1): weighted sum of grid coordinates with the probability map → `(B, K, 2)` in `[-1, 1]`.
- Output: `keypoints (B, K, 2)`, `soft_maps (B, K, H, W)`

#### KeypointGuidedTranslator (`models/translator.py`)
- **Backbone**: `UNet2DModel` from Hugging Face diffusers (used as a standard UNet, not for diffusion — timestep is always zeros).
- **Input**: `[ref_img (3) | ref_kp_maps (K) | tgt_kp_maps (K)]` = `3 + 2K` channels.
- **Output**: 4 channels split into `synth (3)` via tanh + `mask (1)` via sigmoid.
- **Blending** (equation 3): `pred = mask * ref_img + (1 - mask) * synth`
- Architecture: `(64, 128, 256, 256)` channels, attention in the 3 deeper blocks.

#### PatchDiscriminator (`loss.py`)
- 4-layer PatchGAN with `InstanceNorm2d` (more stable than BatchNorm for GANs).
- Hinge discriminator loss. Generator uses non-saturating loss.

#### CombinedLoss (`loss.py`)
```
recon_loss = lambda_lpips * LPIPS(pred, tgt) + lambda_l1 * L1(pred, tgt)
adv_loss   = lambda_adv  * softplus(-D(pred))
g_loss     = recon_loss + adv_loss
```
- LPIPS uses AlexNet backbone, weights are frozen (it is a fixed metric, not trained).
- KL loss method exists but is only used by the CVAE (generator.py), not in Stage 1 training.

#### Gaussian Maps (`utils.py`)
- `keypoints_to_gaussian_maps(keypoints, H, W, sigma)` — converts `(B, K, 2)` coordinates to `(B, K, H, W)` Gaussian heatmaps (equation 2).
- `sigma` controls the width. **Current value: 0.15** (increased from 0.1 to prevent keypoint collapse).

---

### Stage 2 Models

#### ActionClassifier (`models/classifier.py`)
- **Input projection**: flattens `K×2` coords → `d_model=256`.
- **Learnable CLS token** prepended to the sequence (ViT/BERT-style).
- **Positional embedding**: learned, supports up to 101 positions (100 frames + CLS).
- **Transformer encoder**: 4 layers, 8 heads, pre-LN (`norm_first=True`), GELU, dropout=0.1.
- **Classification head**: `LayerNorm → Linear(d_model→d_model//2) → GELU → Dropout → Linear(→num_actions)`.
- Only the CLS token output is used for classification.

#### TransformerCVAE (`models/generator.py`)
- Conditional VAE for generating future keypoint sequences from an initial keypoint + action label.
- **Not currently used** in Stage 1 or Stage 2 training — reserved for future video generation work.

---

## Dataset

**Penn Action** — 15 action classes:
```
baseball_pitch, baseball_swing, bench_press, bowl, clean_and_jerk,
golf_swing, jump_rope, jumping_jacks, pullup, pushup, situp,
squat, strum_guitar, tennis_forehand, tennis_serve
```

**VideoFramePairDataset** (Stage 1):
- Samples pairs `(ref, tgt)` from the same video with a random temporal gap of 1–8 frames.
- Per-class balancing via oversampling minority classes (max 20 pairs per video).
- Augmentation: random horizontal flip, color jitter, resize to `img_size=64`.
- 90/10 train/val split (shuffled with seed 42).

**VideoSequenceDataset** (Stage 2):
- Returns fixed-length sequences (`seq_len=32` frames) from each video.
- Training: random temporal window start. Validation: centre window (reproducible).
- Short videos padded by looping. Augmentation applied consistently across all frames in a clip.
- `clips_per_video=3` multiplies effective dataset size during training.

---

## Hyperparameters (`config.py`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `K` | 40 | Number of keypoints |
| `img_size` | 64 | Input image resolution |
| `sigma` | **0.15** | Gaussian map width — increased from 0.1 to fix keypoint collapse |
| `batch_size` | 32 | Fits on RTX 4090 24 GB |
| `num_workers` | 4 | Reduced from 8 for Windows stability |
| `lr_g` | 1e-4 | Generator + detector LR (Stage 1) |
| `lr_d` | 1e-4 | Discriminator LR (Stage 1) |
| `lambda_l1` | 10.0 | L1 pixel loss weight |
| `lambda_lpips` | 1.0 | Perceptual loss weight |
| `lambda_adv` | **0.3** | Adversarial loss weight — increased from 0.1 to penalise reference-copy shortcut |
| `lambda_kl` | 0.001 | KL loss weight (CVAE only) |
| `max_epochs_s1` | 70 | Stage 1 epochs — extended from 50 to continue after sigma/adv fix |
| `max_epochs_s2` | 80 | Stage 2 epochs |
| `warmup_epochs_s2` | 10 | Epochs before partial ViT unfreeze in Stage 2 |
| `seq_len` | 32 | Frames per clip (Stage 2) |
| `d_model` | 256 | Transformer hidden size |
| `nhead` | 8 | Attention heads |
| `num_layers` | 4 | Transformer depth |
| `num_actions` | 15 | Penn Action classes |
| `vit_model` | vit_small_patch16_224 | Backbone |

---

## Training

```bash
# Stage 1 — keypoint discovery (auto-resumes from latest stage1 checkpoint)
python main.py --stage 1

# Stage 2 — action recognition (requires a stage1 checkpoint)
python main.py --stage 2
```

**Stage 1 schedule:**
1. Epochs 0–4: backbone frozen, only heatmap head + translator + discriminator train.
2. Epoch 5+: full backbone unfrozen (`unfreeze_backbone()` called in `on_train_epoch_start`).
3. Manual optimisation (`automatic_optimization=False`) — generator and discriminator updated separately per step.
4. Gradient clipping: `clip_val=1.0`, norm-based.
5. Precision: `bf16-mixed`.
6. Checkpoints: every 5 epochs, top-3 by `s1/g_loss_epoch` + `last.ckpt`.

**Stage 2 schedule:**
1. Epochs 0–9 (warmup): detector fully frozen, only classifier trains. LR ramps linearly from 0.1× to 1× over these 10 epochs.
2. Epoch 10+: last 2 ViT blocks + heatmap head unfrozen, added to optimiser with `lr=1e-5`. Cosine annealing runs for the remaining epochs.
3. Validation step runs every epoch — logs `s2/val_loss`, `s2/val_top1`, `s2/val_top3`.
4. Checkpoints monitored on `s2/val_top1` (mode=max), saved every 5 epochs, all kept (`save_top_k=-1`).
5. Precision: `16-mixed`.
6. Logs `s2/loss`, `s2/top1`, `s2/top3` (train) and `s2/val_*` (val) per epoch.

---

## Evaluation

```bash
python evaluate.py --stage 1   # reconstruction metrics + visual grid
python evaluate.py --stage 2   # classification accuracy + confusion matrix

# Evaluate a specific checkpoint
python evaluate.py --stage 1 --ckpt checkpoints/stage1-epoch49-loss0.4976.ckpt
```

**Stage 1 outputs:**
- Console: MSE, PSNR (dB), SSIM, LPIPS, KP Spread, KP Motion + diagnosis
- `eval_stage1_visuals.png` — 8-sample grid: reference+keypoints | target+keypoints | reconstruction

**Stage 2 outputs:**
- Console: Top-1, Top-3 accuracy, per-class accuracy table, confidence analysis
- `eval_stage2_confusion.png` — row-normalised confusion matrix
- `eval_stage2_per_class.png` — horizontal accuracy bar chart

**Current Stage 1 results (epoch 49, before sigma/adv fixes):**

| Metric | Value | Status |
|--------|-------|--------|
| PSNR | 25.7 dB | Good |
| SSIM | 0.910 | Good |
| LPIPS | 0.030 | Good |
| KP Spread | 0.027 | Poor — keypoints collapsing |
| KP Motion | 0.023 | Poor — not tracking motion |

Reconstruction quality is good but keypoints collapse to one region. Fixes applied: `sigma 0.1→0.15`, `lambda_adv 0.1→0.3`. Continue training from epoch 49 checkpoint to evaluate improvement.

---

## Known Issues & Fixes Applied

### Windows CUDA "unknown error" at epoch boundaries
- **Cause**: DataLoader workers killed and restarted between epochs fail to re-acquire CUDA context.
- **Fix**: `persistent_workers=True` in `main.py` DataLoader, `num_workers` reduced from 8 to 4.

### Keypoint collapse (KP Spread ~0.027, should be >0.30)
- **Cause**: `sigma=0.1` produces very narrow Gaussian maps; overlapping keypoints merge into one blob so the translator ignores their positions and copies the reference instead.
- **Fix**: `sigma` increased to `0.15`, `lambda_adv` increased to `0.3`.

### test.py loaded wrong checkpoint
- **Fix**: Now filters for `"stage1"` in filename before selecting latest checkpoint.

### evaluate.py UnicodeEncodeError on Windows
- **Cause**: Windows PowerShell defaults to cp1252 which cannot print box-drawing characters.
- **Fix**: `sys.stdout.reconfigure(encoding='utf-8')` at the top of `evaluate.py`.

---

## Environment

- **OS**: Windows 11 Pro
- **Shell**: PowerShell (use PowerShell syntax in commands)
- **GPU**: CUDA-enabled (RTX class, 24 GB VRAM)
- **Python**: 3.12, venv at `./venv`
- **Key dependencies**: PyTorch, PyTorch Lightning, timm, diffusers, lpips, torchvision, scipy, opencv-python, matplotlib
