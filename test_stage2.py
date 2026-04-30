import os
import glob
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from config import CFG, DEVICE
from data import VideoSequenceDataset, VideoFramePairDataset
from train import Stage2Module

def denormalize(tensor):
    """Convert image tensor in [-1, 1] to a numpy array in [0, 1] for matplotlib."""
    img = (tensor.detach().cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0
    return np.clip(img, 0, 1)

# Available actions:
# "baseball_pitch", "clean_and_jerk", "pull_ups",
# "baseball_swing", "golf_swing", "tennis_forehand",
# "jumping_jacks", "tennis_serve", "squats"

TARGET_ACTION = "tennis_serve"  # Change this string to test different actions!

def main():
    torch.set_grad_enabled(False)

    all_ckpts = glob.glob(os.path.join(CFG["ckpt_dir"], "*.ckpt"))
    s2_ckpts = [c for c in all_ckpts if "stage2" in os.path.basename(c)]
    
    if not s2_ckpts:
        print("No Stage 2 checkpoints found. Please run Stage 2 training first.")
        return
        
    latest_ckpt = max(s2_ckpts, key=os.path.getmtime)
    print(f"Loading Stage 2 checkpoint: {latest_ckpt}")

    model = Stage2Module.load_from_checkpoint(latest_ckpt, cfg=CFG)
    model.eval()
    model.to(DEVICE)

    val_ds = VideoSequenceDataset(CFG["data_root"], CFG["seq_len"], CFG["img_size"], split="val")

    target_idx = VideoFramePairDataset.ACTION2IDX.get(TARGET_ACTION)
    if target_idx is None:
        print(f"Invalid TARGET_ACTION. Choose from: {list(VideoFramePairDataset.ACTION2IDX.keys())}")
        return

    # Randomly search for a sequence that matches the requested action
    found_sample = None
    import random
    indices = list(range(len(val_ds)))
    random.shuffle(indices)
    
    for idx in indices:
        item = val_ds[idx]
        if item["action"].item() == target_idx:
            found_sample = item
            break

    if found_sample is None:
        print(f"Could not find any validation sequences for action: '{TARGET_ACTION}'")
        return

    frames = found_sample["frames"].unsqueeze(0).to(DEVICE)  # (1, seq_len, 3, H, W)
    actions = found_sample["action"].unsqueeze(0).to(DEVICE) # (1)

    # Forward pass
    B, seq_len, C, H, W = frames.shape
    frames_flat = frames.view(B * seq_len, C, H, W)
    
    print("Extracting keypoints...")
    keypoints, _ = model.detector(frames_flat)
    keypoints = keypoints.view(B, seq_len, CFG["K"], 2)
    
    print("Classifying action...")
    logits = model.classifier(keypoints)
    preds = torch.argmax(logits, dim=1)

    # Mapping idx to string
    idx2action = {v: k for k, v in VideoFramePairDataset.ACTION2IDX.items()}

    # Visualize the first sequence in the batch
    b_idx = 0
    seq_frames = frames[b_idx]
    seq_kps = keypoints[b_idx]
    
    true_idx = actions[b_idx].item()
    pred_idx = preds[b_idx].item()
    
    true_action = idx2action.get(true_idx, str(true_idx))
    pred_action = idx2action.get(pred_idx, str(pred_idx))

    print(f"Ground Truth: {true_action}")
    print(f"Prediction: {pred_action}")

    # Pick 8 evenly spaced frames from the sequence to display
    indices = np.linspace(0, seq_len - 1, min(8, seq_len), dtype=int)
    
    fig, axes = plt.subplots(1, len(indices), figsize=(16, 3))
    fig.suptitle(f"Action Recognition\nGround Truth: {true_action}  |  Predicted: {pred_action}", fontsize=14)

    for idx, frame_idx in enumerate(indices):
        frame_np = denormalize(seq_frames[frame_idx])
        kps = seq_kps[frame_idx]
        
        # Convert kps from [-1, 1] to pixel coords
        kps_x = (kps[:, 0].cpu().numpy() + 1) / 2.0 * W
        kps_y = (kps[:, 1].cpu().numpy() + 1) / 2.0 * H

        ax = axes[idx]
        ax.imshow(frame_np)
        ax.scatter(kps_x, kps_y, s=15, c='red', marker='o')
        ax.set_title(f"Frame {frame_idx}")
        ax.axis("off")

    plt.tight_layout()
    out_path = "stage2_test_results.png"
    plt.savefig(out_path)
    print(f"Visualization saved to {out_path}")

if __name__ == "__main__":
    main()
