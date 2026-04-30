import os
import glob
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from config import CFG, DEVICE
from data import VideoFramePairDataset
from train import Stage1Module
from utils import keypoints_to_gaussian_maps

def denormalize(tensor):
    """
    Convert image tensor in [-1, 1] to a numpy array in [0, 1] for matplotlib.
    """
    img = (tensor.detach().cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0
    return np.clip(img, 0, 1)

def draw_keypoints(img_np, keypoints, color='red'):
    """
    Optional helper to overlay keypoints on image.
    keypoints: shape (K, 2) in normalized coordinates [-1, 1]
    """
    # Just draw a scatter plot over the image using matplotlib
    # keypoints are in (x, y) format from detector
    pass 

def main():
    torch.set_grad_enabled(False)

    # 1. Find latest checkpoint
    all_ckpts = glob.glob(os.path.join(CFG["ckpt_dir"], "*.ckpt"))
    if not all_ckpts:
        print("No checkpoints found in 'checkpoints/' directory. Please run training first.")
        return
    
    latest_ckpt = max(all_ckpts, key=os.path.getmtime)
    print(f"Loading checkpoint: {latest_ckpt}")

    # 2. Load model
    model = Stage1Module.load_from_checkpoint(latest_ckpt, cfg=CFG)
    model.eval()
    model.to(DEVICE)

    # 3. Load dataset
    val_ds = VideoFramePairDataset(
        CFG["data_root"], CFG["img_size"], split="val"
    )
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=True)

    try:
        batch = next(iter(val_dl))
    except StopIteration:
        print("Dataset is empty. Are frames correctly placed in data_root?")
        return

    ref_img = batch["ref"].to(DEVICE)
    tgt_img = batch["tgt"].to(DEVICE)
    H, W = ref_img.shape[-2:]

    # 4. Forward pass
    print("Running inference...")
    ref_kp, _ = model.detector(ref_img)
    tgt_kp, _ = model.detector(tgt_img)

    ref_maps = keypoints_to_gaussian_maps(ref_kp, H, W, CFG["sigma"])
    tgt_maps = keypoints_to_gaussian_maps(tgt_kp, H, W, CFG["sigma"])

    _, _, recon_img = model.translator(ref_img, ref_maps, tgt_maps)

    # 5. Visualize
    B = ref_img.shape[0]
    fig, axes = plt.subplots(B, 3, figsize=(9, 3 * B))
    if B == 1:
        # To handle indexing correctly when B=1
        axes = [axes]
    
    # Set titles for columns
    if B > 0:
        axes[0][0].set_title("Reference Image")
        axes[0][1].set_title("Target Image")
        axes[0][2].set_title("Reconstructed Target")

    for i in range(B):
        ref_np = denormalize(ref_img[i])
        tgt_np = denormalize(tgt_img[i])
        recon_np = denormalize(recon_img[i])

        # We can also plot keypoints
        # ref_kp[i] is (K, 2) in [-1, 1], (x,y)
        kps_ref_x = (ref_kp[i, :, 0].cpu().numpy() + 1) / 2.0 * W
        kps_ref_y = (ref_kp[i, :, 1].cpu().numpy() + 1) / 2.0 * H
        
        kps_tgt_x = (tgt_kp[i, :, 0].cpu().numpy() + 1) / 2.0 * W
        kps_tgt_y = (tgt_kp[i, :, 1].cpu().numpy() + 1) / 2.0 * H

        axes[i][0].imshow(ref_np)
        axes[i][0].scatter(kps_ref_x, kps_ref_y, s=10, c='red', marker='o')
        axes[i][0].axis("off")
        
        axes[i][1].imshow(tgt_np)
        axes[i][1].scatter(kps_tgt_x, kps_tgt_y, s=10, c='red', marker='o')
        axes[i][1].axis("off")

        axes[i][2].imshow(recon_np)
        axes[i][2].axis("off")

    plt.tight_layout()
    
    # Save the figure
    out_path = "test_results.png"
    plt.savefig(out_path)
    print(f"Visualization saved to {out_path}")

if __name__ == "__main__":
    main()
