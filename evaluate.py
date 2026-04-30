import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CFG, DEVICE
from data import VideoFramePairDataset, VideoSequenceDataset
from train import Stage1Module, Stage2Module
from utils import keypoints_to_gaussian_maps

def evaluate_stage1(s1_ckpt):
    print(f"\n--- Evaluating Stage 1 (Keypoint Discovery) ---")
    print(f"Loading Checkpoint: {s1_ckpt}")
    model = Stage1Module.load_from_checkpoint(s1_ckpt, cfg=CFG).to(DEVICE)
    model.eval()
    
    val_ds = VideoFramePairDataset(CFG["data_root"], CFG["img_size"], split="val")
    val_dl = DataLoader(val_ds, batch_size=8, shuffle=False)
    
    total_mse = 0.0
    num_samples = 0
    
    for batch in val_dl:
        ref_img = batch["ref"].to(DEVICE)
        tgt_img = batch["tgt"].to(DEVICE)
        H, W = ref_img.shape[-2:]
        
        ref_kp, _ = model.detector(ref_img)
        tgt_kp, _ = model.detector(tgt_img)
        
        ref_maps = keypoints_to_gaussian_maps(ref_kp, H, W, CFG["sigma"])
        tgt_maps = keypoints_to_gaussian_maps(tgt_kp, H, W, CFG["sigma"])
        
        _, _, recon_img = model.translator(ref_img, ref_maps, tgt_maps)
        
        # Calculate Mean Squared Error for the reconstruction
        mse = F.mse_loss(recon_img, tgt_img, reduction='sum')
        total_mse += mse.item()
        num_samples += ref_img.size(0)
        
        # For a quick evaluation, limit to ~100 samples
        if num_samples >= 100:
            break
            
    if num_samples > 0:
        avg_mse = total_mse / (num_samples * 3 * H * W)
        print(f"-> Stage 1 Validation MSE: {avg_mse:.4f}")
    else:
        print("-> Stage 1 Validation dataset is empty.")

def evaluate_stage2(s2_ckpt):
    print(f"\n--- Evaluating Stage 2 (Action Recognition) ---")
    print(f"Loading Checkpoint: {s2_ckpt}")
    model = Stage2Module.load_from_checkpoint(s2_ckpt, cfg=CFG).to(DEVICE)
    model.eval()
    
    val_ds = VideoSequenceDataset(CFG["data_root"], CFG["seq_len"], CFG["img_size"], split="val")
    val_dl = DataLoader(val_ds, batch_size=8, shuffle=False)
    
    correct = 0
    total = 0
    
    for batch in val_dl:
        frames = batch["frames"].to(DEVICE)
        action = batch["action"].to(DEVICE)
        
        B, seq_len, C, H, W = frames.shape
        frames_flat = frames.view(B * seq_len, C, H, W)
        
        with torch.no_grad():
            keypoints, _ = model.detector(frames_flat)
        keypoints = keypoints.view(B, seq_len, CFG["K"], 2)
        
        logits = model.classifier(keypoints)
        preds = torch.argmax(logits, dim=1)
        
        correct += (preds == action).sum().item()
        total += B
        
        if total >= 100:
            break
            
    if total > 0:
        acc = correct / total
        print(f"-> Stage 2 Action Recognition Accuracy: {acc * 100:.2f}%")
    else:
        print("-> Stage 2 Validation dataset is empty.")

def main():
    torch.set_grad_enabled(False)
    print("=== Model Evaluation ===")
    
    all_ckpts = glob.glob(os.path.join(CFG["ckpt_dir"], "*.ckpt"))
    
    s1_ckpts = [c for c in all_ckpts if "stage1" in os.path.basename(c)]
    s2_ckpts = [c for c in all_ckpts if "stage2" in os.path.basename(c)]
    
    if s1_ckpts:
        latest_s1 = max(s1_ckpts, key=os.path.getmtime)
        evaluate_stage1(latest_s1)
    else:
        print("\nNo Stage 1 checkpoint found.")
        
    if s2_ckpts:
        latest_s2 = max(s2_ckpts, key=os.path.getmtime)
        evaluate_stage2(latest_s2)
    else:
        print("\nNo Stage 2 checkpoint found.")
        
    print("\n=== Evaluation Complete ===")

if __name__ == "__main__":
    main()
