import os
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from torch.utils.data import DataLoader

from config import CFG, DEVICE
from data import VideoFramePairDataset
from train import Stage1Module

def main():
    torch.set_float32_matmul_precision('high')
    print(f"Using device: {DEVICE}")

    # Prepare data
    train_ds = VideoFramePairDataset(
        CFG["data_root"], CFG["img_size"], split="train"
    )
    train_dl = DataLoader(
        train_ds,
        batch_size  = CFG["batch_size"],
        shuffle     = True,
        num_workers = CFG["num_workers"],
        pin_memory  = True,
    )

    # Initialize model
    s1_module = Stage1Module(CFG)

    # Simple checkpoint callback without Kaggle specific loading logic
    s1_ckpt_cb = ModelCheckpoint(
        dirpath        = CFG["ckpt_dir"],
        filename       = "stage1-epoch{epoch:02d}",
        every_n_epochs = 5,
        save_top_k     = -1,
        save_last      = True,
    )

    trainer_s1 = L.Trainer(
        max_epochs       = CFG["max_epochs_s1"],
        accelerator      = "gpu" if DEVICE == "cuda" else "cpu",
        devices          = 1,
        precision        = "16-mixed",
        callbacks        = [s1_ckpt_cb, TQDMProgressBar(refresh_rate=20)],
        default_root_dir = CFG["log_dir"],
        log_every_n_steps= 10,
    )

    # Find the latest checkpoint (handling last.ckpt, last-v1.ckpt, etc.)
    import glob
    all_ckpts = glob.glob(os.path.join(CFG["ckpt_dir"], "*.ckpt"))
    if all_ckpts:
        latest_ckpt = max(all_ckpts, key=os.path.getmtime)
        print(f"\nResuming Stage 1 training from {latest_ckpt}...")
        trainer_s1.fit(s1_module, train_dl, ckpt_path=latest_ckpt)
    else:
        print("\nStarting Stage 1 training from scratch...")
        trainer_s1.fit(s1_module, train_dl)
        
    print("Done ✓ Training complete.")

if __name__ == "__main__":
    main()
