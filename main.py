import os
import argparse
import glob
import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from torch.utils.data import DataLoader

from config import CFG, DEVICE
from data import VideoFramePairDataset, VideoSequenceDataset
from train import Stage1Module, Stage2Module

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2], help="Which stage to train")
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')
    print(f"Using device: {DEVICE}")

    if args.stage == 1:
        print("\n--- Starting Stage 1 (Keypoint Discovery) ---")
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

        # Simple checkpoint callback
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

        # Find the latest checkpoint
        all_ckpts = glob.glob(os.path.join(CFG["ckpt_dir"], "*.ckpt"))
        s1_ckpts = [c for c in all_ckpts if "stage1" in os.path.basename(c)]
        
        if s1_ckpts:
            latest_ckpt = max(s1_ckpts, key=os.path.getmtime)
            print(f"\nResuming Stage 1 training from {latest_ckpt}...")
            trainer_s1.fit(s1_module, train_dl, ckpt_path=latest_ckpt)
        else:
            print("\nStarting Stage 1 training from scratch...")
            trainer_s1.fit(s1_module, train_dl)

    elif args.stage == 2:
        print("\n--- Starting Stage 2 (Action Recognition) ---")
        # Prepare data
        train_ds = VideoSequenceDataset(
            CFG["data_root"], CFG["seq_len"], CFG["img_size"], split="train",
            clips_per_video=CFG.get("clips_per_video", 3),
        )
        train_dl = DataLoader(
            train_ds,
            batch_size  = CFG["batch_size"],
            shuffle     = True,
            num_workers = CFG["num_workers"],
            pin_memory  = True,
        )

        # Find the latest Stage 1 checkpoint
        all_ckpts = glob.glob(os.path.join(CFG["ckpt_dir"], "*.ckpt"))
        s1_ckpts = [c for c in all_ckpts if "stage1" in os.path.basename(c)]
        
        if not s1_ckpts:
            print("No Stage 1 checkpoint found! Please run Stage 1 first (python main.py --stage 1).")
            return
            
        latest_s1_ckpt = max(s1_ckpts, key=os.path.getmtime)
        print(f"Loading Stage 1 detector from {latest_s1_ckpt}")
        
        s2_module = Stage2Module(CFG, stage1_ckpt_path=latest_s1_ckpt)

        s2_ckpt_cb = ModelCheckpoint(
            dirpath        = CFG["ckpt_dir"],
            filename       = "stage2-epoch{epoch:02d}-loss{s2/loss_epoch:.4f}",
            monitor        = "s2/loss_epoch",
            mode           = "min",
            every_n_epochs = 5,
            save_top_k     = 3,          # keep the 3 best checkpoints
            save_last      = True,
            auto_insert_metric_name = False,
        )

        trainer_s2 = L.Trainer(
            max_epochs       = CFG["max_epochs_s2"],
            accelerator      = "gpu" if DEVICE == "cuda" else "cpu",
            devices          = 1,
            precision        = "16-mixed",
            callbacks        = [s2_ckpt_cb, TQDMProgressBar(refresh_rate=20)],
            default_root_dir = CFG["log_dir"],
            log_every_n_steps= 10,
        )

        s2_ckpts = [c for c in all_ckpts if "stage2" in os.path.basename(c)]
        if s2_ckpts:
            latest_s2_ckpt = max(s2_ckpts, key=os.path.getmtime)
            print(f"\nResuming Stage 2 from {latest_s2_ckpt}...")
            trainer_s2.fit(s2_module, train_dl, ckpt_path=latest_s2_ckpt)
        else:
            trainer_s2.fit(s2_module, train_dl)

    print("Done ✓ Training complete.")

if __name__ == "__main__":
    main()
