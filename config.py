import os
import torch

# ── Global config ──────────────────────────────────────────────
CFG = dict(
    # dataset
    data_root    = "./dataset/Penn_Action",
    img_size     = 64,
    num_workers  = 12,

    # keypoints
    K            = 40,        # 40 for Penn Action, 15 for UvA-NEMO, 60 for MGIF

    # model
    vit_model    = "vit_small_patch16_224",
    latent_dim   = 256,
    d_model      = 256,
    nhead        = 8,
    num_layers   = 4,
    seq_len      = 32,
    num_actions  = 15,        # Penn Action uses 15 classes

    # training stage 1
    lr_g         = 1e-4,
    lr_d         = 1e-4,
    batch_size   = 64,        # fits comfortably on RTX 4090 24GB
    max_epochs_s1= 50,

    # training stage 2
    max_epochs_s2    = 80,
    lr_s2_backbone   = 1e-5,   # much lower LR for partially-unfrozen ViT layers
    clips_per_video  = 3,      # multi-clip sampling per video
    label_smoothing  = 0.1,    # cross-entropy label smoothing
    warmup_epochs_s2 = 10,     # epochs before partially unfreezing the ViT

    # loss weights  (eq. 4 and 5 in paper)
    lambda_lpips = 1.0,
    lambda_kl    = 0.001,
    lambda_adv   = 0.1,
    lambda_l1    = 10.0,

    # gaussian map sigma (eq. 2)
    sigma        = 0.1,

    # paths
    ckpt_dir     = "./checkpoints",
    log_dir      = "./logs",
)

os.makedirs(CFG["ckpt_dir"], exist_ok=True)
os.makedirs(CFG["log_dir"],  exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
