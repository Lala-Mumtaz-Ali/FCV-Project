import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips as lpips_lib
from config import DEVICE

class CombinedLoss(nn.Module):
    def __init__(self, lambda_lpips=1.0, lambda_kl=0.001,
                 lambda_adv=0.1, lambda_l1=10.0):
        super().__init__()
        self.lpips_fn     = lpips_lib.LPIPS(net="alex").to(DEVICE)
        self.l1           = nn.L1Loss()
        self.lambda_lpips = lambda_lpips
        self.lambda_kl    = lambda_kl
        self.lambda_adv   = lambda_adv
        self.lambda_l1    = lambda_l1

        # Freeze LPIPS weights — it's a fixed metric, not trained
        for p in self.lpips_fn.parameters():
            p.requires_grad = False

    def reconstruction_loss(self, pred, target):
        """LPIPS + L1 for perceptual + pixel accuracy."""
        lp = self.lpips_fn(pred, target).mean()
        l1 = self.l1(pred, target)
        return self.lambda_lpips * lp + self.lambda_l1 * l1

    def kl_loss(self, mu, logvar):
        """KL divergence against N(0, I)  (equation 5)."""
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return self.lambda_kl * kl.sum(dim=-1).mean()

    def generator_adv_loss(self, disc_fake_logits):
        """Non-saturating generator loss."""
        return self.lambda_adv * F.softplus(-disc_fake_logits).mean()

    def discriminator_loss(self, disc_real, disc_fake):
        """Hinge discriminator loss — more stable than BCE."""
        real_loss = F.relu(1.0 - disc_real).mean()
        fake_loss = F.relu(1.0 + disc_fake).mean()
        return real_loss + fake_loss

    def keypoint_seq_loss(self, pred_seq, gt_seq):
        """L1 on keypoint sequence  (lambda_2 in equation 5)."""
        return self.l1(pred_seq, gt_seq)

class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator — penalises local artifacts."""
    def __init__(self, in_ch=3, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1 — no BN on first layer
            nn.Conv2d(in_ch, ndf,   4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            nn.Conv2d(ndf,   ndf*2, 4, 2, 1),
            nn.InstanceNorm2d(ndf*2),   # IN instead of BN = more stable
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1),
            nn.InstanceNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # Patch output
            nn.Conv2d(ndf*4, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.net(x)   # (B, 1, H', W')  patch logits
