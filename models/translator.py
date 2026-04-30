import torch
import torch.nn as nn
from diffusers import UNet2DModel

class KeypointGuidedTranslator(nn.Module):
    def __init__(self, K=40, img_size=128):
        super().__init__()
        # Input channels: ref image (3) + ref kp maps (K) + target kp maps (K)
        in_ch = 3 + K + K

        self.unet = UNet2DModel(
            sample_size      = img_size,
            in_channels      = in_ch,
            out_channels     = 4,           # 3 RGB synth + 1 mask
            layers_per_block = 2,
            block_out_channels = (64, 128, 256, 256),
            down_block_types = (
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types = (
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, ref_img, ref_kp_maps, tgt_kp_maps):
        """
        ref_img     : (B, 3, H, W)   reference image
        ref_kp_maps : (B, K, H, W)   Gaussian maps for reference keypoints
        tgt_kp_maps : (B, K, H, W)   Gaussian maps for target keypoints

        returns:
            mask  : (B, 1, H, W)  background mask  (1=keep bg, 0=use synth)
            synth : (B, 3, H, W)  synthesised foreground
            pred  : (B, 3, H, W)  blended output   (equation 3)
        """
        x   = torch.cat([ref_img, ref_kp_maps, tgt_kp_maps], dim=1)
        # UNet2DModel needs a timestep; pass zeros (not diffusion)
        t   = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        out = self.unet(x, timestep=t).sample        # (B, 4, H, W)

        synth = torch.tanh(out[:, :3])               # synthesised object
        mask  = torch.sigmoid(out[:, 3:4])           # background mask

        # Equation (3): blend background and synthesised image
        pred  = mask * ref_img + (1 - mask) * synth

        return mask, synth, pred
