import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class KeypointDetector(nn.Module):
    def __init__(self, K=40, img_size=64, sigma=0.1,
                 vit_model="vit_small_patch16_224"):
        super().__init__()
        self.K     = K
        self.sigma = sigma

        # ViT backbone pretrained with DINOv2
        self.backbone = timm.create_model(
            vit_model,
            pretrained=True,
            img_size=img_size,
            num_classes=0,       # remove classifier head
        )
        for p in self.backbone.parameters():
            p.requires_grad = False  # freeze backbone initially

        feat_dim = self.backbone.num_features   # 192 for ViT-Tiny
        self.patch_size = 16
        self.grid_h = self.grid_w = img_size // self.patch_size # 4 for 64px

        # Lightweight heatmap projection head
        self.heatmap_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Linear(256, K),
        )
        self.upsample = nn.Upsample(
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )

    def unfreeze_backbone(self):
        """Call after a few warm-up epochs."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x):
        """
        x      : (B, 3, H, W)
        returns: keypoints (B, K, 2),  soft_maps (B, K, H, W)
        """
        B, _, H, W = x.shape

        # Extract patch tokens (drop CLS)
        tokens = self.backbone.forward_features(x)   # (B, N+1, D)
        patch_tokens = tokens[:, 1:, :]               # (B, N, D)

        # Project to K channels, reshape to spatial grid
        heatmaps = self.heatmap_head(patch_tokens)    # (B, N, K)
        heatmaps = heatmaps.permute(0, 2, 1)          # (B, K, N)
        heatmaps = heatmaps.reshape(
            B, self.K, self.grid_h, self.grid_w
        )
        heatmaps = self.upsample(heatmaps)            # (B, K, H, W)

        # Softmax over spatial dims  →  probability map
        flat     = heatmaps.reshape(B, self.K, -1)
        soft_map = F.softmax(flat, dim=-1).reshape(B, self.K, H, W)

        # Expected coordinates  (equation 1)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij",
        )
        coords_x = (soft_map * grid_x).sum(dim=(-2, -1))  # (B, K)
        coords_y = (soft_map * grid_y).sum(dim=(-2, -1))
        keypoints = torch.stack([coords_x, coords_y], dim=-1)  # (B, K, 2)

        return keypoints, soft_map
