import torch
import torch.nn.functional as F

def keypoints_to_gaussian_maps(keypoints, H, W, sigma=0.1):
    """
    Convert keypoint coordinates to spatial Gaussian heatmaps.
    Implements equation (2) from the paper.

    Args:
        keypoints : (B, K, 2)  normalised coords in [-1, 1]
        H, W      : output spatial dimensions
        sigma     : Gaussian standard deviation (in normalised coords)

    Returns:
        maps : (B, K, H, W)  Gaussian heatmaps
    """
    B, K, _ = keypoints.shape
    device   = keypoints.device

    # Build normalised grid  (H, W, 2)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    grid = torch.stack([grid_x, grid_y], dim=-1)   # (H, W, 2)
    grid = grid.unsqueeze(0).unsqueeze(0)           # (1, 1, H, W, 2)

    # Broadcast keypoints  →  (B, K, 1, 1, 2)
    kp = keypoints.unsqueeze(2).unsqueeze(3)

    # Squared distance
    dist2 = ((grid - kp) ** 2).sum(dim=-1)          # (B, K, H, W)

    # Gaussian
    maps = torch.exp(-dist2 / (2 * sigma ** 2))
    return maps                                      # (B, K, H, W)
