import torch
import torch.nn as nn


class ActionClassifier(nn.Module):
    """
    Transformer-based action classifier over a sequence of keypoint frames.

    Improvements over the original:
      - Learnable CLS token used for classification (instead of mean pooling).
        This is the standard ViT/BERT approach and lets the model learn what
        temporal context to aggregate.
      - Deeper MLP classification head with dropout for regularisation.
      - Dropout inside the Transformer encoder layers.
    """

    def __init__(self, K=40, d_model=256, nhead=8, num_layers=4,
                 num_actions=15, dropout=0.1):
        super().__init__()

        # Input projection: LayerNorm → Linear → Dropout
        self.input_norm = nn.LayerNorm(K * 2)
        self.input_proj = nn.Linear(K * 2, d_model)
        self.input_drop = nn.Dropout(dropout)

        # Learnable CLS token prepended to the sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional embedding: seq_len + 1 (for CLS) positions
        # Supports up to 101 positions (100 frames + CLS)
        self.pos_embed = nn.Parameter(torch.randn(1, 101, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,        # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Classification head: LayerNorm → Linear → GELU → Dropout → Linear
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_actions),
        )

    def forward(self, x):
        """
        Args:
            x: (B, seq_len, K, 2)  — keypoints for each frame
        Returns:
            logits: (B, num_actions)
        """
        B, seq_len, K, _ = x.shape

        # Normalise → project → dropout
        x = self.input_norm(x.view(B, seq_len, K * 2))
        x = self.input_drop(self.input_proj(x))         # (B, seq_len, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)                # (B, seq_len+1, d_model)

        # Add positional embedding
        x = x + self.pos_embed[:, :seq_len + 1, :]

        # Transformer
        x = self.transformer(x)                         # (B, seq_len+1, d_model)

        # Use only the CLS token output for classification
        cls_out = x[:, 0, :]                            # (B, d_model)

        return self.head(cls_out)                       # (B, num_actions)
