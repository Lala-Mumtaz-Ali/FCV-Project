import torch
import torch.nn as nn

class ActionClassifier(nn.Module):
    def __init__(self, K=40, d_model=256, nhead=8, num_layers=4, num_actions=9):
        super().__init__()
        # Input keypoints shape: (B, seq_len, K, 2)
        # We flatten K*2 -> project to d_model
        self.input_proj = nn.Linear(K * 2, d_model)
        
        # Positional embedding for sequence length
        # Assuming seq_len <= 100 for safety
        self.pos_embed = nn.Parameter(torch.randn(1, 100, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_actions)
        )

    def forward(self, x):
        """
        x: (B, seq_len, K, 2)
        """
        B, seq_len, K, _ = x.shape
        
        # Flatten K and 2
        x = x.view(B, seq_len, K * 2)
        
        # Project to d_model
        x = self.input_proj(x)
        
        # Add positional embedding
        x = x + self.pos_embed[:, :seq_len, :]
        
        # Pass through transformer
        # x is (B, seq_len, d_model)
        x = self.transformer(x)
        
        # Mean pooling over time dimension
        x = x.mean(dim=1)
        
        # Classify
        logits = self.head(x)
        return logits
