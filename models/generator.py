import torch
import torch.nn as nn

class TransformerCVAE(nn.Module):
    def __init__(self, K=40, latent_dim=256, num_actions=9,
                 seq_len=32, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.K          = K
        self.seq_len    = seq_len
        self.latent_dim = latent_dim

        # Input projections
        self.kp_proj    = nn.Linear(K * 2, d_model)
        self.action_emb = nn.Embedding(num_actions, d_model)
        self.pos_enc    = nn.Parameter(torch.randn(seq_len + 1, d_model))

        # Transformer encoder  →  mu, logvar
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead,
            dim_feedforward=512, dropout=0.1,
            batch_first=True, norm_first=True,   # pre-norm = more stable
        )
        self.encoder   = nn.TransformerEncoder(enc_layer, num_layers)
        self.to_mu     = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)

        # Transformer decoder  →  keypoint sequence
        self.z_proj    = nn.Linear(latent_dim, d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead,
            dim_feedforward=512, dropout=0.1,
            batch_first=True, norm_first=True,
        )
        self.decoder   = nn.TransformerDecoder(dec_layer, num_layers)
        self.out_proj  = nn.Linear(d_model, K * 2)

    # ── Encoder ────────────────────────────────────────────────
    def encode(self, kp_seq, action):
        """
        kp_seq : (B, T, K, 2)
        action : (B,)  long tensor
        """
        B, T, K, _ = kp_seq.shape
        flat = kp_seq.reshape(B, T, K * 2)       # (B, T, K*2)
        x    = self.kp_proj(flat)                 # (B, T, d_model)
        x    = x + self.pos_enc[:T]

        # Prepend action token
        a_tok = self.action_emb(action).unsqueeze(1)   # (B, 1, d_model)
        x     = torch.cat([a_tok, x], dim=1)           # (B, T+1, d_model)

        h     = self.encoder(x)                         # (B, T+1, d_model)
        h_cls = h[:, 0]                                 # CLS = action token

        return self.to_mu(h_cls), self.to_logvar(h_cls)

    # ── Reparameterise ─────────────────────────────────────────
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu   # deterministic at eval time

    # ── Decoder ────────────────────────────────────────────────
    def decode(self, z, init_kp, action, T):
        """
        z      : (B, latent_dim)
        init_kp: (B, K, 2)
        action : (B,)
        T      : number of future frames to generate
        """
        B   = z.shape[0]
        a   = self.action_emb(action)                  # (B, d_model)
        mem = (self.z_proj(z) + a).unsqueeze(1)        # (B, 1, d_model)

        # Learned positional queries for each future step
        queries = self.pos_enc[:T].unsqueeze(0).expand(B, -1, -1)  # (B, T, d)
        out     = self.decoder(queries, mem)                         # (B, T, d)
        kp_flat = self.out_proj(out)                                 # (B, T, K*2)
        return kp_flat.reshape(B, T, self.K, 2)

    # ── Forward ────────────────────────────────────────────────
    def forward(self, kp_seq, init_kp, action):
        """
        kp_seq  : (B, T, K, 2)  ground-truth future keypoints (training only)
        init_kp : (B, K, 2)     initial keypoints at t=0
        action  : (B,)

        returns: pred_seq (B, T, K, 2),  mu (B, D),  logvar (B, D)
        """
        mu, logvar = self.encode(kp_seq, action)
        z          = self.reparameterize(mu, logvar)
        pred_seq   = self.decode(z, init_kp, action, kp_seq.shape[1])
        return pred_seq, mu, logvar

    @torch.no_grad()
    def sample(self, init_kp, action, T):
        """Inference: sample z ~ N(0,I) and generate sequence."""
        B  = init_kp.shape[0]
        z  = torch.randn(B, self.latent_dim, device=init_kp.device)
        return self.decode(z, init_kp, action, T)
