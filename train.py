import torch
import pytorch_lightning as L
from models.detector import KeypointDetector
from models.translator import KeypointGuidedTranslator
from models.classifier import ActionClassifier
from loss import PatchDiscriminator, CombinedLoss
from utils import keypoints_to_gaussian_maps

class Stage1Module(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg        = cfg
        self.detector   = KeypointDetector(
            K         = cfg["K"],
            img_size  = cfg["img_size"],
            sigma     = cfg["sigma"],
            vit_model = cfg["vit_model"],
        )
        self.translator = KeypointGuidedTranslator(
            K        = cfg["K"],
            img_size = cfg["img_size"],
        )
        self.disc       = PatchDiscriminator()
        self.loss_fn    = CombinedLoss(
            lambda_lpips = cfg["lambda_lpips"],
            lambda_kl    = cfg["lambda_kl"],
            lambda_adv   = cfg["lambda_adv"],
            lambda_l1    = cfg["lambda_l1"],
        )
        self.automatic_optimization = False

    def on_train_epoch_start(self):
        if self.current_epoch == 5:
            self.detector.unfreeze_backbone()
            print("ViT backbone unfrozen ✓")

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        ref_img = batch["ref"].to(self.device)
        tgt_img = batch["tgt"].to(self.device)
        H, W    = ref_img.shape[-2:]

        ref_kp, _ = self.detector(ref_img)
        tgt_kp, _ = self.detector(tgt_img)

        ref_maps  = keypoints_to_gaussian_maps(ref_kp, H, W, self.cfg["sigma"])
        tgt_maps  = keypoints_to_gaussian_maps(tgt_kp, H, W, self.cfg["sigma"])

        mask, synth, pred = self.translator(ref_img, ref_maps, tgt_maps)

        opt_d.zero_grad()
        d_real = self.disc(tgt_img)
        d_fake = self.disc(pred.detach())
        d_loss = self.loss_fn.discriminator_loss(d_real, d_fake)
        self.manual_backward(d_loss)
        self.clip_gradients(opt_d, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        opt_d.step()
        opt_d.zero_grad()

        opt_g.zero_grad()
        recon_loss = self.loss_fn.reconstruction_loss(pred, tgt_img)
        adv_loss   = self.loss_fn.generator_adv_loss(self.disc(pred))
        g_loss     = recon_loss + adv_loss
        self.manual_backward(g_loss)
        self.clip_gradients(opt_g, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        opt_g.step()
        opt_g.zero_grad()

        self.log_dict({
            "s1/d_loss" : d_loss,
            "s1/recon"  : recon_loss,
            "s1/adv"    : adv_loss,
            "s1/g_loss" : g_loss,
        }, prog_bar=True, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(
            list(self.detector.parameters())
            + list(self.translator.parameters()),
            lr=self.cfg["lr_g"], betas=(0.5, 0.999), weight_decay=1e-4,
        )
        opt_d = torch.optim.AdamW(
            self.disc.parameters(),
            lr=self.cfg["lr_d"], betas=(0.5, 0.999),
        )
        return [opt_g, opt_d]

class Stage2Module(L.LightningModule):
    def __init__(self, cfg, stage1_ckpt_path=None):
        super().__init__()
        self.cfg = cfg

        # ── Stage 1 detector ─────────────────────────────────────────
        self.detector = KeypointDetector(
            K=cfg["K"],
            img_size=cfg["img_size"],
            sigma=cfg["sigma"],
            vit_model=cfg["vit_model"],
        )
        if stage1_ckpt_path:
            s1_module = Stage1Module.load_from_checkpoint(
                stage1_ckpt_path, cfg=cfg
            )
            self.detector.load_state_dict(s1_module.detector.state_dict())

        # Start fully frozen; the last ViT blocks are unfrozen after warm-up
        for p in self.detector.parameters():
            p.requires_grad = False
        self.detector.eval()

        # ── Classifier ───────────────────────────────────────────────
        self.classifier = ActionClassifier(
            K=cfg["K"],
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            num_actions=cfg["num_actions"],
        )

        # ── Loss ─────────────────────────────────────────────────────
        self.loss_fn = torch.nn.CrossEntropyLoss(
            label_smoothing=cfg.get("label_smoothing", 0.1)
        )

        # Track whether the backbone has been partially unfrozen
        self._backbone_unfrozen = False

    # ── Partial backbone unfreeze ─────────────────────────────────────
    def _unfreeze_last_vit_blocks(self, n_blocks=2):
        """Unfreeze the last `n_blocks` transformer blocks of the ViT backbone
        plus the heatmap head, so keypoints can be fine-tuned for action
        discrimination.  The rest of the backbone stays frozen."""
        # The ViT backbone blocks live at self.detector.backbone.blocks
        blocks = list(self.detector.backbone.blocks)
        for blk in blocks[-n_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
        # Also unfreeze the heatmap projection head
        for p in self.detector.heatmap_head.parameters():
            p.requires_grad = True
        self.detector.train()   # switch BN/LN to train mode for fine-tuned parts
        print(f"[Stage 2] Unfroze last {n_blocks} ViT blocks + heatmap head ✓")

    def on_train_epoch_start(self):
        warmup = self.cfg.get("warmup_epochs_s2", 10)
        if not self._backbone_unfrozen and self.current_epoch >= warmup:
            self._unfreeze_last_vit_blocks(n_blocks=2)
            self._backbone_unfrozen = True
            # Optimizers need to be reconfigured to pick up the new params.
            # Lightning re-uses the existing optimizer, so we manually add the
            # new parameter group with a lower LR.
            backbone_lr = self.cfg.get("lr_s2_backbone", 1e-5)
            new_params = [p for p in self.detector.parameters()
                          if p.requires_grad]
            if new_params:
                self.optimizers().add_param_group(
                    {"params": new_params, "lr": backbone_lr}
                )
                print(f"[Stage 2] Added backbone param group "
                      f"(lr={backbone_lr}) ✓")

    # ── Training step ─────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        frames = batch["frames"].to(self.device)   # (B, seq_len, 3, H, W)
        action = batch["action"].to(self.device)   # (B,)

        B, seq_len, C, H, W = frames.shape
        frames_flat = frames.view(B * seq_len, C, H, W)

        # Run detector — use no_grad only when backbone is still frozen
        if self._backbone_unfrozen:
            keypoints, _ = self.detector(frames_flat)
        else:
            with torch.no_grad():
                self.detector.eval()
                keypoints, _ = self.detector(frames_flat)

        keypoints = keypoints.view(B, seq_len, self.cfg["K"], 2)

        logits = self.classifier(keypoints)
        loss   = self.loss_fn(logits, action)

        # Top-1 and top-3 accuracy
        preds   = torch.argmax(logits, dim=1)
        top1    = (preds == action).float().mean()
        top3    = (torch.topk(logits, k=min(3, self.cfg["num_actions"]),
                              dim=1).indices
                   == action.unsqueeze(1)).any(dim=1).float().mean()

        self.log_dict({
            "s2/loss": loss,
            "s2/top1": top1,
            "s2/top3": top3,
        }, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    # ── Validation step ───────────────────────────────────────────────
    def validation_step(self, batch, batch_idx):
        frames = batch["frames"].to(self.device)
        action = batch["action"].to(self.device)

        B, seq_len, C, H, W = frames.shape
        keypoints, _ = self.detector(frames.view(B * seq_len, C, H, W))
        keypoints = keypoints.view(B, seq_len, self.cfg["K"], 2)

        logits = self.classifier(keypoints)
        loss   = self.loss_fn(logits, action)

        preds = torch.argmax(logits, dim=1)
        top1  = (preds == action).float().mean()
        top3  = (torch.topk(logits, k=min(3, self.cfg["num_actions"]),
                             dim=1).indices
                 == action.unsqueeze(1)).any(dim=1).float().mean()

        self.log_dict({
            "s2/val_loss": loss,
            "s2/val_top1": top1,
            "s2/val_top3": top3,
        }, prog_bar=True, on_epoch=True)

    # ── Optimiser + scheduler ─────────────────────────────────────────
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=self.cfg.get("lr_s2", 1e-4),
            weight_decay=1e-4,
        )
        warmup  = self.cfg.get("warmup_epochs_s2", 10)
        total   = self.cfg.get("max_epochs_s2", 80)

        # Linear warmup for the first `warmup` epochs (backbone still frozen),
        # then cosine decay for the remaining epochs.
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total - warmup, 1), eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

