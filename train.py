import torch
import lightning as L
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
        }, prog_bar=True)

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
        
        # Load Stage 1 detector and freeze it
        self.detector = KeypointDetector(
            K=cfg["K"],
            img_size=cfg["img_size"],
            sigma=cfg["sigma"],
            vit_model=cfg["vit_model"],
        )
        if stage1_ckpt_path:
            s1_module = Stage1Module.load_from_checkpoint(stage1_ckpt_path, cfg=cfg)
            self.detector.load_state_dict(s1_module.detector.state_dict())
            
        for p in self.detector.parameters():
            p.requires_grad = False
        self.detector.eval()
            
        self.classifier = ActionClassifier(
            K=cfg["K"],
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            num_actions=cfg["num_actions"]
        )
        
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        frames = batch["frames"].to(self.device)  # (B, seq_len, 3, H, W)
        action = batch["action"].to(self.device)  # (B,)
        
        B, seq_len, C, H, W = frames.shape
        frames_flat = frames.view(B * seq_len, C, H, W)
        
        with torch.no_grad():
            self.detector.eval()
            keypoints, _ = self.detector(frames_flat)
            
        keypoints = keypoints.view(B, seq_len, self.cfg["K"], 2)
        
        logits = self.classifier(keypoints)
        loss = self.loss_fn(logits, action)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == action).float().mean()
        
        self.log_dict({
            "s2/loss": loss,
            "s2/acc": acc
        }, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=self.cfg.get("lr_s2", 1e-4),
            weight_decay=1e-4
        )
        return optimizer
