import os
import sys
import glob
import argparse
import torch

sys.stdout.reconfigure(encoding='utf-8')
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from collections import defaultdict

from config import CFG, DEVICE
from data import VideoFramePairDataset, VideoSequenceDataset
from train import Stage1Module, Stage2Module
from utils import keypoints_to_gaussian_maps

ACTIONS = VideoFramePairDataset.ACTIONS


# ── Utilities ────────────────────────────────────────────────────────────────

def denormalize(t):
    img = (t.detach().cpu().float().numpy().transpose(1, 2, 0) + 1.0) / 2.0
    return np.clip(img, 0, 1)


def psnr(pred, target, max_val=2.0):
    mse = F.mse_loss(pred, target).clamp(min=1e-10)
    return (20 * torch.log10(torch.tensor(max_val, device=pred.device) / mse.sqrt())).item()


def ssim_batch(pred, target):
    C1, C2 = (0.01 * 2) ** 2, (0.03 * 2) ** 2
    ks = 11
    pad = ks // 2
    k = torch.ones(1, 1, ks, ks, device=pred.device) / (ks * ks)
    vals = []
    for c in range(pred.shape[1]):
        p = pred[:, c:c+1]
        t = target[:, c:c+1]
        mu_p = F.conv2d(p, k, padding=pad)
        mu_t = F.conv2d(t, k, padding=pad)
        sig_p  = F.conv2d(p * p, k, padding=pad) - mu_p ** 2
        sig_t  = F.conv2d(t * t, k, padding=pad) - mu_t ** 2
        sig_pt = F.conv2d(p * t, k, padding=pad) - mu_p * mu_t
        num = (2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)
        den = (mu_p ** 2 + mu_t ** 2 + C1) * (sig_p + sig_t + C2)
        vals.append((num / den.clamp(min=1e-8)).mean())
    return torch.stack(vals).mean().item()


def confusion_matrix_np(labels, preds, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(labels, preds):
        cm[int(t)][int(p)] += 1
    return cm


def find_latest(stage):
    ckpts = glob.glob(os.path.join(CFG["ckpt_dir"], "*.ckpt"))
    tagged = [c for c in ckpts if f"stage{stage}" in os.path.basename(c)]
    return max(tagged, key=os.path.getmtime) if tagged else None


# ── Stage 1 ──────────────────────────────────────────────────────────────────

def evaluate_stage1(ckpt_path):
    print(f"\n{'='*64}")
    print(f"  STAGE 1 — Keypoint Discovery & Image Reconstruction")
    print(f"{'='*64}")
    print(f"  Checkpoint : {os.path.basename(ckpt_path)}")

    model = Stage1Module.load_from_checkpoint(ckpt_path, cfg=CFG).to(DEVICE)
    model.eval()

    val_ds = VideoFramePairDataset(CFG["data_root"], CFG["img_size"], split="val")
    val_dl = DataLoader(val_ds, batch_size=8, shuffle=False,
                        num_workers=0, pin_memory=False)

    # Try loading LPIPS (already a dependency of this project)
    try:
        import lpips as lpips_lib
        lpips_fn = lpips_lib.LPIPS(net='alex').to(DEVICE)
        lpips_fn.eval()
        use_lpips = True
    except Exception:
        use_lpips = False

    mse_vals, psnr_vals, ssim_vals, lpips_vals = [], [], [], []
    kp_spread_vals, kp_motion_vals = [], []

    vis_refs, vis_tgts, vis_recons = [], [], []
    vis_kp_ref, vis_kp_tgt = [], []
    MAX_VIS = 8

    print(f"\n  Evaluating {len(val_ds)} samples...\n")
    header = f"  {'Batch':>5}  {'MSE':>7}  {'PSNR':>8}  {'SSIM':>7}"
    if use_lpips:
        header += f"  {'LPIPS':>7}"
    print(header)
    print(f"  {'-'*55}")

    with torch.no_grad():
        for i, batch in enumerate(val_dl):
            ref_img = batch["ref"].to(DEVICE)
            tgt_img = batch["tgt"].to(DEVICE)
            H, W = ref_img.shape[-2:]

            ref_kp, _ = model.detector(ref_img)
            tgt_kp, _ = model.detector(tgt_img)
            ref_maps  = keypoints_to_gaussian_maps(ref_kp, H, W, CFG["sigma"])
            tgt_maps  = keypoints_to_gaussian_maps(tgt_kp, H, W, CFG["sigma"])
            _, _, recon = model.translator(ref_img, ref_maps, tgt_maps)

            mse  = F.mse_loss(recon, tgt_img).item()
            p    = psnr(recon, tgt_img)
            s    = ssim_batch(recon, tgt_img)
            mse_vals.append(mse); psnr_vals.append(p); ssim_vals.append(s)

            if use_lpips:
                lp = lpips_fn(recon, tgt_img).mean().item()
                lpips_vals.append(lp)

            # Keypoint spread: std of positions across K keypoints
            kp_spread_vals.append(ref_kp.std(dim=1).mean().item())
            # Keypoint motion: mean L2 shift between ref and tgt keypoints
            kp_motion_vals.append((ref_kp - tgt_kp).norm(dim=-1).mean().item())

            if i % 5 == 0:
                row = f"  {i:>5}  {mse:>7.4f}  {p:>8.2f}  {s:>7.4f}"
                if use_lpips:
                    row += f"  {lpips_vals[-1]:>7.4f}"
                print(row)

            if len(vis_refs) < MAX_VIS:
                n = min(MAX_VIS - len(vis_refs), ref_img.shape[0])
                vis_refs.extend(ref_img[:n].cpu())
                vis_tgts.extend(tgt_img[:n].cpu())
                vis_recons.extend(recon[:n].cpu())
                vis_kp_ref.extend(ref_kp[:n].cpu())
                vis_kp_tgt.extend(tgt_kp[:n].cpu())

    avg_mse    = float(np.mean(mse_vals))
    avg_psnr   = float(np.mean(psnr_vals))
    avg_ssim   = float(np.mean(ssim_vals))
    avg_spread = float(np.mean(kp_spread_vals))
    avg_motion = float(np.mean(kp_motion_vals))

    print(f"\n{'─'*64}")
    print(f"  RESULTS  ({len(val_ds)} validation samples)")
    print(f"{'─'*64}")
    print(f"\n  Reconstruction Quality")
    print(f"  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  MSE   {avg_mse:>8.4f}   (lower is better)            │")
    print(f"  │  PSNR  {avg_psnr:>7.2f} dB  (>20 dB is good)            │")
    print(f"  │  SSIM  {avg_ssim:>8.4f}   (>0.70 is good, max=1.0)    │")
    if use_lpips:
        avg_lpips = float(np.mean(lpips_vals))
        print(f"  │  LPIPS {avg_lpips:>8.4f}   (<0.30 is good)              │")
    print(f"  └─────────────────────────────────────────────────────┘")
    print(f"\n  Keypoint Analysis")
    print(f"  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  Spread {avg_spread:>7.4f}   (>0.30 = well spread across img) │")
    print(f"  │  Motion {avg_motion:>7.4f}   (>0.05 = tracking temporal change)│")
    print(f"  └─────────────────────────────────────────────────────┘")

    print(f"\n  DIAGNOSIS:")
    issues = 0
    if avg_psnr < 15:
        print(f"  [POOR ]  PSNR {avg_psnr:.1f} dB — reconstruction quality is very low.")
        issues += 1
    elif avg_psnr < 20:
        print(f"  [FAIR ]  PSNR {avg_psnr:.1f} dB — model is learning, continue training.")
    else:
        print(f"  [GOOD ]  PSNR {avg_psnr:.1f} dB — reconstruction is high quality.")

    if avg_ssim < 0.5:
        print(f"  [POOR ]  SSIM {avg_ssim:.3f} — structure not well preserved.")
        issues += 1
    elif avg_ssim < 0.7:
        print(f"  [FAIR ]  SSIM {avg_ssim:.3f} — moderate structural preservation.")
    else:
        print(f"  [GOOD ]  SSIM {avg_ssim:.3f} — structure well preserved.")

    if avg_spread < 0.2:
        print(f"  [WARN ]  KP spread {avg_spread:.3f} — keypoints may be collapsing.")
        issues += 1
    else:
        print(f"  [GOOD ]  KP spread {avg_spread:.3f} — keypoints cover the image well.")

    if avg_motion < 0.03:
        print(f"  [WARN ]  KP motion {avg_motion:.3f} — keypoints barely moving between frames.")
        issues += 1
    else:
        print(f"  [GOOD ]  KP motion {avg_motion:.3f} — keypoints track motion across frames.")

    print(f"{'─'*64}")

    _save_stage1_visuals(vis_refs, vis_tgts, vis_recons,
                         vis_kp_ref, vis_kp_tgt, H, W)

    return {
        "mse": avg_mse, "psnr": avg_psnr, "ssim": avg_ssim,
        "kp_spread": avg_spread, "kp_motion": avg_motion,
    }


def _save_stage1_visuals(refs, tgts, recons, kp_refs, kp_tgts, H, W):
    n = len(refs)
    fig, axes = plt.subplots(n, 3, figsize=(10, 3.0 * n))
    if n == 1:
        axes = [axes]

    col_titles = ["Reference + Keypoints", "Target + Keypoints", "Reconstructed Target"]
    for ci, title in enumerate(col_titles):
        axes[0][ci].set_title(title, fontsize=9, fontweight='bold', pad=4)

    for i in range(n):
        ref_np   = denormalize(refs[i])
        tgt_np   = denormalize(tgts[i])
        recon_np = denormalize(recons[i])

        kpr = kp_refs[i]   # (K, 2)
        kpt = kp_tgts[i]

        rx = (kpr[:, 0].numpy() + 1) / 2.0 * W
        ry = (kpr[:, 1].numpy() + 1) / 2.0 * H
        tx = (kpt[:, 0].numpy() + 1) / 2.0 * W
        ty = (kpt[:, 1].numpy() + 1) / 2.0 * H

        axes[i][0].imshow(ref_np)
        axes[i][0].scatter(rx, ry, s=6, c='red',  marker='o', linewidths=0)
        axes[i][0].axis('off')

        axes[i][1].imshow(tgt_np)
        axes[i][1].scatter(tx, ty, s=6, c='lime', marker='o', linewidths=0)
        axes[i][1].axis('off')

        axes[i][2].imshow(recon_np)
        axes[i][2].axis('off')

    plt.tight_layout(pad=0.5)
    out = "eval_stage1_visuals.png"
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\n  Visual grid    → {out}")


# ── Stage 2 ──────────────────────────────────────────────────────────────────

def evaluate_stage2(ckpt_path):
    print(f"\n{'='*64}")
    print(f"  STAGE 2 — Action Recognition")
    print(f"{'='*64}")
    print(f"  Checkpoint : {os.path.basename(ckpt_path)}")

    model = Stage2Module.load_from_checkpoint(ckpt_path, cfg=CFG).to(DEVICE)
    model.eval()
    model.detector.eval()

    val_ds = VideoSequenceDataset(
        CFG["data_root"], CFG["seq_len"], CFG["img_size"], split="val"
    )
    val_dl = DataLoader(val_ds, batch_size=8, shuffle=False,
                        num_workers=0, pin_memory=False)

    all_preds, all_labels, all_confs, all_top3_correct = [], [], [], []

    print(f"\n  Evaluating {len(val_ds)} samples...\n")

    with torch.no_grad():
        for batch in val_dl:
            frames = batch["frames"].to(DEVICE)
            action = batch["action"].to(DEVICE)

            B, seq_len, C, H, W = frames.shape
            keypoints, _ = model.detector(frames.view(B * seq_len, C, H, W))
            keypoints = keypoints.view(B, seq_len, CFG["K"], 2)

            logits = model.classifier(keypoints)
            probs  = torch.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1)
            confs  = probs.max(dim=1).values

            k3 = min(3, CFG["num_actions"])
            top3_idx = probs.topk(k3, dim=1).indices
            top3_hit = (top3_idx == action.unsqueeze(1)).any(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(action.cpu().tolist())
            all_confs.extend(confs.cpu().tolist())
            all_top3_correct.extend(top3_hit.cpu().tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confs  = np.array(all_confs)
    top3_acc   = float(np.mean(all_top3_correct)) * 100
    top1_acc   = float((all_preds == all_labels).mean()) * 100

    correct_mask = all_preds == all_labels
    conf_correct = all_confs[correct_mask].mean() * 100 if correct_mask.any() else 0.0
    conf_wrong   = all_confs[~correct_mask].mean() * 100 if (~correct_mask).any() else 0.0

    print(f"{'─'*64}")
    print(f"  RESULTS  ({len(all_labels)} validation samples)")
    print(f"{'─'*64}")
    print(f"\n  Overall Accuracy")
    print(f"  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  Top-1 Accuracy  : {top1_acc:>6.2f}%                       │")
    print(f"  │  Top-3 Accuracy  : {top3_acc:>6.2f}%                       │")
    print(f"  └─────────────────────────────────────────────────────┘")
    print(f"\n  Confidence Analysis")
    print(f"  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  Mean confidence (correct) : {conf_correct:>5.1f}%               │")
    print(f"  │  Mean confidence (wrong)   : {conf_wrong:>5.1f}%               │")
    print(f"  └─────────────────────────────────────────────────────┘")

    # Per-class accuracy
    print(f"\n  PER-CLASS ACCURACY  ({CFG['num_actions']} classes)")
    print(f"  {'Class':<24}  {'Correct':>7}  {'Total':>6}  {'Acc%':>6}  Bar")
    print(f"  {'-'*62}")
    per_class = {}
    for idx, name in enumerate(ACTIONS):
        mask = all_labels == idx
        if mask.sum() == 0:
            continue
        n_correct = int((all_preds[mask] == idx).sum())
        n_total   = int(mask.sum())
        acc       = n_correct / n_total * 100
        per_class[name] = acc
        bar = '█' * int(acc / 5) + '░' * (20 - int(acc / 5))
        print(f"  {name:<24}  {n_correct:>7}  {n_total:>6}  {acc:>5.1f}%  {bar}")

    # Confusion matrix
    cm_path = _save_confusion_matrix(all_labels, all_preds)

    # Accuracy bar chart
    bar_path = _save_per_class_bar(per_class)

    # Diagnosis
    print(f"\n  DIAGNOSIS:")
    if top1_acc < 30:
        print(f"  [POOR ]  Top-1 {top1_acc:.1f}% — needs significantly more Stage 2 training.")
    elif top1_acc < 60:
        print(f"  [FAIR ]  Top-1 {top1_acc:.1f}% — model is learning action patterns.")
    elif top1_acc < 80:
        print(f"  [GOOD ]  Top-1 {top1_acc:.1f}% — approaching state-of-the-art performance.")
    else:
        print(f"  [EXCEL]  Top-1 {top1_acc:.1f}% — excellent classification performance.")

    if conf_wrong > 70:
        print(f"  [WARN ]  High confidence on wrong predictions ({conf_wrong:.1f}%) — model is overconfident.")
    if per_class:
        worst = min(per_class, key=per_class.get)
        best  = max(per_class, key=per_class.get)
        print(f"  [INFO ]  Best  class: {best} ({per_class[best]:.1f}%)")
        print(f"  [INFO ]  Worst class: {worst} ({per_class[worst]:.1f}%)")
        spread = per_class[best] - per_class[worst]
        if spread > 50:
            print(f"  [WARN ]  Large accuracy spread ({spread:.1f}%) — some classes are much harder.")

    print(f"{'─'*64}")

    return {"top1": top1_acc, "top3": top3_acc, "per_class": per_class}


def _save_confusion_matrix(labels, preds):
    n = CFG["num_actions"]
    cm = confusion_matrix_np(labels, preds, n)

    # Row-normalise (avoid div by zero)
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm  = cm.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(ACTIONS, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ACTIONS, fontsize=8)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax.set_ylabel("True Label", fontsize=11, labelpad=8)
    ax.set_title("Stage 2 — Confusion Matrix (row-normalised)", fontsize=12, fontweight='bold', pad=10)

    for i in range(n):
        for j in range(n):
            v = cm_norm[i, j]
            if cm[i, j] == 0:
                continue
            color = 'white' if v > 0.5 else 'black'
            ax.text(j, i, f"{v:.2f}", ha='center', va='center',
                    fontsize=6.5, color=color)

    plt.tight_layout()
    out = "eval_stage2_confusion.png"
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\n  Confusion matrix  → {out}")
    return out


def _save_per_class_bar(per_class):
    names = list(per_class.keys())
    accs  = [per_class[n] for n in names]
    colors = ['#2ecc71' if a >= 60 else '#f39c12' if a >= 30 else '#e74c3c' for a in accs]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(names, accs, color=colors, edgecolor='white', height=0.6)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Accuracy (%)", fontsize=11)
    ax.set_title("Stage 2 — Per-Class Accuracy", fontsize=12, fontweight='bold')
    ax.axvline(x=float(np.mean(accs)), color='navy', linestyle='--',
               linewidth=1.2, label=f"Mean {np.mean(accs):.1f}%")
    for bar, acc in zip(bars, accs):
        ax.text(acc + 1, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%", va='center', fontsize=8)
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    out = "eval_stage2_per_class.png"
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Per-class chart   → {out}")
    return out


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate FCV model")
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True,
                        help="Stage to evaluate: 1 (reconstruction) or 2 (classification)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Specific checkpoint path (default: latest for the stage)")
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    ckpt = args.ckpt or find_latest(args.stage)
    if ckpt is None:
        print(f"\nNo Stage {args.stage} checkpoint found in '{CFG['ckpt_dir']}'.")
        print("Run training first:  python main.py --stage " + str(args.stage))
        return

    if args.stage == 1:
        evaluate_stage1(ckpt)
    else:
        evaluate_stage2(ckpt)

    print("\n=== Evaluation complete ===\n")


if __name__ == "__main__":
    main()
