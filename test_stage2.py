import os
import sys
import glob
import argparse
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

from config import CFG, DEVICE
from data import VideoSequenceDataset, VideoFramePairDataset
from train import Stage2Module

ACTIONS    = VideoFramePairDataset.ACTIONS
ACTION2IDX = VideoFramePairDataset.ACTION2IDX
IDX2ACTION = {v: k for k, v in ACTION2IDX.items()}

N_FRAMES = 8   # evenly-spaced frames shown in static grid


def denormalize(t):
    img = (t.detach().cpu().float().numpy().transpose(1, 2, 0) + 1.0) / 2.0
    return np.clip(img, 0, 1)


def find_clips(val_ds, target_idx, n):
    clips = []
    for i in range(len(val_ds)):
        item = val_ds[i]
        if item["action"].item() == target_idx:
            clips.append(item)
        if len(clips) >= n:
            break
    return clips


def run_inference(model, clip):
    frames = clip["frames"].unsqueeze(0).to(DEVICE)
    B, T, C, H, W = frames.shape

    with torch.no_grad():
        keypoints, _ = model.detector(frames.view(B * T, C, H, W))
        keypoints = keypoints.view(B, T, CFG["K"], 2)
        logits    = model.classifier(keypoints)

    probs                = F.softmax(logits, dim=1)[0].cpu()
    top3_vals, top3_idxs = probs.topk(3)
    pred_idx  = top3_idxs[0].item()
    pred_conf = top3_vals[0].item() * 100

    return (
        keypoints[0].cpu(),
        probs,
        pred_idx,
        pred_conf,
        top3_idxs.tolist(),
        top3_vals.tolist(),
        H, W,
    )


# ── Static grid ──────────────────────────────────────────────────────────────

def draw_row(axes_row, ax_bar, frames_tensor, keypoints,
             target_action, pred_action, pred_conf,
             top3_idxs, top3_vals, H, W, is_correct, clip_num):

    T          = frames_tensor.shape[0]
    frame_idxs = np.linspace(0, T - 1, N_FRAMES, dtype=int)
    kp_colors  = plt.cm.hsv(np.linspace(0, 1, CFG["K"], endpoint=False))
    border_col = '#2ecc71' if is_correct else '#e74c3c'

    for col, f_idx in enumerate(frame_idxs):
        ax = axes_row[col]
        ax.imshow(denormalize(frames_tensor[f_idx]))

        kps   = keypoints[f_idx]
        kps_x = (kps[:, 0].numpy() + 1) / 2.0 * W
        kps_y = (kps[:, 1].numpy() + 1) / 2.0 * H
        ax.scatter(kps_x, kps_y, s=8, c=kp_colors, marker='o', linewidths=0, zorder=3)

        ax.set_title(f"t={f_idx}", fontsize=6, pad=1)
        ax.axis('off')

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_col)
            spine.set_linewidth(2.5)

    status = "CORRECT" if is_correct else "WRONG"
    axes_row[0].set_ylabel(
        f"Clip {clip_num}\n{status}\n{pred_conf:.0f}%",
        rotation=0, labelpad=48, va='center', ha='right',
        fontsize=8, fontweight='bold', color=border_col,
    )

    bar_labels = [IDX2ACTION[i] for i in top3_idxs]
    bar_vals   = [v * 100 for v in top3_vals]
    bar_colors = ['#2ecc71' if IDX2ACTION[i] == target_action else '#5dade2'
                  for i in top3_idxs]

    bars = ax_bar.barh(range(3), bar_vals, color=bar_colors, height=0.5)
    ax_bar.set_xlim(0, 110)
    ax_bar.set_yticks(range(3))
    ax_bar.set_yticklabels(bar_labels, fontsize=7)
    ax_bar.set_xticks([0, 50, 100])
    ax_bar.set_xticklabels(['0', '50%', '100%'], fontsize=6)
    ax_bar.invert_yaxis()
    ax_bar.set_title("Top-3 Probs", fontsize=7, pad=2)
    ax_bar.spines[['top', 'right']].set_visible(False)

    for bar, val in zip(bars, bar_vals):
        ax_bar.text(val + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va='center', fontsize=6.5)


def save_static_grid(clips, results, target_action):
    n_rows = len(clips)
    n_cols = N_FRAMES + 1

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.0 + 0.5, n_rows * 2.8),
        gridspec_kw={"width_ratios": [1] * N_FRAMES + [1.8],
                     "wspace": 0.06, "hspace": 0.45},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f'Stage 2 Qualitative Test  —  Action: "{target_action}"',
        fontsize=12, fontweight='bold', y=1.01,
    )

    for row, (clip, res) in enumerate(zip(clips, results)):
        kps, probs, pred_idx, pred_conf, top3_idxs, top3_vals, H, W = res
        pred_action = IDX2ACTION[pred_idx]
        is_correct  = (pred_idx == ACTION2IDX[target_action])

        draw_row(
            axes[row, :N_FRAMES], axes[row, N_FRAMES],
            clip["frames"], kps,
            target_action, pred_action, pred_conf,
            top3_idxs, top3_vals, H, W, is_correct, row + 1,
        )

    out = f"stage2_test_{target_action}.png"
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Static grid -> {out}")


# ── Animated GIF ─────────────────────────────────────────────────────────────

def save_animation(clips, results, target_action):
    n    = len(clips)
    T    = CFG["seq_len"]
    kp_c = plt.cm.hsv(np.linspace(0, 1, CFG["K"], endpoint=False))

    fig, axes = plt.subplots(
        1, n,
        figsize=(n * 3.2, 3.6),
        facecolor='#111111',
    )
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f'Action: "{target_action}"',
        fontsize=11, fontweight='bold', color='white', y=0.98,
    )

    ims, scatters, frame_texts = [], [], []

    for col, (clip, res) in enumerate(zip(clips, results)):
        kps, probs, pred_idx, pred_conf, top3_idxs, top3_vals, H, W = res
        ax = axes[col]
        ax.set_facecolor('#111111')

        frame0 = denormalize(clip["frames"][0])
        im = ax.imshow(frame0, animated=True)

        kp0   = kps[0]
        kps_x = (kp0[:, 0].numpy() + 1) / 2.0 * W
        kps_y = (kp0[:, 1].numpy() + 1) / 2.0 * H
        sc = ax.scatter(kps_x, kps_y, s=12, c=kp_c,
                        marker='o', linewidths=0, zorder=3, animated=True)

        is_correct  = (pred_idx == ACTION2IDX[target_action])
        pred_name   = IDX2ACTION[pred_idx]
        title_color = '#2ecc71' if is_correct else '#e74c3c'
        status      = 'CORRECT' if is_correct else f'WRONG: {pred_name}'

        ax.set_title(
            f"Clip {col + 1}  |  {status}\n{pred_conf:.0f}% confidence",
            fontsize=8, color=title_color, fontweight='bold', pad=4,
        )
        ax.axis('off')

        ft = ax.text(
            0.5, 0.02, 'f=0',
            transform=ax.transAxes,
            ha='center', va='bottom',
            fontsize=7, color='white',
            bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'),
            animated=True,
        )

        ims.append(im)
        scatters.append(sc)
        frame_texts.append(ft)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    def update(f_idx):
        for col, (clip, res) in enumerate(zip(clips, results)):
            kps, _, _, _, _, _, H, W = res

            ims[col].set_data(denormalize(clip["frames"][f_idx]))

            kp    = kps[f_idx]
            kps_x = (kp[:, 0].numpy() + 1) / 2.0 * W
            kps_y = (kp[:, 1].numpy() + 1) / 2.0 * H
            scatters[col].set_offsets(np.column_stack([kps_x, kps_y]))

            frame_texts[col].set_text(f'f={f_idx}')

        return ims + scatters + frame_texts

    anim = animation.FuncAnimation(
        fig, update, frames=T, interval=120, blit=True,
    )

    out = f"stage2_anim_{target_action}.gif"
    print(f"  Saving animation ({T} frames × {n} clips) ...", end=' ', flush=True)
    anim.save(out, writer=PillowWriter(fps=8), dpi=100)
    plt.close()
    print(f"done")
    print(f"  Animation   -> {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Qualitative Stage 2 test")
    parser.add_argument("--action", type=str, default="tennis_serve",
                        help="Action class to inspect")
    parser.add_argument("--clips", type=int, default=3,
                        help="Number of validation clips to display")
    args = parser.parse_args()

    target_action = args.action
    n_clips       = args.clips

    if target_action not in ACTION2IDX:
        print(f"Unknown action '{target_action}'. Valid options:")
        for a in ACTIONS:
            print(f"  {a}")
        return

    all_ckpts = glob.glob(os.path.join(CFG["ckpt_dir"], "*.ckpt"))
    s2_ckpts  = [c for c in all_ckpts if "stage2" in os.path.basename(c)]
    if not s2_ckpts:
        print("No Stage 2 checkpoints found. Run: python main.py --stage 2")
        return

    latest_ckpt = max(s2_ckpts, key=os.path.getmtime)
    print(f"Checkpoint : {os.path.basename(latest_ckpt)}")
    print(f"Action     : {target_action}")
    print(f"Clips      : {n_clips}\n")

    model = Stage2Module.load_from_checkpoint(latest_ckpt, cfg=CFG)
    model.eval().to(DEVICE)
    model.detector.eval()

    val_ds = VideoSequenceDataset(
        CFG["data_root"], CFG["seq_len"], CFG["img_size"], split="val"
    )

    target_idx = ACTION2IDX[target_action]
    clips      = find_clips(val_ds, target_idx, n=n_clips)

    if not clips:
        print(f"No validation clips found for '{target_action}'.")
        return

    # Run inference on all clips up front
    results = []
    correct_count = 0
    for i, clip in enumerate(clips):
        res = run_inference(model, clip)
        results.append(res)
        pred_idx, pred_conf = res[2], res[3]
        pred_action = IDX2ACTION[pred_idx]
        is_correct  = (pred_idx == target_idx)
        if is_correct:
            correct_count += 1
        status = "CORRECT" if is_correct else f"WRONG  (predicted: {pred_action})"
        print(f"  Clip {i + 1}: {status}  —  confidence {pred_conf:.1f}%")

    print(f"\n  Score: {correct_count}/{len(clips)} clips correct for '{target_action}'\n")

    save_static_grid(clips, results, target_action)
    save_animation(clips, results, target_action)


if __name__ == "__main__":
    main()
