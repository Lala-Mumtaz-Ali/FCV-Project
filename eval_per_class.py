"""
eval_per_class.py — Per-class validation accuracy for Stage 2.

Usage:
    python eval_per_class.py                      # auto-finds latest stage2 checkpoint
    python eval_per_class.py --ckpt path/to.ckpt  # specific checkpoint
    python eval_per_class.py --no-plot            # skip saving the bar chart
"""

import os
import sys
import glob
import argparse

sys.stdout.reconfigure(encoding="utf-8")

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import CFG, DEVICE
from data import VideoSequenceDataset, VideoFramePairDataset
from train import Stage2Module

ACTIONS = VideoFramePairDataset.ACTIONS


def find_latest_stage2():
    ckpts = glob.glob(os.path.join(CFG["ckpt_dir"], "*.ckpt"))
    tagged = [c for c in ckpts if "stage2" in os.path.basename(c)]
    return max(tagged, key=os.path.getmtime) if tagged else None


def run(ckpt_path, save_plot=True):
    print(f"\n{'='*60}")
    print(f"  Per-Class Validation Accuracy — Stage 2")
    print(f"{'='*60}")
    print(f"  Checkpoint : {os.path.basename(ckpt_path)}")
    print(f"  Device     : {DEVICE}\n")

    model = Stage2Module.load_from_checkpoint(ckpt_path, cfg=CFG).to(DEVICE)
    model.eval()
    model.detector.eval()

    val_ds = VideoSequenceDataset(
        CFG["data_root"], CFG["seq_len"], CFG["img_size"], split="val"
    )
    val_dl = DataLoader(
        val_ds, batch_size=16, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    print(f"  Validation set: {len(val_ds)} samples")
    print(f"  Running inference...\n")

    all_preds, all_labels = [], []
    top3_hits = []
    all_top3_indices = []

    with torch.no_grad():
        for batch in val_dl:
            frames = batch["frames"].to(DEVICE)
            labels = batch["action"].to(DEVICE)

            B, T, C, H, W = frames.shape
            keypoints, _ = model.detector(frames.view(B * T, C, H, W))
            keypoints = keypoints.view(B, T, CFG["K"], 2)

            logits = model.classifier(keypoints)
            probs  = torch.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1)

            k = min(3, CFG["num_actions"])
            top3_idx = probs.topk(k, dim=1).indices
            top3 = (top3_idx == labels.unsqueeze(1)).any(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            top3_hits.extend(top3.cpu().tolist())
            all_top3_indices.extend(top3_idx.cpu().tolist())

    all_preds       = np.array(all_preds)
    all_labels      = np.array(all_labels)
    all_top3_indices = np.array(all_top3_indices)  # (N, 3)

    top1_acc = (all_preds == all_labels).mean() * 100
    top3_acc = np.mean(top3_hits) * 100

    # ── Per-class breakdown ───────────────────────────────────────────────────
    results = {}
    for idx, name in enumerate(ACTIONS):
        mask = all_labels == idx
        total      = int(mask.sum())
        correct1   = int((all_preds[mask] == idx).sum()) if total > 0 else 0
        correct3   = int((all_top3_indices[mask] == idx).any(axis=1).sum()) if total > 0 else 0
        acc1       = correct1 / total * 100 if total > 0 else 0.0
        acc3       = correct3 / total * 100 if total > 0 else 0.0
        results[name] = {"correct": correct1, "correct3": correct3,
                         "total": total, "acc": acc1, "acc3": acc3}

    # ── Console table ─────────────────────────────────────────────────────────
    print(f"  {'Class':<24}  {'Top-1':>6}  {'Top-3':>6}  {'Total':>6}")
    print(f"  {'-'*55}")

    for name in ACTIONS:
        r = results[name]
        flag = "  [no data]" if r["total"] == 0 else ""
        print(f"  {name:<24}  {r['acc']:>5.1f}%  {r['acc3']:>5.1f}%  {r['total']:>6}{flag}")

    print(f"  {'-'*65}")
    print(f"\n  Overall  Top-1: {top1_acc:.2f}%   Top-3: {top3_acc:.2f}%")

    best  = max(results, key=lambda n: (results[n]["acc"], results[n]["total"]))
    worst = min(results, key=lambda n: (results[n]["acc"] if results[n]["total"] > 0 else 101,
                                        -results[n]["total"]))
    spread = results[best]["acc"] - results[worst]["acc"]

    print(f"\n  Best  class : {best:<24} {results[best]['acc']:.1f}%")
    print(f"  Worst class : {worst:<24} {results[worst]['acc']:.1f}%")
    print(f"  Spread      : {spread:.1f} pp")
    if spread > 50:
        print(f"  [WARN] Large spread — some classes are significantly harder.")
    print(f"{'='*60}\n")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    if save_plot:
        _save_chart(results, top1_acc, top3_acc)

    return results, top1_acc, top3_acc


def _save_chart(results, top1_acc, top3_acc):
    names  = list(ACTIONS)
    accs1  = [results[n]["acc"]   for n in names]
    accs3  = [results[n]["acc3"]  for n in names]
    totals = [results[n]["total"] for n in names]

    n_classes = len(names)
    y = np.arange(n_classes)
    bar_h = 0.38

    fig, ax = plt.subplots(figsize=(13, 7))

    # Top-3 bars (behind, slightly offset up)
    bars3 = ax.barh(y + bar_h / 2, accs3, height=bar_h,
                    color="#5dade2", edgecolor="white", label="Top-3", alpha=0.85)
    # Top-1 bars (in front, slightly offset down)
    bars1 = ax.barh(y - bar_h / 2, accs1, height=bar_h,
                    color=[
                        "#2ecc71" if a >= 70 else "#f39c12" if a >= 40 else "#e74c3c"
                        for a in accs1
                    ], edgecolor="white", label="Top-1")

    mean1 = float(np.mean([a for a, t in zip(accs1, totals) if t > 0]))
    mean3 = float(np.mean([a for a, t in zip(accs3, totals) if t > 0]))
    ax.axvline(mean1, color="navy",  linestyle="--", linewidth=1.2, label=f"Top-1 mean {mean1:.1f}%")
    ax.axvline(mean3, color="#1a6fa8", linestyle=":",  linewidth=1.2, label=f"Top-3 mean {mean3:.1f}%")

    # Labels on bars
    for bar, acc in zip(bars1, accs1):
        ax.text(acc + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%", va="center", fontsize=7.5)
    for bar, acc, total in zip(bars3, accs3, totals):
        ax.text(acc + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%  (n={total})", va="center", fontsize=7.5)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Accuracy (%)", fontsize=11)
    ax.set_title(
        f"Stage 2 — Per-Class Validation Accuracy\n"
        f"Top-1: {top1_acc:.1f}%     Top-3: {top3_acc:.1f}%",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.invert_yaxis()
    plt.tight_layout()

    out = "eval_per_class.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {out}\n")


def main():
    parser = argparse.ArgumentParser(description="Per-class Stage 2 validation accuracy")
    parser.add_argument("--ckpt",    type=str, default=None,
                        help="Checkpoint path (default: latest stage2 checkpoint)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip saving the bar chart PNG")
    args = parser.parse_args()

    ckpt = args.ckpt or find_latest_stage2()
    if ckpt is None:
        print(f"\nNo stage2 checkpoint found in '{CFG['ckpt_dir']}'.")
        print("Train first:  python main.py --stage 2")
        sys.exit(1)

    torch.set_grad_enabled(False)
    run(ckpt, save_plot=not args.no_plot)


if __name__ == "__main__":
    main()
