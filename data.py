import os
import cv2
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class VideoFramePairDataset(Dataset):
    """
    Returns pairs of frames (ref, tgt) from the same video,
    plus the action class label.
    Expects Penn Action folder structure:
        data_root/
            frames/
                0001/
                    000001.jpg ...
            labels/
                0001.mat ...
    Adjust load_video_paths() if your dataset is structured differently.
    """
    ACTIONS = [
        "baseball_pitch", "clean_and_jerk", "pullup",  "strum_guitar",
        "baseball_swing", "golf_swing",     "pushup",  "tennis_forehand",
        "bench_press",    "jumping_jacks",  "situp",   "tennis_serve",
        "bowl",           "jump_rope",      "squat",
    ]
    ACTION2IDX = {a: i for i, a in enumerate(ACTIONS)}

    def __init__(self, data_root, img_size=128, split="train",
                 max_gap=8):
        self.img_size = img_size
        self.max_gap  = max_gap          # max temporal distance between pair
        self.samples  = []               # list of (frame_path_1, frame_path_2, action_idx)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),  # → [-1, 1]
        ])

        self._build_samples(data_root, split)

    def _build_samples(self, data_root, split):
        frames_dir = os.path.join(data_root, "frames")
        if not os.path.exists(frames_dir):
            print(f"WARNING: {frames_dir} not found. Using dummy data.")
            return

        video_dirs = sorted(os.listdir(frames_dir))
        
        import random
        rng = random.Random(42)
        rng.shuffle(video_dirs)
        
        n = len(video_dirs)
        video_dirs = video_dirs[:int(n * 0.9)] if split == "train" else video_dirs[int(n * 0.9):]

        from collections import defaultdict
        class_samples = defaultdict(list)

        for vid in video_dirs:
            vid_path = os.path.join(frames_dir, vid)
            frames   = sorted(os.listdir(vid_path))
            if len(frames) < 2:
                continue

            action_idx = 0
            label_path = os.path.join(data_root, "labels", f"{vid}.mat")
            if os.path.exists(label_path):
                mat = scipy.io.loadmat(label_path)
                # mat["action"] is a numpy array of shape (1,) containing a
                # unicode/object sub-array.  We need to unwrap it fully.
                raw_action = mat.get("action")
                if raw_action is not None:
                    # Flatten until we reach a scalar, then convert to str
                    act_val = raw_action.flat[0]
                    # act_val may itself be a numpy array (object dtype)
                    if hasattr(act_val, 'flat'):
                        act_val = act_val.flat[0]
                    act_str = str(act_val).strip()
                    if act_str in self.ACTION2IDX:
                        action_idx = self.ACTION2IDX[act_str]

            max_pairs = min(len(frames) - 1, 20)
            if max_pairs > 0:
                indices = np.linspace(0, len(frames) - 2, max_pairs, dtype=int)
                for i in indices:
                    j = min(i + np.random.randint(1, self.max_gap + 1), len(frames) - 1)
                    class_samples[action_idx].append((
                        os.path.join(vid_path, frames[i]),
                        os.path.join(vid_path, frames[j]),
                        action_idx,
                    ))

        if class_samples:
            max_count = max(len(samples) for samples in class_samples.values())
            for act_idx, samples in class_samples.items():
                if len(samples) < max_count:
                    # Oversample minority classes to achieve even distribution
                    extras = rng.choices(samples, k=max_count - len(samples))
                    self.samples.extend(samples + extras)
                else:
                    self.samples.extend(samples)
            rng.shuffle(self.samples)

        print(f"Built {len(self.samples)} frame pairs from {len(video_dirs)} videos (balanced to {max_count} per class)")

    def __len__(self):
        return max(len(self.samples), 100)   # fallback for dummy mode

    def __getitem__(self, idx):
        if not self.samples:
            # Dummy mode — useful for testing pipeline without data
            H = W = self.img_size
            return {
                "ref"    : torch.randn(3, H, W),
                "tgt"    : torch.randn(3, H, W),
                "action" : torch.tensor(0, dtype=torch.long),
            }

        ref_path, tgt_path, action_idx = self.samples[idx % len(self.samples)]
        ref = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
        tgt = cv2.cvtColor(cv2.imread(tgt_path), cv2.COLOR_BGR2RGB)

        return {
            "ref"    : self.transform(ref),
            "tgt"    : self.transform(tgt),
            "action" : torch.tensor(action_idx, dtype=torch.long),
        }

class VideoSequenceDataset(Dataset):
    """
    Returns a fixed-length sequence of frames from a video, with the following
    improvements over the naive approach:
      - Random temporal window sampling (instead of always taking the first N frames)
      - Multiple clips per video (clips_per_video) to multiply effective dataset size
      - Short videos are padded by looping so nothing is discarded
      - Consistent spatial augmentation across all frames in a clip (train split only)
    """
    def __init__(self, data_root, seq_len=32, img_size=128, split="train",
                 clips_per_video=3):
        self.img_size        = img_size
        self.seq_len         = seq_len
        self.split           = split
        self.clips_per_video = clips_per_video if split == "train" else 1
        # samples: list of (list_of_all_frame_paths, action_idx)
        # The actual clip window is sampled randomly at __getitem__ time (train)
        # or fixed to a center window (val).
        self.samples = []

        # Base transforms (applied individually to each frame)
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

        self._build_samples(data_root, split)

    # ------------------------------------------------------------------
    # Consistent spatial augmentation helpers
    # ------------------------------------------------------------------
    def _get_aug_params(self):
        """Sample a single set of augmentation parameters to apply
        consistently across every frame in a clip."""
        import random
        flip   = random.random() < 0.5
        # ColorJitter params: brightness, contrast, saturation, hue
        brightness = 1.0 + random.uniform(-0.2, 0.2)
        contrast   = 1.0 + random.uniform(-0.2, 0.2)
        saturation = 1.0 + random.uniform(-0.2, 0.2)
        hue        = random.uniform(-0.05, 0.05)
        return dict(flip=flip, brightness=brightness,
                    contrast=contrast, saturation=saturation, hue=hue)

    def _apply_aug(self, pil_img, aug_params):
        """Apply pre-sampled augmentation to a single PIL image."""
        import torchvision.transforms.functional as TF
        if aug_params["flip"]:
            pil_img = TF.hflip(pil_img)
        pil_img = TF.adjust_brightness(pil_img, aug_params["brightness"])
        pil_img = TF.adjust_contrast(pil_img,   aug_params["contrast"])
        pil_img = TF.adjust_saturation(pil_img, aug_params["saturation"])
        pil_img = TF.adjust_hue(pil_img,        aug_params["hue"])
        return pil_img

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------
    def _build_samples(self, data_root, split):
        frames_dir = os.path.join(data_root, "frames")
        if not os.path.exists(frames_dir):
            print(f"WARNING: {frames_dir} not found. Using dummy data.")
            return

        video_dirs = sorted(os.listdir(frames_dir))

        import random
        rng = random.Random(42)
        rng.shuffle(video_dirs)

        n = len(video_dirs)
        video_dirs = (video_dirs[:int(n * 0.9)] if split == "train"
                      else video_dirs[int(n * 0.9):])

        skipped = 0
        for vid in video_dirs:
            vid_path = os.path.join(frames_dir, vid)
            frames   = sorted(os.listdir(vid_path))

            # Need at least 2 frames; short clips are padded in __getitem__
            if len(frames) < 2:
                skipped += 1
                continue

            action_idx = 0
            label_path = os.path.join(data_root, "labels", f"{vid}.mat")
            if os.path.exists(label_path):
                mat = scipy.io.loadmat(label_path)
                raw_action = mat.get("action")
                if raw_action is not None:
                    act_val = raw_action.flat[0]
                    if hasattr(act_val, 'flat'):
                        act_val = act_val.flat[0]
                    act_str = str(act_val).strip()
                    if act_str in VideoFramePairDataset.ACTION2IDX:
                        action_idx = VideoFramePairDataset.ACTION2IDX[act_str]

            all_frame_paths = [os.path.join(vid_path, f) for f in frames]
            # Store one entry per clip; the clip start is sampled at getitem time
            for _ in range(self.clips_per_video):
                self.samples.append((all_frame_paths, action_idx))

        print(f"Built {len(self.samples)} video sequences "
              f"({len(video_dirs) - skipped} videos × up to {self.clips_per_video} clips; "
              f"{skipped} videos skipped)")

    # ------------------------------------------------------------------
    def __len__(self):
        return max(len(self.samples), 100)

    def __getitem__(self, idx):
        if not self.samples:
            H = W = self.img_size
            return {
                "frames": torch.randn(self.seq_len, 3, H, W),
                "action": torch.tensor(0, dtype=torch.long),
            }

        all_frame_paths, action_idx = self.samples[idx % len(self.samples)]
        n_frames = len(all_frame_paths)

        # ── Temporal window selection ─────────────────────────────────
        if n_frames <= self.seq_len:
            # Short video: use all frames, pad by looping if needed
            selected = all_frame_paths
        else:
            if self.split == "train":
                # Random start within the valid range
                import random
                start = random.randint(0, n_frames - self.seq_len)
            else:
                # Validation: always take a centered window for reproducibility
                start = (n_frames - self.seq_len) // 2
            selected = all_frame_paths[start: start + self.seq_len]

        # ── Pad to seq_len by looping ─────────────────────────────────
        while len(selected) < self.seq_len:
            selected = (selected + selected)[:self.seq_len]

        # ── Spatial augmentation params (sampled once per clip) ───────
        aug_params = self._get_aug_params() if self.split == "train" else None

        # ── Load and transform frames ─────────────────────────────────
        frames_tensor = []
        for path in selected:
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            pil = self.base_transform.transforms[0](img)   # ToPILImage
            pil = self.base_transform.transforms[1](pil)   # Resize
            if aug_params is not None:
                pil = self._apply_aug(pil, aug_params)
            # ToTensor + Normalize
            t = self.base_transform.transforms[2](pil)     # ToTensor
            t = self.base_transform.transforms[3](t)       # Normalize
            frames_tensor.append(t)

        return {
            "frames": torch.stack(frames_tensor, dim=0),
            "action": torch.tensor(action_idx, dtype=torch.long),
        }


