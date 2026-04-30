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
        "baseball_pitch", "clean_and_jerk", "pull_ups",
        "baseball_swing", "golf_swing", "tennis_forehand",
        "jumping_jacks",  "tennis_serve",  "squats",
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

        for vid in video_dirs:
            vid_path = os.path.join(frames_dir, vid)
            frames   = sorted(os.listdir(vid_path))
            if len(frames) < 2:
                continue

            action_idx = 0
            label_path = os.path.join(data_root, "labels", f"{vid}.mat")
            if os.path.exists(label_path):
                mat = scipy.io.loadmat(label_path)
                act_str = str(mat.get("action", [""])[0])
                if act_str in self.ACTION2IDX:
                    action_idx = self.ACTION2IDX[act_str]

            max_pairs = min(len(frames) - 1, 20)
            for i in range(max_pairs):
                j = min(i + np.random.randint(1, self.max_gap + 1), len(frames) - 1)
                self.samples.append((
                    os.path.join(vid_path, frames[i]),
                    os.path.join(vid_path, frames[j]),
                    action_idx,
                ))

        print(f"Built {len(self.samples)} frame pairs from {len(video_dirs)} videos")

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
    def __init__(self, data_root, seq_len=32, img_size=128, split="train"):
        self.img_size = img_size
        self.seq_len  = seq_len
        self.samples  = [] # list of (list_of_frame_paths, action_idx)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
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

        for vid in video_dirs:
            vid_path = os.path.join(frames_dir, vid)
            frames   = sorted(os.listdir(vid_path))
            if len(frames) < self.seq_len:
                continue

            action_idx = 0
            label_path = os.path.join(data_root, "labels", f"{vid}.mat")
            if os.path.exists(label_path):
                mat = scipy.io.loadmat(label_path)
                act_str = str(mat.get("action", [""])[0])
                if act_str in VideoFramePairDataset.ACTION2IDX:
                    action_idx = VideoFramePairDataset.ACTION2IDX[act_str]

            frame_paths = [os.path.join(vid_path, f) for f in frames[:self.seq_len]]
            self.samples.append((frame_paths, action_idx))

        print(f"Built {len(self.samples)} video sequences")

    def __len__(self):
        return max(len(self.samples), 100)

    def __getitem__(self, idx):
        if not self.samples:
            H = W = self.img_size
            return {
                "frames" : torch.randn(self.seq_len, 3, H, W),
                "action" : torch.tensor(0, dtype=torch.long),
            }

        frame_paths, action_idx = self.samples[idx % len(self.samples)]
        
        frames_tensor = []
        for path in frame_paths:
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            frames_tensor.append(self.transform(img))
            
        return {
            "frames" : torch.stack(frames_tensor, dim=0),
            "action" : torch.tensor(action_idx, dtype=torch.long),
        }
