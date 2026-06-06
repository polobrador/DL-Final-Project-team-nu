import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class LazyContextBirdDataset(Dataset):
    def __init__(self, df, context_size=1):
        self.index_map = []
        self.context_size = context_size
        self.pad_size = context_size // 2
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"context={context_size}"):
            npy_path = row["npy_path"]
            label = row["label_id"]
            try:
                shape = np.load(npy_path, mmap_mode="r").shape
                num_windows = shape[0]
                for i in range(num_windows):
                    self.index_map.append((npy_path, i, label))
            except Exception:
                continue

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        npy_path, window_idx, label = self.index_map[idx]
        embeddings = np.load(npy_path, mmap_mode="r")
        if self.pad_size > 0:
            padding = np.zeros((self.pad_size, embeddings.shape[1]), dtype=embeddings.dtype)
            padded_emb = np.vstack([padding, embeddings, padding])
        else:
            padded_emb = embeddings
        context_window = padded_emb[window_idx : window_idx + self.context_size]
        flat_context = context_window.flatten().astype(np.float32)
        return torch.tensor(flat_context), torch.tensor(label, dtype=torch.long)
