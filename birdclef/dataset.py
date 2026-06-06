import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class BirdDataset(Dataset):
    def __init__(self, df, is_tta=False):
        self.samples = []
        self.is_tta = is_tta
        for _, row in tqdm(df.iterrows(), total=len(df)):
            npy_path = row["npy_path"]
            label = row["label_id"]
            try:
                embeddings = np.load(npy_path)
                if self.is_tta:
                    for emb in embeddings:
                        self.samples.append((emb, label))
                else:
                    if embeddings.ndim > 1:
                        self.samples.append((embeddings[0], label))
                    else:
                        self.samples.append((embeddings, label))
            except Exception:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        emb, label = self.samples[idx]
        return torch.tensor(emb, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
