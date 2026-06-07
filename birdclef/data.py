import os

import pandas as pd
from sklearn.model_selection import GroupKFold

from birdclef import paths


def get_npy_path(ogg_filename, embedding_dir):
    safe_name = ogg_filename.replace("/", "_").replace(".ogg", ".npy")
    return os.path.join(embedding_dir, safe_name)


def load_training_frame(embedding_dir=None):
    if embedding_dir is None:
        embedding_dir = paths.EMBEDDINGS_DIR
    df = pd.read_csv(paths.TRAIN_CSV)
    sample_sub = pd.read_csv(paths.SAMPLE_SUBMISSION_CSV)
    all_birds = sample_sub.columns.tolist()[1:]
    bird_to_idx = {bird: i for i, bird in enumerate(all_birds)}
    num_classes = len(all_birds)
    df["label_id"] = df["primary_label"].map(bird_to_idx)
    df["npy_path"] = df["filename"].apply(lambda f: get_npy_path(f, embedding_dir))
    df["file_exists"] = df["npy_path"].apply(os.path.exists)
    df = df[df["file_exists"]].reset_index(drop=True)
    return df, num_classes, bird_to_idx


def assign_group_kfold(df, n_splits=5):
    df = df.sort_values(by=["filename"]).reset_index(drop=True)
    gkf = GroupKFold(n_splits=n_splits)
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(gkf.split(df, groups=df["filename"])):
        df.loc[val_idx, "fold"] = fold
    return df


def prepare_baseline_data():
    df, num_classes, bird_to_idx = load_training_frame(paths.EMBEDDINGS_DIR)
    df = assign_group_kfold(df)
    return df, num_classes, bird_to_idx


def prepare_tta_data():
    df, num_classes, bird_to_idx = load_training_frame(paths.EMBEDDINGS_TTA_DIR)
    df = assign_group_kfold(df)
    return df, num_classes, bird_to_idx
