import gc
import os

import numpy as np
import onnxruntime as ort
import soundfile as sf
from tqdm import tqdm


def _load_session(onnx_path):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    out_map = {o.name: i for i, o in enumerate(session.get_outputs())}
    return session, input_name, out_map


def _run_batched(session, input_name, out_map, audio_batch, max_batch_size=16):
    num_windows = audio_batch.shape[0]
    chunks = []
    for start_idx in range(0, num_windows, max_batch_size):
        sub_batch = audio_batch[start_idx : start_idx + max_batch_size]
        outs = session.run(None, {input_name: sub_batch})
        chunks.append(outs[out_map["embedding"]])
    return np.concatenate(chunks, axis=0)


def extract_baseline_embeddings(df, onnx_path, audio_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    session, input_name, out_map = _load_session(onnx_path)
    unique_files = df["filename"].unique()
    for filename in tqdm(unique_files, desc="baseline embeddings"):
        audio_path = os.path.join(audio_dir, filename)
        safe_name = filename.replace("/", "_").replace(".ogg", ".npy")
        save_path = os.path.join(save_dir, safe_name)
        if os.path.exists(save_path):
            continue
        try:
            audio_data, _ = sf.read(audio_path, dtype="float32")
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            window_length = 5 * 32000
            num_windows = len(audio_data) // window_length
            if num_windows == 0:
                pad_length = window_length - len(audio_data)
                audio_data = np.pad(audio_data, (0, pad_length), mode="constant")
                num_windows = 1
            else:
                audio_data = audio_data[: num_windows * window_length]
            audio_batch = audio_data.reshape(num_windows, window_length)
            embeddings = _run_batched(session, input_name, out_map, audio_batch)
            np.save(save_path, embeddings)
            del audio_data, audio_batch, embeddings
            gc.collect()
        except Exception as exc:
            print(f"Error processing {filename}: {exc}")


def extract_tta_embeddings(df, onnx_path, audio_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    session, input_name, out_map = _load_session(onnx_path)
    unique_files = df["filename"].unique()
    for filename in tqdm(unique_files, desc="tta embeddings"):
        audio_path = os.path.join(audio_dir, filename)
        safe_name = filename.replace("/", "_").replace(".ogg", ".npy")
        save_path = os.path.join(save_dir, safe_name)
        if os.path.exists(save_path):
            continue
        try:
            audio_data, _ = sf.read(audio_path, dtype="float32")
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            window_length = 5 * 32000
            stride_length = int(2.5 * 32000)
            windows = []
            start_idx = 0
            while start_idx < len(audio_data):
                end_idx = start_idx + window_length
                if end_idx <= len(audio_data):
                    windows.append(audio_data[start_idx:end_idx])
                else:
                    tail = audio_data[start_idx:]
                    if len(tail) > 0:
                        pad_length = window_length - len(tail)
                        windows.append(np.pad(tail, (0, pad_length), mode="constant"))
                start_idx += stride_length
            audio_batch = np.stack(windows)
            embeddings = _run_batched(session, input_name, out_map, audio_batch)
            np.save(save_path, embeddings)
            del audio_data, audio_batch, embeddings
            gc.collect()
        except Exception as exc:
            print(f"Error processing {filename}: {exc}")
