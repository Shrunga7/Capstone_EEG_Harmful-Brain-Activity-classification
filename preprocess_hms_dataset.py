import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
TRAIN_CSV = "C:\\AIG_Sem2\\CapstoneProject\\Model_Training\\train.csv"
VAL_CSV   = "C:\\AIG_Sem2\\CapstoneProject\\Model_Training\\val.csv"

TRAIN_PARQUET_DIR = "C:\\AIG_Sem2\\CapstoneProject\\Model_Training\\train_spectrograms"
VAL_PARQUET_DIR   = "C:\\AIG_Sem2\\CapstoneProject\\Model_Training\\val_spectrograms"

OUT_ROOT = "C:\\AIG_Sem2\\CapstoneProject\\Model_Training\\preprocessed"

IMG_HEIGHT = 128
IMG_WIDTH = 256

LABEL_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote"
]

# -----------------------------
# HELPERS
# -----------------------------
def preprocess_spectrogram(parquet_path: str, img_height: int = 128, img_width: int = 256):
    """
    Reads one parquet spectrogram and returns a processed float32 array
    of shape (1, 128, 256).
    """
    spectrogram = pd.read_parquet(parquet_path).values  # shape usually (time, freq)

    # 1) NaN handling + log scale
    spectrogram = np.nan_to_num(spectrogram)
    spectrogram = np.log1p(spectrogram)

    # 2) Crop/pad time dimension to fixed width
    desired_time = img_width
    current_time = spectrogram.shape[0]

    if current_time > desired_time:
        start = (current_time - desired_time) // 2
        spectrogram = spectrogram[start:start + desired_time, :]
    else:
        pad_needed = desired_time - current_time
        spectrogram = np.pad(spectrogram, ((0, pad_needed), (0, 0)), mode="constant")

    # 3) Resize to (time=256, freq=128) before transpose
    spectrogram = cv2.resize(spectrogram, (img_height, img_width))

    # 4) Standardize
    mean = spectrogram.mean()
    std = spectrogram.std() + 1e-6
    spectrogram = (spectrogram - mean) / std

    # 5) Final shape (1, 128, 256)
    spectrogram = spectrogram.T.astype(np.float32)   # (128, 256)
    spectrogram = np.expand_dims(spectrogram, axis=0)  # (1, 128, 256)

    return spectrogram


def process_split(input_csv, parquet_dir, out_split_dir, out_csv):
    os.makedirs(out_split_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    processed_rows = []
    failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(input_csv)}"):
        spec_id = row["spectrogram_id"]
        parquet_path = os.path.join(parquet_dir, f"{spec_id}.parquet")
        npy_path = os.path.join(out_split_dir, f"{spec_id}.npy")

        try:
            arr = preprocess_spectrogram(parquet_path)
            np.save(npy_path, arr)

            label_values = row[LABEL_COLS].values.astype(np.float32)
            label_values = label_values / (label_values.sum() + 1e-6)

            new_row = {
                "spectrogram_id": spec_id,
                "npy_path": npy_path,
            }

            for col, val in zip(LABEL_COLS, label_values):
                new_row[col] = float(val)

            processed_rows.append(new_row)

        except Exception as e:
            failed += 1
            print(f"Failed: {spec_id} | {e}")

    processed_df = pd.DataFrame(processed_rows)
    processed_df.to_csv(out_csv, index=False)

    print(f"\nSaved: {out_csv}")
    print(f"Successful: {len(processed_rows)}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    train_out_dir = os.path.join(OUT_ROOT, "train_npy")
    val_out_dir = os.path.join(OUT_ROOT, "val_npy")

    train_out_csv = os.path.join(OUT_ROOT, "train_preprocessed.csv")
    val_out_csv = os.path.join(OUT_ROOT, "val_preprocessed.csv")

    process_split(TRAIN_CSV, TRAIN_PARQUET_DIR, train_out_dir, train_out_csv)
    process_split(VAL_CSV, VAL_PARQUET_DIR, val_out_dir, val_out_csv)

    print("\nDone preprocessing both train and val.")