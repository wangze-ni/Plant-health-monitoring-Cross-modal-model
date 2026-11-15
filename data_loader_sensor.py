import io
import re
import csv
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset
from config import PERIOD_TO_INDEX


# Dense overlapping windows improve cycle-level voting without creating exact
# duplicate records. With 5-fold cycle_window split, each fold validates on
# three windows from every physical response cycle and trains on the other 12.
RESPONSE_WINDOW_STARTS = (
    300, 320, 340, 360, 380,
    400, 420, 440, 460, 480,
    500, 520, 540, 560, 580,
)
RESPONSE_WINDOW_LEN = 600
COMMON_BASELINE_LEN = 80


def parse_name(path_in_zip):
    base = path_in_zip.split("/")[-1]
    m = re.match(r"([A-Za-z]+)_(\d+)h_(\d+)\.csv$", base)
    if not m:
        return None

    kind = m.group(1).lower()
    hour = int(m.group(2))
    file_fold = int(m.group(3))

    if "drought" in kind:
        stress = 0
    elif "salt" in kind:
        stress = 1
    elif "heat" in kind:
        stress = 2
    else:
        return None

    return base, hour, file_fold, stress


def load_records_from_zip(zip_path, start_row, target_len):
    records = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in [n for n in zf.namelist() if n.lower().endswith(".csv")]:
            meta = parse_name(name)
            if not meta:
                continue

            base, hour, file_fold, stress = meta

            with zf.open(name, "r") as fb:
                reader = csv.reader(io.TextIOWrapper(fb, encoding="utf-8-sig"))
                header = next(reader)
                cols = [i for i, c in enumerate(header) if c.startswith("Chan")]
                if len(cols) != 8:
                    continue

                rows = []
                for ridx, row in enumerate(reader):
                    if ridx < start_row - 1:
                        continue
                    rows.append([float(row[j]) if row[j] else np.nan for j in cols])

            if len(rows) < 20:
                continue

            X = np.asarray(rows, dtype=np.float32).T

            for ch in range(8):
                valid = np.isfinite(X[ch])
                if 0 < valid.sum() < X.shape[1]:
                    X[ch, ~valid] = np.interp(
                        np.arange(X.shape[1])[~valid],
                        np.arange(X.shape[1])[valid],
                        X[ch, valid],
                    )
                elif valid.sum() == 0:
                    X[ch] = 0

            # Split one source CSV into 5 gas-response cycles.
            seg_len = X.shape[1] // 5
            for seg_id, seg in enumerate(np.split(X[:, :seg_len * 5], 5, axis=1)):
                baseline_end = RESPONSE_WINDOW_STARTS[0]
                baseline_start = max(0, baseline_end - COMMON_BASELINE_LEN)
                common_r0 = seg[:, baseline_start:baseline_end].mean(axis=1, keepdims=True)

                for window_idx, window_start in enumerate(RESPONSE_WINDOW_STARTS):
                    window_end = window_start + RESPONSE_WINDOW_LEN
                    sub_seg = seg[:, window_start:window_end]
                    sub_seg_len = sub_seg.shape[1]

                    if sub_seg_len < 2:
                        continue

                    # Use one common baseline for all windows from this response
                    # cycle, so overlapping windows remain comparable.
                    sub_seg = (sub_seg - common_r0) / (np.abs(common_r0) + 1e-6)

                    arr = np.vstack([
                        np.interp(
                            np.linspace(0, 1, target_len),
                            np.linspace(0, 1, sub_seg_len),
                            sub_seg[ch],
                        )
                        for ch in range(8)
                    ])

                    records.append({
                        "x": arr.astype(np.float32),
                        "stress": stress,
                        "period_idx": PERIOD_TO_INDEX[hour],
                        "group_id": base,
                        "source_file": base,
                        "replicate_id": file_fold,
                        "hour": hour,
                        "segment_idx": seg_id,
                        "window_idx": window_idx,
                        "window_start": window_start,
                        "window_end": window_start + sub_seg_len,
                        "cv_fold": -1,
                    })

    return records


class GasResponseDataset(Dataset):
    def __init__(self, records, mean=None, std=None):
        self.records = records
        X = np.stack([r["x"] for r in records], axis=0)
        self.mean = X.mean(axis=(0, 2), keepdims=True) if mean is None else mean
        self.std = X.std(axis=(0, 2), keepdims=True) + 1e-6 if std is None else std

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        x = (self.records[idx]["x"] - self.mean[0]) / self.std[0]
        return (
            torch.from_numpy(x).float(),
            torch.tensor(self.records[idx]["stress"], dtype=torch.long),
            torch.tensor(self.records[idx]["period_idx"], dtype=torch.long),
        )
