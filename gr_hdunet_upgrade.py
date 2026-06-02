from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / f"matplotlib-{os.getuid()}"))

import cv2
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.feature import hog as sk_hog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models
from tqdm.auto import tqdm


# =============================================================================
# ALL REQUIRED PATHS AND EXPERIMENT SETTINGS — KEEP EVERYTHING HERE
# =============================================================================
CONFIG = {
    # ---------- paths ----------
    "dataset_root": "/data/pc/EBHI-SEG",
    "output_root": "./results0",
    "run_name": "paper_run",
    "log_filename": "run_log.txt",

    # ---------- class names ----------
    # Folder names must match your dataset exactly.
    "class_folders": [
        "Adenocarcinoma",
        "High-grade IN",
        "Low-grade IN",
        "Normal",
        "Polyp",
        "Serrated adenoma",
    ],

    # ---------- strict reproduction checks ----------
    "expected_total_images": 2226, #2855 original
    "expected_num_classes": 6,
    "image_ext": ".png",
    "mask_ext": ".png",
    "require_cuda": True,

    # ---------- paper-faithful geometry ----------
    # The paper is internally inconsistent: Algorithm 1 says segmentation input is 256x256,
    # while the GAN paragraph says training/inference are at 224x224. We follow the
    # training/inference description here to stay closer to the written Methods section.
    "seg_image_size": 224,
    "cls_image_size": 224,

    # ---------- split ----------
    "seed": 42,
    "train_ratio": 0.70,
    "val_ratio": 0.10,
    "test_ratio": 0.20,
    # "case" keeps all tiles from the same source case in one split.
    # EBHI-Seg filenames look like GT2001837-1-400-001.png, where the last
    # numeric token is the tile/image number. The case id is therefore
    # GT2001837-1-400. Use "image" only when intentionally reproducing the old
    # image-wise split.
    "split_strategy": "case",
    "case_id_regex": r"^(.*)-[0-9]+$",
    "group_split_attempts": 750,
    "group_split_size_weight": 0.25,
    "active_split_fingerprint": None,

    # ---------- preprocessing ----------
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid_size": (8, 8),
    "randstain_std_hyper": 0.05,
    "randstain_probability_train": 0.50,
    "randstain_probability_eval": 0.00,
    "seg_imagenet_normalize": True,
    "cls_imagenet_normalize": True,
    "cls_mask_background_alpha": 0.20,
    "cls_mask_blur_kernel": 7,

    # ---------- proposed segmentation ----------
    "seg_epochs": 100,
    "seg_batch_size": 16,
    "seg_lr": 2e-4,
    "seg_beta1": 0.5,
    "seg_beta2": 0.999,
    "seg_lambda_gan": 0.25,
    "seg_lambda_dice": 1.0,
    "seg_lambda_bce": 0.20,
    "seg_lambda_boundary": 0.10,
    "seg_threshold": 0.5,
    "seg_disc_lr_scale": 0.5,
    "seg_disc_update_interval": 2,
    "seg_early_stopping_patience": 20,
    "seg_early_stopping_min_delta": 1e-4,
    "seg_eval_tta": True,
    "seg_tune_inference_on_val": True,
    "seg_threshold_candidates": [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
    "seg_postprocess_close_kernels": [0, 3, 5],
    "seg_postprocess_open_kernels": [0, 3],
    "seg_selection_weights": {
        "Dice": 0.45,
        "Jaccard": 0.20,
        "BF-score": 0.25,
        "HOG-sim": 0.10,
    },

    # ---------- baseline segmentation ----------
    "baseline_seg_epochs": 100,
    "baseline_seg_batch_size": 16,
    "baseline_seg_lr": 2e-4,
    "baseline_seg_early_stopping_patience": 20,
    "baseline_seg_early_stopping_min_delta": 1e-4,

    # ---------- classification ----------
    "cls_epochs": 50,
    "cls_batch_size": 48,
    "cls_lr": 1e-4,
    "cls_weight_decay": 1e-4,
    "cls_early_stopping_patience": 10,
    "cls_early_stopping_min_delta": 1e-4,
    "cls_use_class_weights": True,
    "cls_use_weighted_sampler": False,
    "ensemble_weights": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    "ensemble_selection_metric": "weighted_f1",
    "ensemble_weight_search_step": 0.05,
    "ensemble_fit_logreg": True,
    "ensemble_logreg_c": 1.0,

    # ---------- dataloader ----------
    "num_workers": 16,
    "pin_memory": True,
    "prefetch_factor": 6,
    "persistent_workers": True,

    # ---------- safe performance helpers ----------
    # These caches only memoize deterministic steps and should not change results.
    "enable_decode_cache": True,
    "enable_disk_decode_cache": True,
    "enable_eval_sample_cache": True,
    "warmup_decode_cache": True,

    # ---------- checkpointing ----------
    "save_top_k": 2,
    "save_every_epoch": True,
    "resume": True,

    # ---------- compute ----------
    "use_all_gpus": True,
    "gpu_parallel_mode": "auto",
    "data_parallel_min_batch_per_gpu": 32,
    "amp": True,
    "proposed_seg_amp": False,
    "float32_matmul_precision": "high",

    # ---------- fixed architecture assumptions where paper is silent ----------
    "assumptions": {
        "segmentation_encoder": "DenseNet121 pretrained on ImageNet",
        "vit_model": "vit_base_patch16_224",
        "swin_model": "swin_base_patch4_window7_224",
        "convnext_model": "convnext_base.fb_in22k_ft_in1k",
        "medt_baseline": "Single-file axial-attention MedT-like baseline because exact MedT variant is not specified in the paper",
        "classifier_mask_guidance": "RGB image blended with refined mask guidance while retaining low-weight background context, then resized to 224x224",
    },
}

# =============================================================================
# NO MORE PATH DECLARATIONS BELOW THIS LINE
# =============================================================================


# -----------------------------------------------------------------------------
# strict assumptions manifest
# -----------------------------------------------------------------------------
ASSUMPTIONS = {
    "Paper says": [
        "Segmentation uses DenseNet encoder + multi-scale attention + residual decoder + cGAN refinement.",
        "Classification uses ViT + Swin Transformer + ConvNeXt with weighted averaging and majority voting.",
        "Segmentation is evaluated with Dice, Jaccard, Conformity Coefficient, BF-score, HOG-similarity.",
        "Classification is evaluated with Accuracy, Precision, Recall, F1-score, AUC.",
        "Dataset is EBHI-Seg with 2855 image/mask pairs and 70/10/20 split.",
    ],
    "This script assumes where the paper is silent": list(CONFIG["assumptions"].values()),
}


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
RUN_ROOT = Path(CONFIG["output_root"]) / CONFIG["run_name"]
DIRS = {
    "root": RUN_ROOT,
    "logs": RUN_ROOT / "logs",
    "meta": RUN_ROOT / "meta",
    "splits": RUN_ROOT / "splits",
    "checkpoints": RUN_ROOT / "checkpoints",
    "checkpoints_proposed_seg": RUN_ROOT / "checkpoints" / "proposed_segmentation",
    "checkpoints_unet": RUN_ROOT / "checkpoints" / "unet",
    "checkpoints_segnet": RUN_ROOT / "checkpoints" / "segnet",
    "checkpoints_medt": RUN_ROOT / "checkpoints" / "medt",
    "checkpoints_vit": RUN_ROOT / "checkpoints" / "vit",
    "checkpoints_swin": RUN_ROOT / "checkpoints" / "swin",
    "checkpoints_convnext": RUN_ROOT / "checkpoints" / "convnext",
    "models_final": RUN_ROOT / "models" / "final",
    "masks_train": RUN_ROOT / "results" / "masks" / "train",
    "masks_val": RUN_ROOT / "results" / "masks" / "val",
    "masks_test": RUN_ROOT / "results" / "masks" / "test",
    "plots": RUN_ROOT / "results" / "plots",
    "metrics": RUN_ROOT / "results" / "metrics",
    "tables": RUN_ROOT / "results" / "tables",
    "overlays": RUN_ROOT / "results" / "overlays",
    "predictions": RUN_ROOT / "results" / "predictions",
    "histories": RUN_ROOT / "results" / "histories",
    "epoch_probs": RUN_ROOT / "results" / "epoch_probabilities",
    "samples": RUN_ROOT / "results" / "samples",
    "paper_reference": RUN_ROOT / "paper_reference",
    "cache": RUN_ROOT / "cache",
    "cache_rgb": RUN_ROOT / "cache" / "decoded_rgb",
    "cache_mask": RUN_ROOT / "cache" / "decoded_mask",
}

for p in DIRS.values():
    p.mkdir(parents=True, exist_ok=True)


class Tee:
    def __init__(self, filepath: Path):
        self.file = open(filepath, "a", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        if "\n" in data or "\r" in data:
            self.flush()
        return len(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def isatty(self):
        return bool(getattr(self.stdout, "isatty", lambda: False)())


sys.stdout = Tee(DIRS["logs"] / CONFIG["log_filename"])
sys.stderr = sys.stdout


def log_header(msg: str):
    print("\n" + "=" * 100)
    print(msg)
    print("=" * 100)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> Dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict | None):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def save_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_trusted_checkpoint(path: Path) -> Dict:
    # These checkpoints are generated locally by this script and store
    # optimizer/RNG state in addition to model weights.
    return torch.load(path, map_location="cpu", weights_only=False)


def archive_checkpoint_artifacts(manager, reason: str):
    archive_dir = manager.save_dir / f"archived_{manager.prefix}_{reason}_{int(time.time())}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = [
        manager.latest_path(),
        manager.best_path(),
        manager.history_path(),
        manager.index_path(),
        *manager.epoch_paths(),
    ]

    moved = []
    for path in artifact_paths:
        if path.exists():
            shutil.move(str(path), str(archive_dir / path.name))
            moved.append(path.name)

    if moved:
        print(f"Archived incompatible checkpoints for {manager.prefix} to {archive_dir}")


def checkpoint_config_mismatches(ckpt: Dict | None, keys: List[str]) -> List[str]:
    if ckpt is None:
        return []
    saved_config = ckpt.get("config", {})
    mismatches = []
    for key in keys:
        if saved_config.get(key) != CONFIG.get(key):
            mismatches.append(f"{key}: checkpoint={saved_config.get(key)} current={CONFIG.get(key)}")
    return mismatches


def summarize_early_stopping(metric_history: List[float], min_delta: float) -> Tuple[float, int, int]:
    best_metric = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch_idx, metric in enumerate(metric_history, start=1):
        if metric > best_metric + min_delta:
            best_metric = float(metric)
            best_epoch = epoch_idx
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

    return best_metric, best_epoch, epochs_without_improvement


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def should_use_data_parallel(batch_size: int | None = None) -> bool:
    if not CONFIG["use_all_gpus"]:
        return False
    if torch.cuda.device_count() <= 1:
        return False

    mode = str(CONFIG.get("gpu_parallel_mode", "off")).lower()
    if mode == "force":
        return True
    if mode != "auto":
        return False
    if batch_size is None:
        return False

    per_gpu_batch = batch_size / torch.cuda.device_count()
    return per_gpu_batch >= CONFIG["data_parallel_min_batch_per_gpu"]


def maybe_parallel(model: nn.Module, batch_size: int | None = None, tag: str = "model") -> nn.Module:
    if should_use_data_parallel(batch_size):
        print(
            f"Using DataParallel for {tag} "
            f"(global batch={batch_size}, per-GPU batch={batch_size / torch.cuda.device_count():.1f})"
        )
        return nn.DataParallel(model)
    if CONFIG["use_all_gpus"] and torch.cuda.device_count() > 1:
        print(f"Using single GPU for {tag} to avoid DataParallel overhead.")
    return model


def save_csv(path: Path, rows: List[Dict]):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def save_df(path: Path, df: pd.DataFrame):
    df.to_csv(path, index=False)


def ensure(condition: bool, message: str):
    if not condition:
        raise RuntimeError(message)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def plot_and_save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def atomic_save_npy(path: Path, arr: np.ndarray):
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp_path, "wb") as f:
        np.save(f, arr)
    os.replace(tmp_path, path)


@contextmanager
def make_tqdm(iterable, **kwargs):
    use_tty = sys.stdout.isatty()
    desc = kwargs.get("desc")
    progress = tqdm(iterable, disable=not use_tty, **kwargs)
    try:
        yield progress
    finally:
        progress.close()
        if not use_tty and desc:
            print(f"{desc}: 100%", flush=True)


def cache_key(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()


def disk_cache_path(kind: str, source_path: str) -> Path:
    root = DIRS["cache_rgb"] if kind == "rgb" else DIRS["cache_mask"]
    src = Path(source_path)
    ext = src.suffix.lower().replace(".", "") or "bin"
    stem = src.stem
    return root / f"{stem}_{cache_key(source_path)}.{ext}.npy"


def source_mtime_ns(path: str) -> int:
    return Path(path).stat().st_mtime_ns


def cache_is_fresh(cache_path: Path, source_path: str) -> bool:
    if not cache_path.exists():
        return False
    try:
        return cache_path.stat().st_mtime_ns >= source_mtime_ns(source_path)
    except FileNotFoundError:
        return False


# -----------------------------------------------------------------------------
# args
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposed", action="store_true", help="Run only the proposed model pipeline (default).")
    parser.add_argument("--all", action="store_true", help="Run proposed pipeline plus baseline segmentation models.")
    args = parser.parse_args()
    if not args.proposed and not args.all:
        args.proposed = True
    return args


# -----------------------------------------------------------------------------
# device + environment
# -----------------------------------------------------------------------------
def setup_environment():
    torch.set_float32_matmul_precision(CONFIG["float32_matmul_precision"])
    if CONFIG["require_cuda"]:
        ensure(torch.cuda.is_available(), "CUDA is required by CONFIG['require_cuda']=True, but CUDA is not available.")
    device = torch.device("cuda")
    set_seed(CONFIG["seed"])
    torch.backends.cudnn.benchmark = True
    meta = {
        "started_at": now_str(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "devices": [],
        "config": CONFIG,
        "assumptions": ASSUMPTIONS,
    }
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        meta["devices"].append({
            "index": i,
            "name": props.name,
            "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
        })
    save_json(DIRS["meta"] / "run_meta.json", meta)

    log_header("ENVIRONMENT")
    print(f"Started: {meta['started_at']}")
    print(f"Torch: {meta['torch']}")
    print(f"CUDA device count: {meta['cuda_device_count']}")
    for d in meta["devices"]:
        print(f"GPU {d['index']}: {d['name']} | {d['total_memory_gb']} GB")
    return device


# -----------------------------------------------------------------------------
# dataset indexing (flexible scan; continue with whatever paired data exists)
# -----------------------------------------------------------------------------
@dataclass
class SampleRow:
    image_path: str
    mask_path: str
    class_index: int
    class_folder: str
    class_display: str
    filename: str
    case_id: str


def infer_case_id(filename: str) -> str:
    stem = Path(filename).stem
    match = re.match(CONFIG["case_id_regex"], stem)
    return match.group(1) if match else stem


def build_index() -> pd.DataFrame:
    log_header("DATASET SCAN")
    dataset_root = Path(CONFIG["dataset_root"])
    ensure(dataset_root.exists(), f"Dataset root not found: {dataset_root}")

    configured_folders = CONFIG["class_folders"]
    active_folders: List[str] = []

    rows = []
    total = 0
    for folder in configured_folders:
        display_name = folder
        img_dir = dataset_root / folder / "image"
        mask_dir = dataset_root / folder / "label"
        if not img_dir.exists() or not mask_dir.exists():
            print(f"{folder:20s} | missing image/label folder, skipping")
            continue

        img_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() == CONFIG["image_ext"]])
        mask_files = sorted([p for p in mask_dir.iterdir() if p.suffix.lower() == CONFIG["mask_ext"]])
        img_lookup = {p.name: p for p in img_files}
        mask_lookup = {p.name: p for p in mask_files}
        paired_names = sorted(set(img_lookup).intersection(mask_lookup))

        dropped_images = len(img_files) - len(paired_names)
        dropped_labels = len(mask_files) - len(paired_names)
        print(
            f"{folder:20s} | images={len(img_files):4d} | labels={len(mask_files):4d} "
            f"| paired={len(paired_names):4d} | dropped_img={dropped_images:3d} | dropped_lbl={dropped_labels:3d}"
        )
        if not paired_names:
            continue

        class_idx = len(active_folders)
        active_folders.append(folder)
        for name in paired_names:
            img_path = img_lookup[name]
            rows.append(asdict(SampleRow(
                image_path=str(img_path),
                mask_path=str(mask_lookup[name]),
                class_index=class_idx,
                class_folder=folder,
                class_display=display_name,
                filename=name,
                case_id=infer_case_id(name),
            )))
        total += len(paired_names)

    print(f"Total image/mask pairs found: {total}")
    ensure(total > 0, "No valid image/mask pairs found.")
    CONFIG["class_folders"] = active_folders
    CONFIG["expected_num_classes"] = len(active_folders)

    df = pd.DataFrame(rows)
    save_df(DIRS["meta"] / "dataset_index.csv", df)

    fig, ax = plt.subplots(figsize=(10, 4))
    counts = df["class_display"].value_counts().reindex(CONFIG["class_folders"])
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Dataset class distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plot_and_save(fig, DIRS["plots"] / "dataset_class_distribution.png")

    return df


def make_imagewise_splits(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    idx_all = np.arange(len(df))
    y = df["class_index"].values

    try:
        idx_trainval, idx_test = train_test_split(
            idx_all,
            test_size=CONFIG["test_ratio"],
            stratify=y,
            random_state=CONFIG["seed"],
        )
    except ValueError as exc:
        print(f"Stratified test split unavailable ({exc}); falling back to non-stratified split.")
        idx_trainval, idx_test = train_test_split(
            idx_all,
            test_size=CONFIG["test_ratio"],
            stratify=None,
            random_state=CONFIG["seed"],
        )
    # 70/10/20 overall => val must be 0.10 / 0.80 = 0.125 inside trainval
    try:
        idx_train, idx_val = train_test_split(
            idx_trainval,
            test_size=0.125,
            stratify=y[idx_trainval],
            random_state=CONFIG["seed"],
        )
    except ValueError as exc:
        print(f"Stratified val split unavailable ({exc}); falling back to non-stratified split.")
        idx_train, idx_val = train_test_split(
            idx_trainval,
            test_size=0.125,
            stratify=None,
            random_state=CONFIG["seed"],
        )

    splits = {
        "train": df.iloc[idx_train].reset_index(drop=True),
        "val": df.iloc[idx_val].reset_index(drop=True),
        "test": df.iloc[idx_test].reset_index(drop=True),
    }
    return splits


def split_score(
    split_counts: Dict[str, np.ndarray],
    split_sizes: Dict[str, int],
    target_counts: Dict[str, np.ndarray],
    target_sizes: Dict[str, float],
) -> float:
    score = 0.0
    size_weight = float(CONFIG["group_split_size_weight"])
    for split_name in ["train", "val", "test"]:
        class_denom = np.maximum(target_counts[split_name], 1.0)
        class_error = (split_counts[split_name] - target_counts[split_name]) / class_denom
        score += float(np.sum(class_error ** 2))
        size_error = (split_sizes[split_name] - target_sizes[split_name]) / max(target_sizes[split_name], 1.0)
        score += size_weight * float(size_error ** 2)
    return score


def make_casewise_splits(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    ensure("case_id" in df.columns, "case_id column missing from dataset index.")

    group_infos = []
    num_classes = CONFIG["expected_num_classes"]
    for case_id, group_df in df.groupby("case_id", sort=True):
        indices = group_df.index.to_numpy(dtype=np.int64)
        counts = np.bincount(group_df["class_index"].to_numpy(dtype=np.int64), minlength=num_classes).astype(np.float64)
        dominant_count = float(counts.max())
        # Larger and more mixed-class groups are assigned first because they are
        # the hardest to place while preserving class ratios.
        priority = (len(indices), int(np.count_nonzero(counts)), dominant_count)
        group_infos.append({
            "case_id": str(case_id),
            "indices": indices,
            "counts": counts,
            "size": int(len(indices)),
            "priority": priority,
        })

    total_counts = np.bincount(df["class_index"].to_numpy(dtype=np.int64), minlength=num_classes).astype(np.float64)
    ratios = {
        "train": float(CONFIG["train_ratio"]),
        "val": float(CONFIG["val_ratio"]),
        "test": float(CONFIG["test_ratio"]),
    }
    target_counts = {name: total_counts * ratio for name, ratio in ratios.items()}
    target_sizes = {name: float(len(df)) * ratio for name, ratio in ratios.items()}

    base_order = sorted(group_infos, key=lambda item: item["priority"], reverse=True)
    attempts = max(int(CONFIG["group_split_attempts"]), 1)
    best_assignment = None
    best_score = float("inf")

    for attempt in range(attempts):
        rng = random.Random(int(CONFIG["seed"]) + attempt)
        if attempt == 0:
            ordered_groups = list(base_order)
        else:
            ordered_groups = list(group_infos)
            rng.shuffle(ordered_groups)
            ordered_groups.sort(
                key=lambda item: (
                    item["size"] // 3,
                    int(np.count_nonzero(item["counts"])),
                    rng.random(),
                ),
                reverse=True,
            )

        split_counts = {name: np.zeros(num_classes, dtype=np.float64) for name in ratios}
        split_sizes = {name: 0 for name in ratios}
        assignment = {name: [] for name in ratios}

        for group in ordered_groups:
            best_split = None
            best_candidate_score = float("inf")
            for split_name in ["train", "val", "test"]:
                candidate_counts = {name: counts.copy() for name, counts in split_counts.items()}
                candidate_sizes = dict(split_sizes)
                candidate_counts[split_name] += group["counts"]
                candidate_sizes[split_name] += group["size"]
                candidate_score = split_score(candidate_counts, candidate_sizes, target_counts, target_sizes)
                if candidate_score < best_candidate_score:
                    best_candidate_score = candidate_score
                    best_split = split_name

            ensure(best_split is not None, "Case-wise split assignment failed.")
            split_counts[best_split] += group["counts"]
            split_sizes[best_split] += group["size"]
            assignment[best_split].append(group)

        score = split_score(split_counts, split_sizes, target_counts, target_sizes)
        if score < best_score:
            best_score = score
            best_assignment = assignment

    ensure(best_assignment is not None, "Case-wise split search failed.")

    splits = {}
    for split_name in ["train", "val", "test"]:
        idx = np.concatenate([group["indices"] for group in best_assignment[split_name]])
        splits[split_name] = df.iloc[np.sort(idx)].reset_index(drop=True)
    return splits


def split_fingerprint(splits: Dict[str, pd.DataFrame]) -> str:
    parts = []
    for split_name in ["train", "val", "test"]:
        if CONFIG["split_strategy"] == "case":
            items = sorted(splits[split_name]["case_id"].astype(str).unique().tolist())
        else:
            items = sorted(splits[split_name]["filename"].astype(str).tolist())
        parts.append(f"{split_name}:{','.join(items)}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def case_overlap_report(splits: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    case_sets = {name: set(sdf["case_id"].astype(str)) for name, sdf in splits.items()}
    return {
        "train_val": sorted(case_sets["train"] & case_sets["val"]),
        "train_test": sorted(case_sets["train"] & case_sets["test"]),
        "val_test": sorted(case_sets["val"] & case_sets["test"]),
    }


def make_splits(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    log_header("DATA SPLITS")
    strategy = str(CONFIG["split_strategy"]).lower()
    if strategy == "case":
        splits = make_casewise_splits(df)
    elif strategy == "image":
        splits = make_imagewise_splits(df)
    else:
        raise ValueError(f"Unsupported split_strategy={CONFIG['split_strategy']!r}. Use 'case' or 'image'.")

    fingerprint = split_fingerprint(splits)
    CONFIG["active_split_fingerprint"] = fingerprint
    overlaps = case_overlap_report(splits)
    if strategy == "case":
        leaked_pairs = {name: values for name, values in overlaps.items() if values}
        ensure(not leaked_pairs, f"Case-wise split leakage detected: {leaked_pairs}")

    for name, sdf in splits.items():
        save_df(DIRS["splits"] / f"{name}.csv", sdf)
        print(f"{name:5s}: {len(sdf)}")
        print(sdf["class_display"].value_counts().reindex(CONFIG["class_folders"]).to_string())
        if "case_id" in sdf.columns:
            print(f"{'cases':5s}: {sdf['case_id'].nunique()}")

    save_json(DIRS["splits"] / "split_summary.json", {
        "strategy": strategy,
        "fingerprint": fingerprint,
        "case_id_regex": CONFIG["case_id_regex"],
        "case_overlap_counts": {name: len(values) for name, values in overlaps.items()},
        "case_overlap_examples": {name: values[:10] for name, values in overlaps.items()},
        "splits": {
            k: {
                "n": len(v),
                "cases": int(v["case_id"].nunique()) if "case_id" in v.columns else None,
                "class_counts": v["class_display"].value_counts().to_dict(),
                "class_percent_of_total": {
                    cls: round(
                        100.0 * int(v["class_display"].value_counts().get(cls, 0))
                        / max(int(df["class_display"].value_counts().get(cls, 0)), 1),
                        4,
                    )
                    for cls in CONFIG["class_folders"]
                },
            } for k, v in splits.items()
        },
    })
    return splits


# -----------------------------------------------------------------------------
# preprocessing
# -----------------------------------------------------------------------------
class RandStainNA:
    def __init__(self, std_hyper: float, probability: float):
        self.std_hyper = float(std_hyper)
        self.probability = float(probability)

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        if random.random() > self.probability:
            return img_rgb
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        for c in range(3):
            mean_c = float(lab[:, :, c].mean())
            std_c = float(lab[:, :, c].std()) + 1e-6
            new_mean = mean_c + np.random.normal(0.0, self.std_hyper * max(mean_c, 1e-6))
            new_std = std_c * (1.0 + np.random.normal(0.0, self.std_hyper))
            new_std = max(new_std, 1e-6)
            lab[:, :, c] = (lab[:, :, c] - mean_c) / std_c * new_std + new_mean
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


@lru_cache(maxsize=1)
def get_clahe_operator():
    return cv2.createCLAHE(
        clipLimit=CONFIG["clahe_clip_limit"],
        tileGridSize=tuple(CONFIG["clahe_tile_grid_size"])
    )


def apply_clahe(img_rgb: np.ndarray, clahe=None) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = clahe or get_clahe_operator()
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


@lru_cache(maxsize=4096)
def _load_rgb_cached(path: str) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


@lru_cache(maxsize=8192)
def _load_mask_cached(path: str) -> np.ndarray:
    with Image.open(path) as m:
        m = np.array(m.convert("L"))
    return (m > 0).astype(np.uint8)


def load_rgb_from_disk_cache(path: str) -> np.ndarray | None:
    if not CONFIG["enable_disk_decode_cache"]:
        return None
    cache_path = disk_cache_path("rgb", path)
    if cache_is_fresh(cache_path, path):
        return np.load(cache_path, allow_pickle=False)
    return None


def load_mask_from_disk_cache(path: str) -> np.ndarray | None:
    if not CONFIG["enable_disk_decode_cache"]:
        return None
    cache_path = disk_cache_path("mask", path)
    if cache_is_fresh(cache_path, path):
        return np.load(cache_path, allow_pickle=False)
    return None


def maybe_write_rgb_disk_cache(path: str, arr: np.ndarray):
    if not CONFIG["enable_disk_decode_cache"]:
        return
    cache_path = disk_cache_path("rgb", path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_is_fresh(cache_path, path):
        atomic_save_npy(cache_path, arr)


def maybe_write_mask_disk_cache(path: str, arr: np.ndarray):
    if not CONFIG["enable_disk_decode_cache"]:
        return
    cache_path = disk_cache_path("mask", path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_is_fresh(cache_path, path):
        atomic_save_npy(cache_path, arr)


def load_rgb(path: str) -> np.ndarray:
    cached = load_rgb_from_disk_cache(path)
    if cached is not None:
        return cached.copy()
    if CONFIG["enable_decode_cache"]:
        arr = _load_rgb_cached(path)
        maybe_write_rgb_disk_cache(path, arr)
        return arr.copy()
    with Image.open(path) as img:
        arr = np.array(img.convert("RGB"))
    maybe_write_rgb_disk_cache(path, arr)
    return arr


def load_mask(path: str) -> np.ndarray:
    cached = load_mask_from_disk_cache(path)
    if cached is not None:
        return cached.copy()
    if CONFIG["enable_decode_cache"]:
        arr = _load_mask_cached(path)
        maybe_write_mask_disk_cache(path, arr)
        return arr.copy()
    with Image.open(path) as m:
        arr = np.array(m.convert("L"))
    arr = (arr > 0).astype(np.uint8)
    maybe_write_mask_disk_cache(path, arr)
    return arr


def warmup_decode_cache(df: pd.DataFrame):
    if not CONFIG["enable_disk_decode_cache"] or not CONFIG["warmup_decode_cache"]:
        return

    log_header("WARMUP DECODE CACHE")
    image_paths = sorted(df["image_path"].unique().tolist())
    mask_paths = sorted(df["mask_path"].unique().tolist())

    with make_tqdm(image_paths, desc="Caching RGB decodes", leave=False) as pbar:
        for image_path in pbar:
            if load_rgb_from_disk_cache(image_path) is None:
                maybe_write_rgb_disk_cache(image_path, _load_rgb_cached(image_path))

    with make_tqdm(mask_paths, desc="Caching mask decodes", leave=False) as pbar:
        for mask_path in pbar:
            if load_mask_from_disk_cache(mask_path) is None:
                maybe_write_mask_disk_cache(mask_path, _load_mask_cached(mask_path))

    _load_rgb_cached.cache_clear()
    _load_mask_cached.cache_clear()

    print(f"RGB cache files: {sum(1 for _ in DIRS['cache_rgb'].glob('*.npy'))}")
    print(f"Mask cache files: {sum(1 for _ in DIRS['cache_mask'].glob('*.npy'))}")


def resize_rgb(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def maybe_imagenet_normalize(img_t: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return img_t
    return (img_t - IMAGENET_MEAN) / IMAGENET_STD


def maybe_imagenet_denormalize(img_t: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return img_t
    return img_t * IMAGENET_STD + IMAGENET_MEAN


def rgb_to_tensor(img_rgb: np.ndarray, imagenet_normalize: bool = False) -> torch.Tensor:
    arr = img_rgb.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    img_t = torch.from_numpy(arr)
    return maybe_imagenet_normalize(img_t, imagenet_normalize)


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(mask[None, :, :].astype(np.float32))


def soften_mask(mask: np.ndarray, blur_kernel: int) -> np.ndarray:
    mask_f = mask.astype(np.float32)
    kernel = max(int(blur_kernel), 0)
    if kernel > 1:
        if kernel % 2 == 0:
            kernel += 1
        mask_f = cv2.GaussianBlur(mask_f, (kernel, kernel), 0)
    return np.clip(mask_f, 0.0, 1.0)


def apply_classifier_mask_guidance(img_rgb: np.ndarray, refined_mask: np.ndarray) -> np.ndarray:
    mask_f = soften_mask(refined_mask, CONFIG["cls_mask_blur_kernel"])
    alpha = float(CONFIG["cls_mask_background_alpha"])
    alpha = float(np.clip(alpha, 0.0, 1.0))
    guidance = alpha + (1.0 - alpha) * mask_f[:, :, None]
    guided = img_rgb.astype(np.float32) * guidance
    return np.clip(guided, 0.0, 255.0).astype(np.uint8)


def tensor_to_display_image(img_t: torch.Tensor, imagenet_normalize: bool) -> np.ndarray:
    img = maybe_imagenet_denormalize(img_t.detach().cpu(), imagenet_normalize)
    img = img.clamp(0.0, 1.0)
    return img.permute(1, 2, 0).numpy()


def compute_balanced_class_weights(class_indices: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(class_indices.astype(np.int64), minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    return (counts.sum() / (num_classes * counts)).astype(np.float32)


def save_preprocessing_figure(df: pd.DataFrame):
    row = df.sample(n=1, random_state=CONFIG["seed"]).iloc[0]
    rgb = load_rgb(row["image_path"])
    rs = RandStainNA(CONFIG["randstain_std_hyper"], 1.0)(rgb.copy())
    cl = apply_clahe(rs.copy())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(rs); axes[1].set_title("RandStainNA"); axes[1].axis("off")
    axes[2].imshow(cl); axes[2].set_title("CLAHE"); axes[2].axis("off")
    plot_and_save(fig, DIRS["plots"] / "figure3_preprocessing.png")


# -----------------------------------------------------------------------------
# datasets
# -----------------------------------------------------------------------------
class SegmentationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int, train_mode: bool):
        self.records = list(
            df[["image_path", "mask_path", "class_index", "filename"]].itertuples(index=False, name=None)
        )
        self.image_size = int(image_size)
        self.train_mode = bool(train_mode)
        self.clahe = get_clahe_operator()
        self.randstain = RandStainNA(
            CONFIG["randstain_std_hyper"],
            CONFIG["randstain_probability_train"] if train_mode else CONFIG["randstain_probability_eval"],
        )
        self.eval_cache = {} if (not train_mode and CONFIG["enable_eval_sample_cache"]) else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        if self.eval_cache is not None and idx in self.eval_cache:
            img_t, mask_t, class_idx, filename = self.eval_cache[idx]
            return img_t.clone(), mask_t.clone(), class_idx, filename

        image_path, mask_path, class_index, filename = self.records[idx]
        img = load_rgb(image_path)
        mask = load_mask(mask_path)

        img = self.randstain(img)
        img = apply_clahe(img, clahe=self.clahe)

        if self.train_mode and random.random() < 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        if self.train_mode and random.random() < 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()

        img = resize_rgb(img, self.image_size)
        mask = resize_mask(mask, self.image_size)

        sample = (
            rgb_to_tensor(img, imagenet_normalize=CONFIG["seg_imagenet_normalize"]),
            mask_to_tensor(mask),
            int(class_index),
            filename,
        )
        if self.eval_cache is not None:
            self.eval_cache[idx] = sample
        return sample


class ClassificationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, refined_mask_dir: Path, image_size: int, train_mode: bool):
        self.records = list(
            df[["image_path", "class_index", "class_folder", "filename"]].itertuples(index=False, name=None)
        )
        self.refined_mask_dir = refined_mask_dir
        self.image_size = int(image_size)
        self.train_mode = bool(train_mode)
        self.clahe = get_clahe_operator()
        self.randstain = RandStainNA(
            CONFIG["randstain_std_hyper"],
            CONFIG["randstain_probability_train"] if train_mode else CONFIG["randstain_probability_eval"],
        )
        self.eval_cache = {} if (not train_mode and CONFIG["enable_eval_sample_cache"]) else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        if self.eval_cache is not None and idx in self.eval_cache:
            img_t, class_idx, filename = self.eval_cache[idx]
            return img_t.clone(), class_idx, filename

        image_path, class_index, class_folder, filename = self.records[idx]
        img = load_rgb(image_path)
        img = self.randstain(img)
        img = apply_clahe(img, clahe=self.clahe)

        refined_mask_path = self.refined_mask_dir / class_folder / filename
        ensure(refined_mask_path.exists(), f"Refined mask not found: {refined_mask_path}")
        refined_mask = load_mask(str(refined_mask_path))

        img = resize_rgb(img, self.image_size)
        refined_mask = resize_mask(refined_mask, self.image_size)
        guided = apply_classifier_mask_guidance(img, refined_mask)

        if self.train_mode and random.random() < 0.5:
            guided = np.fliplr(guided).copy()
        if self.train_mode and random.random() < 0.5:
            guided = np.flipud(guided).copy()

        sample = (
            rgb_to_tensor(guided, imagenet_normalize=CONFIG["cls_imagenet_normalize"]),
            int(class_index),
            filename,
        )
        if self.eval_cache is not None:
            self.eval_cache[idx] = sample
        return sample


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool = False,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
        drop_last=drop_last,
        persistent_workers=CONFIG["persistent_workers"] if CONFIG["num_workers"] > 0 else False,
        prefetch_factor=CONFIG["prefetch_factor"] if CONFIG["num_workers"] > 0 else None,
    )


# -----------------------------------------------------------------------------
# losses and metrics
# -----------------------------------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.contiguous().view(pred.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)
        inter = (pred * target).sum(dim=1)
        denom = (pred.pow(2).sum(dim=1) + target.pow(2).sum(dim=1) + self.eps)
        dice = (2.0 * inter + self.eps) / denom
        return 1.0 - dice.mean()


def sobel_edge_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(x, kernel_x, padding=1)
    gy = F.conv2d(x, kernel_y, padding=1)
    mag = torch.sqrt(gx.square() + gy.square() + eps)
    denom = mag.amax(dim=(2, 3), keepdim=True).clamp_min(eps)
    return mag / denom


class BoundaryDiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_edge = sobel_edge_map(pred.float(), eps=self.eps)
        target_edge = sobel_edge_map(target.float(), eps=self.eps)
        pred_edge = pred_edge.contiguous().view(pred_edge.size(0), -1)
        target_edge = target_edge.contiguous().view(target_edge.size(0), -1)
        inter = (pred_edge * target_edge).sum(dim=1)
        denom = pred_edge.sum(dim=1) + target_edge.sum(dim=1) + self.eps
        dice = (2.0 * inter + self.eps) / denom
        return 1.0 - dice.mean()


def batch_dice_score(pred_bin: torch.Tensor, target_bin: torch.Tensor, eps: float = 1e-6) -> float:
    pred_bin = pred_bin.float().view(pred_bin.size(0), -1)
    target_bin = target_bin.float().view(target_bin.size(0), -1)
    inter = (pred_bin * target_bin).sum(dim=1)
    denom = pred_bin.sum(dim=1) + target_bin.sum(dim=1) + eps
    dice = (2.0 * inter + eps) / denom
    return float(dice.mean().item())


def compute_dice_np(pred: np.ndarray, gt: np.ndarray) -> float:
    return float((2.0 * np.sum(pred * gt) + 1e-6) / (np.sum(pred) + np.sum(gt) + 1e-6))


def compute_jaccard_np(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.sum(pred * gt)
    return float((inter + 1e-6) / (np.sum(pred) + np.sum(gt) - inter + 1e-6))


def compute_conformity_np(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = np.sum(pred * gt)
    if tp == 0:
        return -3.0
    errors = np.sum(pred * (1 - gt)) + np.sum((1 - pred) * gt)
    return float(1.0 - errors / tp)


def compute_bfscore_np(pred: np.ndarray, gt: np.ndarray, tol: int = 2) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    pred_boundary = pred ^ binary_erosion(pred)
    gt_boundary = gt ^ binary_erosion(gt)
    if pred_boundary.sum() == 0 and gt_boundary.sum() == 0:
        return 1.0
    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return 0.0
    precision = np.sum(pred_boundary & binary_dilation(gt_boundary, iterations=tol)) / (pred_boundary.sum() + 1e-6)
    recall = np.sum(gt_boundary & binary_dilation(pred_boundary, iterations=tol)) / (gt_boundary.sum() + 1e-6)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def compute_hog_similarity_np(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_img = (pred.astype(np.uint8) * 255)
    gt_img = (gt.astype(np.uint8) * 255)
    hp = sk_hog(pred_img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    hg = sk_hog(gt_img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    return float(np.dot(hp, hg) / (np.linalg.norm(hp) * np.linalg.norm(hg) + 1e-6))


def make_morph_kernel(kernel_size: int) -> np.ndarray | None:
    size = int(kernel_size)
    if size <= 1:
        return None
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def postprocess_binary_mask(mask: np.ndarray, close_kernel: int = 0, open_kernel: int = 0) -> np.ndarray:
    out = mask.astype(np.uint8)
    kernel_close = make_morph_kernel(close_kernel)
    kernel_open = make_morph_kernel(open_kernel)
    if kernel_close is not None:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel_close)
    if kernel_open is not None:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel_open)
    return out.astype(np.uint8)


def default_segmentation_inference_config(model_name: str) -> Dict:
    return {
        "model_name": model_name,
        "threshold": float(CONFIG["seg_threshold"]),
        "close_kernel": 0,
        "open_kernel": 0,
        "use_tta": bool(CONFIG["seg_eval_tta"]),
        "selection_score": None,
    }


def predict_segmentation_probs(model: nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    def forward(batch: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=CONFIG["amp"]):
            return model(batch).float()

    if not CONFIG["seg_eval_tta"]:
        return forward(imgs)

    preds = [forward(imgs)]
    preds.append(torch.flip(forward(torch.flip(imgs, dims=[3])), dims=[3]))
    preds.append(torch.flip(forward(torch.flip(imgs, dims=[2])), dims=[2]))
    preds.append(torch.flip(forward(torch.flip(imgs, dims=[2, 3])), dims=[2, 3]))
    return torch.stack(preds, dim=0).mean(dim=0)


def binarize_segmentation_probs(probs: np.ndarray, inference_cfg: Dict) -> np.ndarray:
    threshold = float(inference_cfg["threshold"])
    close_kernel = int(inference_cfg.get("close_kernel", 0))
    open_kernel = int(inference_cfg.get("open_kernel", 0))
    masks = (np.asarray(probs)[:, 0] >= threshold).astype(np.uint8)
    out = np.empty_like(masks, dtype=np.uint8)
    for idx, mask in enumerate(masks):
        out[idx] = postprocess_binary_mask(mask, close_kernel=close_kernel, open_kernel=open_kernel)
    return out[:, None, :, :]


def summarize_segmentation_predictions(preds_np: np.ndarray, masks_np: np.ndarray, class_idx_np: np.ndarray) -> Dict[str, float]:
    per_class = {
        disp: {"Dice": [], "Jaccard": [], "BF-score": [], "HOG-sim": []}
        for disp in CONFIG["class_folders"]
    }

    for idx in range(preds_np.shape[0]):
        disp = CONFIG["class_folders"][int(class_idx_np[idx])]
        pred = preds_np[idx, 0].astype(np.uint8)
        gt = masks_np[idx, 0].astype(np.uint8)
        per_class[disp]["Dice"].append(compute_dice_np(pred, gt))
        per_class[disp]["Jaccard"].append(compute_jaccard_np(pred, gt))
        per_class[disp]["BF-score"].append(compute_bfscore_np(pred, gt))
        per_class[disp]["HOG-sim"].append(compute_hog_similarity_np(pred, gt))

    summary = {}
    for metric in ["Dice", "Jaccard", "BF-score", "HOG-sim"]:
        summary[metric] = float(np.mean([
            np.mean(per_class[disp][metric]) for disp in CONFIG["class_folders"] if per_class[disp][metric]
        ]))
    return summary


def segmentation_selection_score(summary: Dict[str, float]) -> float:
    weights = CONFIG["seg_selection_weights"]
    return float(sum(float(weights[k]) * float(summary[k]) for k in weights))


def tune_segmentation_inference(
    model_name: str,
    model: nn.Module,
    val_df: pd.DataFrame,
    device: torch.device,
) -> Dict:
    inference_cfg = default_segmentation_inference_config(model_name)
    if not CONFIG["seg_tune_inference_on_val"]:
        save_json(DIRS["metrics"] / f"{model_name}_seg_inference_tuning.json", inference_cfg)
        return inference_cfg

    ds = SegmentationDataset(val_df, CONFIG["seg_image_size"], train_mode=False)
    loader = make_loader(ds, batch_size=CONFIG["seg_batch_size"], shuffle=False)

    probs_all, masks_all, class_idx_all = [], [], []
    model.eval()
    with torch.inference_mode():
        with make_tqdm(loader, desc=f"Tuning seg inference {model_name}", leave=False) as pbar:
            for imgs, masks, class_idx, _ in pbar:
                imgs = imgs.to(device, non_blocking=True)
                probs = predict_segmentation_probs(model, imgs)
                probs_all.append(probs.cpu().numpy())
                masks_all.append(masks.numpy().astype(np.uint8))
                class_idx_all.append(class_idx.numpy())

    probs_all = np.concatenate(probs_all, axis=0)
    masks_all = np.concatenate(masks_all, axis=0)
    class_idx_all = np.concatenate(class_idx_all, axis=0)

    best = None
    for threshold in CONFIG["seg_threshold_candidates"]:
        for close_kernel in CONFIG["seg_postprocess_close_kernels"]:
            for open_kernel in CONFIG["seg_postprocess_open_kernels"]:
                candidate_cfg = {
                    "model_name": model_name,
                    "threshold": float(threshold),
                    "close_kernel": int(close_kernel),
                    "open_kernel": int(open_kernel),
                    "use_tta": bool(CONFIG["seg_eval_tta"]),
                }
                pred_np = binarize_segmentation_probs(probs_all, candidate_cfg)
                summary = summarize_segmentation_predictions(pred_np, masks_all, class_idx_all)
                score = segmentation_selection_score(summary)
                candidate = {
                    **candidate_cfg,
                    "selection_score": score,
                    "summary": summary,
                }
                if best is None or score > best["selection_score"]:
                    best = candidate

    ensure(best is not None, f"Failed to tune segmentation inference for {model_name}.")
    print(json.dumps({
        "model_name": model_name,
        "selected_threshold": best["threshold"],
        "selected_close_kernel": best["close_kernel"],
        "selected_open_kernel": best["open_kernel"],
        "selection_score": round(float(best["selection_score"]), 6),
        "summary": {k: round(float(v), 6) for k, v in best["summary"].items()},
    }, indent=2))
    save_json(DIRS["metrics"] / f"{model_name}_seg_inference_tuning.json", best)
    return best


def logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits.float(), dim=1)
    denom = probs.sum(dim=1, keepdim=True).clamp_min(torch.finfo(probs.dtype).eps)
    return probs / denom


def normalize_probability_rows(y_prob: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    probs = np.asarray(y_prob, dtype=np.float64)
    ensure(probs.ndim == 2, f"Expected probability matrix with shape [N, C], got {probs.shape}.")
    ensure(np.isfinite(probs).all(), "Probability matrix contains NaN or Inf values.")
    probs = np.clip(probs, 0.0, None)
    row_sums = probs.sum(axis=1, keepdims=True)
    ensure((row_sums > 0).all(), "Probability normalization failed due to zero-sum rows.")
    probs = probs / row_sums
    ensure(np.allclose(1.0, probs.sum(axis=1)), "Probability normalization failed to produce valid class probabilities.")
    return probs


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None) -> Dict[str, float]:
    out = {
        "Accuracy (%)": accuracy_score(y_true, y_pred) * 100.0,
        "Precision (%)": precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0,
        "Recall (%)": recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0,
        "F1-score (%)": f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0,
        "AUC": None,
    }
    if y_prob is not None:
        y_prob = normalize_probability_rows(y_prob)
        out["AUC"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="micro"))
    return out


# -----------------------------------------------------------------------------
# model blocks
# -----------------------------------------------------------------------------
class MultiScaleAttentionBlock(nn.Module):
    def __init__(self, in_channels: int, gating_channels: int, inter_channels: int | None = None):
        super().__init__()
        inter_channels = inter_channels or max(8, in_channels // 2)
        self.wx3 = nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1)
        self.wx5 = nn.Conv2d(in_channels, inter_channels, kernel_size=5, padding=2)
        self.wx7 = nn.Conv2d(in_channels, inter_channels, kernel_size=7, padding=3)
        self.wg = nn.Conv2d(gating_channels, inter_channels, kernel_size=1)
        self.psi = nn.Sequential(nn.Conv2d(inter_channels, 1, kernel_size=1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x_ms = (self.wx3(x) + self.wx5(x) + self.wx7(x)) / 3.0
        g_proj = self.wg(g)
        if x_ms.shape[2:] != g_proj.shape[2:]:
            g_proj = F.interpolate(g_proj, size=x_ms.shape[2:], mode="bilinear", align_corners=True)
        attn = self.psi(self.relu(x_ms + g_proj))
        return self.bn(x * attn)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.residual = ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.residual(x)


class GRHDUNet(nn.Module):
    def __init__(self):
        super().__init__()
        features = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT).features
        self.enc0 = nn.Sequential(features.conv0, features.norm0, features.relu0, features.pool0)   # 64
        self.enc1 = nn.Sequential(features.denseblock1, features.transition1)                         # 128
        self.enc2 = nn.Sequential(features.denseblock2, features.transition2)                         # 256
        self.enc3 = nn.Sequential(features.denseblock3, features.transition3)                         # 512
        self.enc4 = features.denseblock4                                                              # 1024

        self.attn4 = MultiScaleAttentionBlock(512, 1024)
        self.attn3 = MultiScaleAttentionBlock(256, 512)
        self.attn2 = MultiScaleAttentionBlock(128, 256)
        self.attn1 = MultiScaleAttentionBlock(64, 128)

        self.dec4 = DecoderBlock(1024, 512, 512)
        self.dec3 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64, 64)

        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d4 = self.dec4(e4, self.attn4(e3, e4))
        d3 = self.dec3(d4, self.attn3(e2, d4))
        d2 = self.dec2(d3, self.attn2(e1, d3))
        d1 = self.dec1(d2, self.attn1(e0, d2))
        out = self.final_up(d1)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=True)
        return self.final_conv(out)


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 4):
        super().__init__()

        def block(cin, cout, norm=True):
            layers = [nn.Conv2d(cin, cout, kernel_size=4, stride=2, padding=1)]
            if norm:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            block(in_channels, 64, norm=False),
            block(64, 128, norm=True),
            block(128, 256, norm=True),
            block(256, 512, norm=True),
            nn.Conv2d(512, 1, kernel_size=4, padding=1),
        )

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([img, mask], dim=1))


# -----------------------------------------------------------------------------
# baseline segmentation models
# -----------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = DoubleConv(3, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        self.d4 = DoubleConv(256, 512)
        self.b = DoubleConv(512, 1024)
        self.pool = nn.MaxPool2d(2)

        self.u4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.c4 = DoubleConv(1024, 512)
        self.u3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.c3 = DoubleConv(512, 256)
        self.u2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.c2 = DoubleConv(256, 128)
        self.u1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.c1 = DoubleConv(128, 64)
        self.out = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))
        b = self.b(self.pool(d4))
        x = self.u4(b); x = self.c4(torch.cat([x, d4], 1))
        x = self.u3(x); x = self.c3(torch.cat([x, d3], 1))
        x = self.u2(x); x = self.c2(torch.cat([x, d2], 1))
        x = self.u1(x); x = self.c1(torch.cat([x, d1], 1))
        return self.out(x)


class SegNetBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )

        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)

        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x1 = self.enc1(x)
        s1 = x1.size()
        x1, i1 = self.pool(x1)

        x2 = self.enc2(x1)
        s2 = x2.size()
        x2, i2 = self.pool(x2)

        x3 = self.enc3(x2)
        s3 = x3.size()
        x3, i3 = self.pool(x3)

        x = self.unpool(x3, i3, output_size=s3)
        x = self.dec3(x)
        x = self.unpool(x, i2, output_size=s2)
        x = self.dec2(x)
        x = self.unpool(x, i1, output_size=s1)
        x = self.dec1(x)
        return x


class AxialAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Conv1d(dim, dim * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

    def forward_axis(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        # x: B,C,H,W
        if axis == 2:   # H
            b, c, h, w = x.shape
            x = x.permute(0, 3, 1, 2).reshape(b * w, c, h)
        else:           # W
            b, c, h, w = x.shape
            x = x.permute(0, 2, 1, 3).reshape(b * h, c, w)

        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        head_dim = q.shape[1] // self.heads
        q = q.view(q.shape[0], self.heads, head_dim, q.shape[-1]).transpose(2, 3)
        k = k.view(k.shape[0], self.heads, head_dim, k.shape[-1]).transpose(2, 3)
        v = v.view(v.shape[0], self.heads, head_dim, v.shape[-1]).transpose(2, 3)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn, v).transpose(2, 3).contiguous().view(x.shape[0], -1, x.shape[-1])
        out = self.proj(out)

        if axis == 2:
            out = out.view(b, w, c, h).permute(0, 2, 3, 1)
        else:
            out = out.view(b, h, c, w).permute(0, 2, 1, 3)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_axis(x, 2) + self.forward_axis(x, 3)


class MedTBaseline(nn.Module):
    """
    Honest note:
    The paper names MedT only as a comparison model and does not specify the exact MedT variant.
    This is a single-file MedT-like axial-attention encoder-decoder baseline so that --all actually
    runs a transformer-style medical segmentation baseline instead of silently skipping it.
    """
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            AxialAttention(128, heads=8),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            AxialAttention(256, heads=8),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            AxialAttention(256, heads=8),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = DoubleConv(128, 64)
        self.out = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        b = self.bottleneck(s2)
        x = self.up2(b)
        x = self.dec2(torch.cat([x, s1], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, s0], dim=1))
        return self.out(x)


# -----------------------------------------------------------------------------
# classification backbones
# -----------------------------------------------------------------------------
def build_vit(num_classes: int):
    return timm.create_model(CONFIG["assumptions"]["vit_model"], pretrained=True, num_classes=num_classes)


def build_swin(num_classes: int):
    return timm.create_model(CONFIG["assumptions"]["swin_model"], pretrained=True, num_classes=num_classes)


def build_convnext(num_classes: int):
    return timm.create_model(CONFIG["assumptions"]["convnext_model"], pretrained=True, num_classes=num_classes)


# -----------------------------------------------------------------------------
# checkpoint manager
# -----------------------------------------------------------------------------
class CheckpointManager:
    def __init__(self, save_dir: Path, prefix: str, top_k: int):
        self.save_dir = save_dir
        self.prefix = prefix
        self.top_k = int(top_k)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def latest_path(self) -> Path:
        return self.save_dir / f"{self.prefix}_latest.pt"

    def best_path(self) -> Path:
        return self.save_dir / f"{self.prefix}_best.pt"

    def history_path(self) -> Path:
        return self.save_dir / f"{self.prefix}_history.json"

    def index_path(self) -> Path:
        return self.save_dir / f"{self.prefix}_epoch_index.json"

    def epoch_pattern(self) -> str:
        return f"{self.prefix}_epoch_*.pt"

    def epoch_paths(self) -> List[Path]:
        return sorted(self.save_dir.glob(self.epoch_pattern()))

    def load_epoch_index(self) -> List[Dict]:
        index_path = self.index_path()
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            return [r for r in records if (self.save_dir / r["filename"]).exists()]

        records = []
        for path in self.epoch_paths():
            payload = load_trusted_checkpoint(path)
            records.append({
                "filename": path.name,
                "metric": float(payload["metric"]),
                "epoch": int(payload["epoch"]),
            })
        return records

    def save_epoch_index(self, records: List[Dict]):
        save_json(self.index_path(), records)

    def prune_epoch_checkpoints(self, records: List[Dict]) -> List[Dict]:
        if self.top_k <= 0:
            for record in records:
                (self.save_dir / record["filename"]).unlink(missing_ok=True)
            self.save_epoch_index([])
            return []

        ranked = sorted(records, key=lambda item: (float(item["metric"]), int(item["epoch"])), reverse=True)
        keep = ranked[:self.top_k]
        keep_names = {item["filename"] for item in keep}

        for record in records:
            if record["filename"] not in keep_names:
                (self.save_dir / record["filename"]).unlink(missing_ok=True)

        self.save_epoch_index(keep)
        return keep

    def save(self, payload: Dict, metric: float, epoch: int):
        latest = self.latest_path()
        torch.save(payload, latest)
        if CONFIG["save_every_epoch"]:
            epoch_filename = f"{self.prefix}_epoch_{epoch:03d}.pt"
            torch.save(payload, self.save_dir / epoch_filename)
            records = [r for r in self.load_epoch_index() if r["filename"] != epoch_filename]
            records.append({
                "filename": epoch_filename,
                "metric": float(metric),
                "epoch": int(epoch),
            })
            self.prune_epoch_checkpoints(records)

        best_metric = -float("inf")
        if self.best_path().exists():
            best_metric = load_trusted_checkpoint(self.best_path())["metric"]
        if metric >= best_metric:
            torch.save(payload, self.best_path())

    def load_latest(self) -> Dict | None:
        if self.latest_path().exists():
            return load_trusted_checkpoint(self.latest_path())
        return None

    def load_best(self) -> Dict | None:
        if self.best_path().exists():
            return load_trusted_checkpoint(self.best_path())
        return None


# -----------------------------------------------------------------------------
# segmentation training
# -----------------------------------------------------------------------------
def run_seg_epoch_proposed(
    generator: nn.Module,
    discriminator: nn.Module,
    loader: DataLoader,
    optimizer_g,
    optimizer_d,
    scaler_g,
    scaler_d,
    device: torch.device,
    train: bool,
) -> Dict[str, float]:
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()
    criterion_recon = nn.BCELoss()
    criterion_boundary = BoundaryDiceLoss()
    use_amp = CONFIG["proposed_seg_amp"] if train else False
    disc_update_interval = max(int(CONFIG["seg_disc_update_interval"]), 1)
    gen_mode = generator.train if train else generator.eval
    disc_mode = discriminator.train if train else discriminator.eval
    gen_mode()
    disc_mode()

    totals = defaultdict(float)
    steps = 0

    context = torch.enable_grad if train else torch.inference_mode
    with context():
        with make_tqdm(loader, leave=False, desc=f"{'train' if train else 'val'} proposed seg") as pbar:
            for step_idx, (imgs, masks, _, _) in enumerate(pbar, start=1):
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                loss_d_finite = True
                loss_g_finite = True

                if train:
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        fake_masks_detached = generator(imgs).detach()
                    pred_real = discriminator(imgs, masks)
                    loss_d_real = criterion_bce(pred_real, torch.ones_like(pred_real, device=device) * 0.9)
                    pred_fake = discriminator(imgs, fake_masks_detached.float())
                    loss_d_fake = criterion_bce(pred_fake, torch.zeros_like(pred_fake, device=device))
                    loss_d = 0.5 * (loss_d_real + loss_d_fake)
                    loss_d_finite = bool(torch.isfinite(loss_d).item())
                    if not loss_d_finite:
                        optimizer_d.zero_grad(set_to_none=True)
                        raise FloatingPointError("Non-finite discriminator loss encountered during proposed segmentation training.")
                    if (step_idx - 1) % disc_update_interval == 0:
                        optimizer_d.zero_grad(set_to_none=True)
                        scaler_d.scale(loss_d).backward()
                        scaler_d.unscale_(optimizer_d)
                        torch.nn.utils.clip_grad_norm_(unwrap_model(discriminator).parameters(), max_norm=1.0)
                        scaler_d.step(optimizer_d)
                        scaler_d.update()

                    optimizer_g.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        fake_masks = generator(imgs)
                    fake_masks_fp32 = fake_masks.float()
                    masks_fp32 = masks.float()
                    loss_g_dice = criterion_dice(fake_masks_fp32, masks_fp32)
                    loss_g_recon = criterion_recon(fake_masks_fp32, masks_fp32)
                    loss_g_boundary = criterion_boundary(fake_masks_fp32, masks_fp32)
                    pred_fake_for_g = discriminator(imgs, fake_masks_fp32)
                    loss_g_gan = criterion_bce(pred_fake_for_g, torch.ones_like(pred_fake_for_g, device=device))
                    loss_g = (
                        CONFIG["seg_lambda_gan"] * loss_g_gan
                        + CONFIG["seg_lambda_dice"] * loss_g_dice
                        + CONFIG["seg_lambda_bce"] * loss_g_recon
                        + CONFIG["seg_lambda_boundary"] * loss_g_boundary
                    )
                    loss_g_finite = bool(torch.isfinite(loss_g).item())
                    if not loss_g_finite:
                        optimizer_g.zero_grad(set_to_none=True)
                        raise FloatingPointError("Non-finite generator loss encountered during proposed segmentation training.")
                    scaler_g.scale(loss_g).backward()
                    scaler_g.unscale_(optimizer_g)
                    torch.nn.utils.clip_grad_norm_(unwrap_model(generator).parameters(), max_norm=1.0)
                    scaler_g.step(optimizer_g)
                    scaler_g.update()
                else:
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        fake_masks = generator(imgs)
                    fake_masks_fp32 = fake_masks.float()
                    masks_fp32 = masks.float()
                    loss_g_dice = criterion_dice(fake_masks_fp32, masks_fp32)
                    loss_g_recon = criterion_recon(fake_masks_fp32, masks_fp32)
                    loss_g_boundary = criterion_boundary(fake_masks_fp32, masks_fp32)
                    pred_real = discriminator(imgs, masks)
                    loss_d_real = criterion_bce(pred_real, torch.ones_like(pred_real, device=device) * 0.9)
                    pred_fake = discriminator(imgs, fake_masks_fp32)
                    loss_d_fake = criterion_bce(pred_fake, torch.zeros_like(pred_fake, device=device))
                    loss_d = 0.5 * (loss_d_real + loss_d_fake)
                    loss_g_gan = criterion_bce(pred_fake, torch.ones_like(pred_fake, device=device))
                    loss_g = (
                        CONFIG["seg_lambda_gan"] * loss_g_gan
                        + CONFIG["seg_lambda_dice"] * loss_g_dice
                        + CONFIG["seg_lambda_bce"] * loss_g_recon
                        + CONFIG["seg_lambda_boundary"] * loss_g_boundary
                    )
                    loss_d_finite = bool(torch.isfinite(loss_d).item())
                    loss_g_finite = bool(torch.isfinite(loss_g).item())
                    if not loss_d_finite:
                        raise FloatingPointError("Non-finite discriminator loss encountered during proposed segmentation validation.")
                    if not loss_g_finite:
                        raise FloatingPointError("Non-finite generator loss encountered during proposed segmentation validation.")

                pred_bin = (fake_masks_fp32 > CONFIG["seg_threshold"]).float()
                batch_dice = batch_dice_score(pred_bin, masks)

                totals["loss_g"] += float(loss_g.item())
                totals["loss_d"] += float(loss_d.item())
                totals["dice"] += batch_dice
                steps += 1

    stats = {
        "loss_g": totals["loss_g"] / max(steps, 1),
        "loss_d": totals["loss_d"] / max(steps, 1),
        "dice": totals["dice"] / max(steps, 1),
    }
    return stats


def train_proposed_segmentation(train_df, val_df, device: torch.device):
    log_header("TRAIN PROPOSED SEGMENTATION")
    train_ds = SegmentationDataset(train_df, CONFIG["seg_image_size"], train_mode=True)
    val_ds = SegmentationDataset(val_df, CONFIG["seg_image_size"], train_mode=False)
    train_loader = make_loader(train_ds, CONFIG["seg_batch_size"], shuffle=True, drop_last=True)
    val_loader = make_loader(val_ds, CONFIG["seg_batch_size"], shuffle=False)

    generator = maybe_parallel(GRHDUNet().to(device), batch_size=CONFIG["seg_batch_size"], tag="proposed generator")
    discriminator = maybe_parallel(
        PatchGANDiscriminator(in_channels=4).to(device),
        batch_size=CONFIG["seg_batch_size"],
        tag="proposed discriminator",
    )

    optimizer_g = optim.Adam(unwrap_model(generator).parameters(), lr=CONFIG["seg_lr"], betas=(CONFIG["seg_beta1"], CONFIG["seg_beta2"]))
    optimizer_d = optim.Adam(
        unwrap_model(discriminator).parameters(),
        lr=CONFIG["seg_lr"] * CONFIG["seg_disc_lr_scale"],
        betas=(CONFIG["seg_beta1"], CONFIG["seg_beta2"]),
    )

    scaler_g = torch.amp.GradScaler("cuda", enabled=CONFIG["proposed_seg_amp"])
    scaler_d = torch.amp.GradScaler("cuda", enabled=CONFIG["proposed_seg_amp"])

    manager = CheckpointManager(DIRS["checkpoints_proposed_seg"], "proposed_seg", CONFIG["save_top_k"])

    history = {
        "train_loss_g": [],
        "train_loss_d": [],
        "train_dice": [],
        "val_loss_g": [],
        "val_loss_d": [],
        "val_dice": [],
    }
    start_epoch = 0
    best_val_dice = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    print(json.dumps({
        "seg_image_size": CONFIG["seg_image_size"],
        "seg_imagenet_normalize": CONFIG["seg_imagenet_normalize"],
        "seg_lambda_gan": CONFIG["seg_lambda_gan"],
        "seg_lambda_dice": CONFIG["seg_lambda_dice"],
        "seg_lambda_bce": CONFIG["seg_lambda_bce"],
        "seg_lambda_boundary": CONFIG["seg_lambda_boundary"],
        "seg_disc_lr": CONFIG["seg_lr"] * CONFIG["seg_disc_lr_scale"],
        "seg_disc_update_interval": CONFIG["seg_disc_update_interval"],
        "seg_early_stopping_patience": CONFIG["seg_early_stopping_patience"],
        "seg_early_stopping_min_delta": CONFIG["seg_early_stopping_min_delta"],
        "seg_eval_tta": CONFIG["seg_eval_tta"],
        "seg_tune_inference_on_val": CONFIG["seg_tune_inference_on_val"],
    }, indent=2))

    if CONFIG["resume"]:
        ckpt = manager.load_latest()
        mismatches = checkpoint_config_mismatches(
            ckpt,
            [
                "seg_image_size",
                "seg_imagenet_normalize",
                "seg_lambda_gan",
                "seg_lambda_dice",
                "seg_lambda_bce",
                "seg_lambda_boundary",
                "split_strategy",
                "active_split_fingerprint",
            ],
        )
        if mismatches:
            print("Skipping resume for proposed segmentation due to config mismatch:")
            for mismatch in mismatches:
                print(f"  - {mismatch}")
            archive_checkpoint_artifacts(manager, "config_mismatch")
            ckpt = None
        if ckpt is not None:
            unwrap_model(generator).load_state_dict(ckpt["generator_state"])
            unwrap_model(discriminator).load_state_dict(ckpt["discriminator_state"])
            optimizer_g.load_state_dict(ckpt["optimizer_g"])
            optimizer_d.load_state_dict(ckpt["optimizer_d"])
            scaler_g.load_state_dict(ckpt["scaler_g"])
            scaler_d.load_state_dict(ckpt["scaler_d"])
            history = ckpt["history"]
            start_epoch = ckpt["epoch"] + 1
            restore_rng_state(ckpt.get("rng_state"))
            best_val_dice, best_epoch, epochs_without_improvement = summarize_early_stopping(
                history["val_dice"],
                CONFIG["seg_early_stopping_min_delta"],
            )
            print(f"Resuming proposed segmentation from epoch {start_epoch}")
    elif history["val_dice"]:
        best_val_dice, best_epoch, epochs_without_improvement = summarize_early_stopping(
            history["val_dice"],
            CONFIG["seg_early_stopping_min_delta"],
        )

    for epoch in range(start_epoch, CONFIG["seg_epochs"]):
        print(f"\n[Proposed Seg] Epoch {epoch + 1}/{CONFIG['seg_epochs']}")
        train_stats = run_seg_epoch_proposed(generator, discriminator, train_loader, optimizer_g, optimizer_d, scaler_g, scaler_d, device, train=True)
        val_stats = run_seg_epoch_proposed(generator, discriminator, val_loader, optimizer_g, optimizer_d, scaler_g, scaler_d, device, train=False)

        history["train_loss_g"].append(train_stats["loss_g"])
        history["train_loss_d"].append(train_stats["loss_d"])
        history["train_dice"].append(train_stats["dice"])
        history["val_loss_g"].append(val_stats["loss_g"])
        history["val_loss_d"].append(val_stats["loss_d"])
        history["val_dice"].append(val_stats["dice"])

        improved = val_stats["dice"] > best_val_dice + CONFIG["seg_early_stopping_min_delta"]
        if improved:
            best_val_dice = float(val_stats["dice"])
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(json.dumps({
            "epoch": epoch + 1,
            "train_loss_g": round(train_stats["loss_g"], 6),
            "train_loss_d": round(train_stats["loss_d"], 6),
            "train_dice": round(train_stats["dice"], 6),
            "val_loss_g": round(val_stats["loss_g"], 6),
            "val_loss_d": round(val_stats["loss_d"], 6),
            "val_dice": round(val_stats["dice"], 6),
            "best_val_dice": round(best_val_dice, 6),
            "best_epoch": best_epoch,
            "improved": improved,
            "epochs_without_improvement": epochs_without_improvement,
        }, indent=2))
        print(
            f"[Proposed Seg] Best val_dice so far: {best_val_dice:.6f} at epoch {best_epoch} "
            f"| no_improve={epochs_without_improvement}/{CONFIG['seg_early_stopping_patience']}"
        )

        payload = {
            "epoch": epoch,
            "metric": val_stats["dice"],
            "history": history,
            "generator_state": unwrap_model(generator).state_dict(),
            "discriminator_state": unwrap_model(discriminator).state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "scaler_g": scaler_g.state_dict(),
            "scaler_d": scaler_d.state_dict(),
            "rng_state": capture_rng_state(),
            "best_val_dice": best_val_dice,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "config": CONFIG,
        }
        manager.save(payload, metric=val_stats["dice"], epoch=epoch)
        save_json(manager.history_path(), history)
        save_json(DIRS["metrics"] / "proposed_seg_best_summary.json", {
            "best_val_dice": best_val_dice,
            "best_epoch": best_epoch,
            "last_epoch": epoch + 1,
            "last_val_dice": val_stats["dice"],
            "epochs_without_improvement": epochs_without_improvement,
            "early_stopping_patience": CONFIG["seg_early_stopping_patience"],
            "seg_image_size": CONFIG["seg_image_size"],
            "seg_lambda_gan": CONFIG["seg_lambda_gan"],
            "seg_lambda_bce": CONFIG["seg_lambda_bce"],
            "seg_lambda_boundary": CONFIG["seg_lambda_boundary"],
            "seg_disc_lr": CONFIG["seg_lr"] * CONFIG["seg_disc_lr_scale"],
            "seg_disc_update_interval": CONFIG["seg_disc_update_interval"],
        })

        if epochs_without_improvement >= CONFIG["seg_early_stopping_patience"]:
            print(
                f"[Proposed Seg] Early stopping triggered at epoch {epoch + 1}. "
                f"Best val_dice={best_val_dice:.6f} at epoch {best_epoch}."
            )
            break

    best = manager.load_best() or manager.load_latest()
    ensure(best is not None, "No proposed segmentation checkpoint available after training.")
    unwrap_model(generator).load_state_dict(best["generator_state"])
    unwrap_model(discriminator).load_state_dict(best["discriminator_state"])

    torch.save(unwrap_model(generator).state_dict(), DIRS["models_final"] / "proposed_generator.pt")
    torch.save(unwrap_model(discriminator).state_dict(), DIRS["models_final"] / "proposed_discriminator.pt")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_dice"], label="Train Dice")
    ax.plot(history["val_dice"], label="Val Dice")
    ax.set_title("Proposed segmentation Dice")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")
    ax.legend()
    plot_and_save(fig, DIRS["plots"] / "proposed_seg_dice_curve.png")

    inference_cfg = tune_segmentation_inference("proposed_methodology", generator, val_df, device)
    return generator, inference_cfg


def run_seg_epoch_baseline(model, loader, optimizer, scaler, device, train: bool):
    criterion_bce = nn.BCELoss()
    criterion_dice = DiceLoss()
    model.train() if train else model.eval()
    totals = defaultdict(float)
    steps = 0

    context = torch.enable_grad if train else torch.inference_mode
    with context():
        with make_tqdm(loader, leave=False, desc=f"{'train' if train else 'val'} baseline seg") as pbar:
            for imgs, masks, _, _ in pbar:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                if train:
                    optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=CONFIG["amp"]):
                    preds = model(imgs)
                preds_fp32 = preds.float()
                masks_fp32 = masks.float()
                loss = criterion_bce(preds_fp32, masks_fp32) + criterion_dice(preds_fp32, masks_fp32)

                if train:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                pred_bin = (preds_fp32 > CONFIG["seg_threshold"]).float()
                totals["loss"] += float(loss.item())
                totals["dice"] += batch_dice_score(pred_bin, masks)
                steps += 1

    return {k: v / max(steps, 1) for k, v in totals.items()}


def train_baseline_segmentation(name: str, model: nn.Module, train_df, val_df, device: torch.device):
    log_header(f"TRAIN BASELINE SEGMENTATION: {name}")
    train_ds = SegmentationDataset(train_df, CONFIG["seg_image_size"], train_mode=True)
    val_ds = SegmentationDataset(val_df, CONFIG["seg_image_size"], train_mode=False)
    train_loader = make_loader(train_ds, CONFIG["baseline_seg_batch_size"], shuffle=True)
    val_loader = make_loader(val_ds, CONFIG["baseline_seg_batch_size"], shuffle=False)

    model = maybe_parallel(model.to(device), batch_size=CONFIG["baseline_seg_batch_size"], tag=f"{name} baseline")
    optimizer = optim.Adam(unwrap_model(model).parameters(), lr=CONFIG["baseline_seg_lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=CONFIG["amp"])
    save_dir = {
        "U-Net": DIRS["checkpoints_unet"],
        "SegNet": DIRS["checkpoints_segnet"],
        "MedT": DIRS["checkpoints_medt"],
    }[name]
    manager = CheckpointManager(save_dir, name.lower().replace("-", "_"), CONFIG["save_top_k"])

    history = {"train_loss": [], "train_dice": [], "val_loss": [], "val_dice": []}
    start_epoch = 0
    best_val_dice = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    print(json.dumps({
        "seg_image_size": CONFIG["seg_image_size"],
        "baseline_seg_early_stopping_patience": CONFIG["baseline_seg_early_stopping_patience"],
        "baseline_seg_early_stopping_min_delta": CONFIG["baseline_seg_early_stopping_min_delta"],
    }, indent=2))

    if CONFIG["resume"]:
        ckpt = manager.load_latest()
        mismatches = checkpoint_config_mismatches(
            ckpt,
            [
                "seg_image_size",
                "seg_imagenet_normalize",
                "seg_threshold",
                "split_strategy",
                "active_split_fingerprint",
            ],
        )
        if mismatches:
            print(f"Skipping resume for {name} due to config mismatch:")
            for mismatch in mismatches:
                print(f"  - {mismatch}")
            archive_checkpoint_artifacts(manager, "config_mismatch")
            ckpt = None
        if ckpt is not None:
            unwrap_model(model).load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scaler.load_state_dict(ckpt["scaler"])
            history = ckpt["history"]
            start_epoch = ckpt["epoch"] + 1
            restore_rng_state(ckpt.get("rng_state"))
            best_val_dice, best_epoch, epochs_without_improvement = summarize_early_stopping(
                history["val_dice"],
                CONFIG["baseline_seg_early_stopping_min_delta"],
            )
            print(f"Resuming {name} from epoch {start_epoch}")
    elif history["val_dice"]:
        best_val_dice, best_epoch, epochs_without_improvement = summarize_early_stopping(
            history["val_dice"],
            CONFIG["baseline_seg_early_stopping_min_delta"],
        )

    for epoch in range(start_epoch, CONFIG["baseline_seg_epochs"]):
        print(f"\n[{name}] Epoch {epoch + 1}/{CONFIG['baseline_seg_epochs']}")
        train_stats = run_seg_epoch_baseline(model, train_loader, optimizer, scaler, device, train=True)
        val_stats = run_seg_epoch_baseline(model, val_loader, optimizer, scaler, device, train=False)

        history["train_loss"].append(train_stats["loss"])
        history["train_dice"].append(train_stats["dice"])
        history["val_loss"].append(val_stats["loss"])
        history["val_dice"].append(val_stats["dice"])

        improved = val_stats["dice"] > best_val_dice + CONFIG["baseline_seg_early_stopping_min_delta"]
        if improved:
            best_val_dice = float(val_stats["dice"])
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(json.dumps({
            "epoch": epoch + 1,
            "train_loss": round(train_stats["loss"], 6),
            "train_dice": round(train_stats["dice"], 6),
            "val_loss": round(val_stats["loss"], 6),
            "val_dice": round(val_stats["dice"], 6),
            "best_val_dice": round(best_val_dice, 6),
            "best_epoch": best_epoch,
            "improved": improved,
            "epochs_without_improvement": epochs_without_improvement,
        }, indent=2))
        print(
            f"[{name}] Best val_dice so far: {best_val_dice:.6f} at epoch {best_epoch} "
            f"| no_improve={epochs_without_improvement}/{CONFIG['baseline_seg_early_stopping_patience']}"
        )

        payload = {
            "epoch": epoch,
            "metric": val_stats["dice"],
            "history": history,
            "model_state": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "rng_state": capture_rng_state(),
            "best_val_dice": best_val_dice,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "config": CONFIG,
        }
        manager.save(payload, metric=val_stats["dice"], epoch=epoch)
        save_json(manager.history_path(), history)
        save_json(DIRS["metrics"] / f"{name.lower().replace('-', '_')}_best_summary.json", {
            "best_val_dice": best_val_dice,
            "best_epoch": best_epoch,
            "last_epoch": epoch + 1,
            "last_val_dice": val_stats["dice"],
            "epochs_without_improvement": epochs_without_improvement,
            "early_stopping_patience": CONFIG["baseline_seg_early_stopping_patience"],
            "seg_image_size": CONFIG["seg_image_size"],
        })

        if epochs_without_improvement >= CONFIG["baseline_seg_early_stopping_patience"]:
            print(
                f"[{name}] Early stopping triggered at epoch {epoch + 1}. "
                f"Best val_dice={best_val_dice:.6f} at epoch {best_epoch}."
            )
            break

    best = manager.load_best() or manager.load_latest()
    ensure(best is not None, f"No {name} checkpoint available after training.")
    unwrap_model(model).load_state_dict(best["model_state"])
    torch.save(unwrap_model(model).state_dict(), DIRS["models_final"] / f"{name.lower().replace('-', '_')}.pt")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_dice"], label="Train Dice")
    ax.plot(history["val_dice"], label="Val Dice")
    ax.set_title(f"{name} segmentation Dice")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")
    ax.legend()
    plot_and_save(fig, DIRS["plots"] / f"{name.lower().replace('-', '_')}_seg_dice_curve.png")

    inference_cfg = tune_segmentation_inference(name.lower().replace("-", "_").replace(" ", "_"), model, val_df, device)
    return model, inference_cfg


# -----------------------------------------------------------------------------
# segmentation inference and evaluation
# -----------------------------------------------------------------------------
def generate_refined_masks(
    split_name: str,
    df: pd.DataFrame,
    generator: nn.Module,
    device: torch.device,
    inference_cfg: Dict | None = None,
):
    log_header(f"GENERATE REFINED MASKS: {split_name}")
    inference_cfg = inference_cfg or default_segmentation_inference_config("proposed_methodology")
    out_dir = {
        "train": DIRS["masks_train"],
        "val": DIRS["masks_val"],
        "test": DIRS["masks_test"],
    }[split_name]
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in out_dir.rglob("*.png"):
        stale_path.unlink()
    for folder in CONFIG["class_folders"]:
        (out_dir / folder).mkdir(parents=True, exist_ok=True)

    ds = SegmentationDataset(df, CONFIG["seg_image_size"], train_mode=False)
    loader = make_loader(ds, batch_size=CONFIG["seg_batch_size"], shuffle=False)

    generator.eval()
    with torch.inference_mode():
        with make_tqdm(loader, desc=f"Generating masks {split_name}", leave=False) as pbar:
            for imgs, _, class_idx, filenames in pbar:
                imgs = imgs.to(device, non_blocking=True)
                probs = predict_segmentation_probs(generator, imgs)
                preds = binarize_segmentation_probs(probs.cpu().numpy(), inference_cfg)

                for i in range(preds.shape[0]):
                    folder = CONFIG["class_folders"][int(class_idx[i])]
                    save_path = out_dir / folder / filenames[i]
                    mask_u8 = (preds[i, 0] * 255).astype(np.uint8)
                    Image.fromarray(mask_u8).save(save_path)

    counts = sum(1 for _ in out_dir.rglob("*.png"))
    ensure(counts == len(df), f"Generated mask count mismatch for {split_name}. Expected {len(df)}, got {counts}")


def evaluate_segmentation_model(
    name: str,
    model: nn.Module,
    df_test: pd.DataFrame,
    device: torch.device,
    inference_cfg: Dict | None = None,
):
    log_header(f"EVALUATE SEGMENTATION: {name}")
    inference_cfg = inference_cfg or default_segmentation_inference_config(name.lower().replace("-", "_").replace(" ", "_"))
    ds = SegmentationDataset(df_test, CONFIG["seg_image_size"], train_mode=False)
    loader = make_loader(ds, batch_size=32, shuffle=False)

    per_class = {
        disp: {"Dice": [], "Jaccard": [], "Conformity Coefficient": [], "BF-score": [], "HOG-sim": []}
        for disp in CONFIG["class_folders"]
    }

    sample_by_class = {}
    model.eval()

    with torch.inference_mode():
        with make_tqdm(loader, desc=f"Evaluating {name}", leave=False) as pbar:
            for imgs, masks, class_idx, filenames in pbar:
                imgs = imgs.to(device, non_blocking=True)
                masks_np = masks.numpy().astype(np.uint8)

                probs = predict_segmentation_probs(model, imgs)
                preds_np = binarize_segmentation_probs(probs.cpu().numpy(), inference_cfg).astype(np.uint8)

                for i in range(preds_np.shape[0]):
                    disp = CONFIG["class_folders"][int(class_idx[i])]
                    p = preds_np[i, 0]
                    g = masks_np[i, 0]
                    per_class[disp]["Dice"].append(compute_dice_np(p, g))
                    per_class[disp]["Jaccard"].append(compute_jaccard_np(p, g))
                    per_class[disp]["Conformity Coefficient"].append(compute_conformity_np(p, g))
                    per_class[disp]["BF-score"].append(compute_bfscore_np(p, g))
                    per_class[disp]["HOG-sim"].append(compute_hog_similarity_np(p, g))

                    if disp not in sample_by_class:
                        sample_by_class[disp] = {
                            "filename": filenames[i],
                            "image": tensor_to_display_image(imgs[i], imagenet_normalize=CONFIG["seg_imagenet_normalize"]),
                            "gt": g,
                            "pred": p,
                        }

    rows = []
    for disp in CONFIG["class_folders"]:
        rows.append({
            "Types of Images": disp,
            "Deep Learning Model": name,
            "Dice Coefficient": np.mean(per_class[disp]["Dice"]),
            "Jaccard Index": np.mean(per_class[disp]["Jaccard"]),
            "Conformity Coefficient": np.mean(per_class[disp]["Conformity Coefficient"]),
            "BF-score": np.mean(per_class[disp]["BF-score"]),
            "HOG-sim": np.mean(per_class[disp]["HOG-sim"]),
        })

    table_df = pd.DataFrame(rows)
    save_df(DIRS["tables"] / f"table2_{name.lower().replace('-', '_')}.csv", table_df)

    # qualitative figure like paper Fig. 4
    fig, axes = plt.subplots(len(CONFIG["class_folders"]), 4, figsize=(14, 3 * len(CONFIG["class_folders"])))
    for r, disp in enumerate(CONFIG["class_folders"]):
        sample = sample_by_class[disp]
        img = sample["image"]
        gt = sample["gt"]
        pred = sample["pred"]
        overlay = img.copy()
        overlay[..., 1] = np.clip(overlay[..., 1] + pred * 0.5, 0, 1)
        axes[r, 0].imshow(img); axes[r, 0].set_title(f"{disp} - image"); axes[r, 0].axis("off")
        axes[r, 1].imshow(gt, cmap="gray"); axes[r, 1].set_title("Ground truth"); axes[r, 1].axis("off")
        axes[r, 2].imshow(pred, cmap="gray"); axes[r, 2].set_title(f"{name} mask"); axes[r, 2].axis("off")
        axes[r, 3].imshow(overlay); axes[r, 3].set_title("Overlay"); axes[r, 3].axis("off")
    plot_and_save(fig, DIRS["plots"] / f"figure4_{name.lower().replace('-', '_')}_qualitative.png")

    return table_df


# -----------------------------------------------------------------------------
# classification training
# -----------------------------------------------------------------------------
def save_epoch_probs(model_name: str, epoch: int, probs: np.ndarray, labels: np.ndarray):
    model_dir = DIRS["epoch_probs"] / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    np.save(model_dir / f"val_probs_epoch_{epoch:03d}.npy", probs)
    np.save(model_dir / "val_labels.npy", labels)


def run_cls_epoch(model, loader, optimizer, scheduler, scaler, criterion, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    probs_all, labels_all = [], []

    context = torch.enable_grad if train else torch.inference_mode
    with context():
        with make_tqdm(loader, desc=f"{'train' if train else 'val'} cls", leave=False) as pbar:
            for imgs, labels, _ in pbar:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if train:
                    optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=CONFIG["amp"]):
                    logits = model(imgs)
                    loss = criterion(logits, labels)

                if train:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                probs = logits_to_probabilities(logits)
                preds = logits.argmax(dim=1)
                total_loss += float(loss.item()) * labels.size(0)
                correct += int((preds == labels).sum().item())
                total += int(labels.size(0))

                probs_all.append(probs.detach().float().cpu().numpy())
                labels_all.append(labels.detach().cpu().numpy())

    if train:
        scheduler.step()

    probs_all = np.concatenate(probs_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    acc = 100.0 * correct / max(total, 1)
    return {
        "loss": total_loss / max(total, 1),
        "acc": acc,
        "probs": probs_all,
        "labels": labels_all,
    }


def train_classifier(model_name: str, build_fn, train_df, val_df, mask_dirs, device: torch.device):
    log_header(f"TRAIN CLASSIFIER: {model_name}")
    train_ds = ClassificationDataset(train_df, mask_dirs["train"], CONFIG["cls_image_size"], train_mode=True)
    val_ds = ClassificationDataset(val_df, mask_dirs["val"], CONFIG["cls_image_size"], train_mode=False)

    train_class_indices = train_df["class_index"].to_numpy(dtype=np.int64)
    balanced_weights = compute_balanced_class_weights(train_class_indices, CONFIG["expected_num_classes"])
    class_weight_tensor = None
    if CONFIG["cls_use_class_weights"]:
        class_weight_tensor = torch.tensor(balanced_weights, dtype=torch.float32, device=device)

    train_sampler = None
    if CONFIG["cls_use_weighted_sampler"]:
        sample_weights = torch.as_tensor(balanced_weights[train_class_indices], dtype=torch.double)
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = make_loader(train_ds, CONFIG["cls_batch_size"], shuffle=True, sampler=train_sampler)
    val_loader = make_loader(val_ds, CONFIG["cls_batch_size"], shuffle=False)

    model = maybe_parallel(build_fn(CONFIG["expected_num_classes"]).to(device), batch_size=CONFIG["cls_batch_size"], tag=model_name)
    optimizer = optim.AdamW(unwrap_model(model).parameters(), lr=CONFIG["cls_lr"], weight_decay=CONFIG["cls_weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["cls_epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=CONFIG["amp"])
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)

    save_dir = {
        "vit": DIRS["checkpoints_vit"],
        "swin": DIRS["checkpoints_swin"],
        "convnext": DIRS["checkpoints_convnext"],
    }[model_name]
    manager = CheckpointManager(save_dir, model_name, CONFIG["save_top_k"])

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start_epoch = 0
    best_val_acc = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    print(json.dumps({
        "cls_image_size": CONFIG["cls_image_size"],
        "cls_imagenet_normalize": CONFIG["cls_imagenet_normalize"],
        "cls_mask_background_alpha": CONFIG["cls_mask_background_alpha"],
        "cls_mask_blur_kernel": CONFIG["cls_mask_blur_kernel"],
        "cls_early_stopping_patience": CONFIG["cls_early_stopping_patience"],
        "cls_early_stopping_min_delta": CONFIG["cls_early_stopping_min_delta"],
        "cls_use_class_weights": CONFIG["cls_use_class_weights"],
        "cls_use_weighted_sampler": CONFIG["cls_use_weighted_sampler"],
        "class_weights": {CONFIG["class_folders"][i]: round(float(w), 4) for i, w in enumerate(balanced_weights)},
    }, indent=2))

    if CONFIG["resume"]:
        ckpt = manager.load_latest()
        mismatches = checkpoint_config_mismatches(
            ckpt,
            [
                "seg_image_size",
                "cls_image_size",
                "cls_imagenet_normalize",
                "cls_mask_background_alpha",
                "cls_mask_blur_kernel",
                "cls_use_class_weights",
                "cls_use_weighted_sampler",
                "split_strategy",
                "active_split_fingerprint",
            ],
        )
        if mismatches:
            print(f"Skipping resume for {model_name} due to config mismatch:")
            for mismatch in mismatches:
                print(f"  - {mismatch}")
            archive_checkpoint_artifacts(manager, "config_mismatch")
            ckpt = None
        if ckpt is not None:
            unwrap_model(model).load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
            history = ckpt["history"]
            start_epoch = ckpt["epoch"] + 1
            restore_rng_state(ckpt.get("rng_state"))
            best_val_acc, best_epoch, epochs_without_improvement = summarize_early_stopping(
                history["val_acc"],
                CONFIG["cls_early_stopping_min_delta"],
            )
            print(f"Resuming {model_name} from epoch {start_epoch}")
    elif history["val_acc"]:
        best_val_acc, best_epoch, epochs_without_improvement = summarize_early_stopping(
            history["val_acc"],
            CONFIG["cls_early_stopping_min_delta"],
        )

    for epoch in range(start_epoch, CONFIG["cls_epochs"]):
        print(f"\n[{model_name}] Epoch {epoch + 1}/{CONFIG['cls_epochs']}")
        train_stats = run_cls_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, device, train=True)
        val_stats = run_cls_epoch(model, val_loader, optimizer, scheduler, scaler, criterion, device, train=False)

        history["train_loss"].append(train_stats["loss"])
        history["train_acc"].append(train_stats["acc"])
        history["val_loss"].append(val_stats["loss"])
        history["val_acc"].append(val_stats["acc"])

        save_epoch_probs(model_name, epoch, val_stats["probs"], val_stats["labels"])

        improved = val_stats["acc"] > best_val_acc + CONFIG["cls_early_stopping_min_delta"]
        if improved:
            best_val_acc = float(val_stats["acc"])
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(json.dumps({
            "epoch": epoch + 1,
            "train_loss": round(train_stats["loss"], 6),
            "train_acc": round(train_stats["acc"], 4),
            "val_loss": round(val_stats["loss"], 6),
            "val_acc": round(val_stats["acc"], 4),
            "best_val_acc": round(best_val_acc, 4),
            "best_epoch": best_epoch,
            "improved": improved,
            "epochs_without_improvement": epochs_without_improvement,
        }, indent=2))
        print(
            f"[{model_name}] Best val_acc so far: {best_val_acc:.4f} at epoch {best_epoch} "
            f"| no_improve={epochs_without_improvement}/{CONFIG['cls_early_stopping_patience']}"
        )

        payload = {
            "epoch": epoch,
            "metric": val_stats["acc"],
            "history": history,
            "model_state": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng_state": capture_rng_state(),
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "config": CONFIG,
        }
        manager.save(payload, metric=val_stats["acc"], epoch=epoch)
        save_json(manager.history_path(), history)
        save_json(DIRS["metrics"] / f"{model_name}_best_summary.json", {
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "last_epoch": epoch + 1,
            "last_val_acc": val_stats["acc"],
            "epochs_without_improvement": epochs_without_improvement,
            "early_stopping_patience": CONFIG["cls_early_stopping_patience"],
            "cls_image_size": CONFIG["cls_image_size"],
        })

        if epochs_without_improvement >= CONFIG["cls_early_stopping_patience"]:
            print(
                f"[{model_name}] Early stopping triggered at epoch {epoch + 1}. "
                f"Best val_acc={best_val_acc:.4f} at epoch {best_epoch}."
            )
            break

    best = manager.load_best() or manager.load_latest()
    ensure(best is not None, f"No {model_name} checkpoint available after training.")
    unwrap_model(model).load_state_dict(best["model_state"])
    torch.save(unwrap_model(model).state_dict(), DIRS["models_final"] / f"{model_name}.pt")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_acc"], label="Train Accuracy")
    ax.plot(history["val_acc"], label="Val Accuracy")
    ax.set_title(f"{model_name} accuracy vs epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    plot_and_save(fig, DIRS["plots"] / f"{model_name}_accuracy_curve.png")

    save_json(DIRS["histories"] / f"{model_name}_history.json", history)
    return model


# -----------------------------------------------------------------------------
# classification evaluation
# -----------------------------------------------------------------------------
def get_test_predictions(model_name: str, model: nn.Module, test_df, mask_dir: Path, device: torch.device):
    ds = ClassificationDataset(test_df, mask_dir, CONFIG["cls_image_size"], train_mode=False)
    loader = make_loader(ds, CONFIG["cls_batch_size"], shuffle=False)

    model.eval()
    probs_all, preds_all, y_all, filenames_all = [], [], [], []
    with torch.inference_mode():
        with make_tqdm(loader, desc=f"Test {model_name}", leave=False) as pbar:
            for imgs, labels, filenames in pbar:
                imgs = imgs.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=CONFIG["amp"]):
                    logits = model(imgs)
                probs = logits_to_probabilities(logits)
                preds = logits.argmax(dim=1)

                probs_all.append(probs.float().cpu().numpy())
                preds_all.append(preds.cpu().numpy())
                y_all.append(labels.numpy())
                filenames_all.extend(list(filenames))

    probs_all = np.concatenate(probs_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    pred_df = pd.DataFrame({
        "filename": filenames_all,
        "y_true": y_all,
        "y_pred": preds_all,
    })
    for i, cname in enumerate(CONFIG["class_folders"]):
        pred_df[f"prob_{cname}"] = probs_all[:, i]
    save_df(DIRS["predictions"] / f"{model_name}_test_predictions.csv", pred_df)
    return y_all, preds_all, probs_all


def majority_vote(preds_list: List[np.ndarray], probs_list: List[np.ndarray]) -> np.ndarray:
    preds_stack = np.stack(preds_list, axis=1)  # N, M
    out = []
    for i in range(preds_stack.shape[0]):
        votes = preds_stack[i]
        values, counts = np.unique(votes, return_counts=True)
        max_count = counts.max()
        winners = values[counts == max_count]
        if len(winners) == 1:
            out.append(int(winners[0]))
        else:
            best_class = None
            best_conf = -1.0
            for c in winners:
                for probs in probs_list:
                    conf = probs[i, int(c)]
                    if conf > best_conf:
                        best_conf = conf
                        best_class = int(c)
            out.append(best_class)
    return np.array(out)


def ensemble_selection_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    metric = CONFIG["ensemble_selection_metric"]
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if metric == "weighted_f1":
        return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    raise ValueError(f"Unsupported ensemble_selection_metric: {metric}")


def weighted_probability_average(probs_by_model: Dict[str, np.ndarray], weights: np.ndarray) -> np.ndarray:
    ordered = ["vit", "swin", "convnext"]
    mix = np.zeros_like(probs_by_model[ordered[0]], dtype=np.float64)
    for idx, name in enumerate(ordered):
        mix += float(weights[idx]) * probs_by_model[name]
    return normalize_probability_rows(mix)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def load_best_validation_probabilities(model_name: str) -> Tuple[np.ndarray, np.ndarray, int]:
    summary_path = DIRS["metrics"] / f"{model_name}_best_summary.json"
    ensure(summary_path.exists(), f"Best summary not found for {model_name}: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    best_epoch = int(summary["best_epoch"])
    zero_indexed_epoch = best_epoch - 1
    model_dir = DIRS["epoch_probs"] / model_name
    probs_path = model_dir / f"val_probs_epoch_{zero_indexed_epoch:03d}.npy"
    labels_path = model_dir / "val_labels.npy"
    ensure(probs_path.exists(), f"Validation probability file missing for {model_name}: {probs_path}")
    ensure(labels_path.exists(), f"Validation labels missing for {model_name}: {labels_path}")
    return np.load(probs_path), np.load(labels_path), best_epoch


def fit_logreg_stacking(val_probs: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict:
    ordered = ["vit", "swin", "convnext"]
    x_val = np.concatenate([val_probs[name] for name in ordered], axis=1)
    stacker = LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        C=float(CONFIG["ensemble_logreg_c"]),
    )
    stacker.fit(x_val, y_true)

    classes = np.asarray(stacker.classes_, dtype=np.int64)
    ensure(
        np.array_equal(classes, np.arange(CONFIG["expected_num_classes"])),
        f"Unexpected meta-learner class order: {classes.tolist()}",
    )
    val_pred_probs = stacker.predict_proba(x_val)
    val_pred = val_pred_probs.argmax(axis=1)
    return {
        "name": "Ensemble (Stacking LogReg)",
        "kind": "stacking_logreg",
        "coef": np.asarray(stacker.coef_, dtype=np.float64),
        "intercept": np.asarray(stacker.intercept_, dtype=np.float64),
        "val_score": ensemble_selection_score(y_true, val_pred),
        "val_metrics": classification_metrics(y_true, val_pred, val_pred_probs),
    }


def fit_ensemble_from_validation() -> Tuple[Dict, List[Dict]]:
    model_names = ["vit", "swin", "convnext"]
    val_probs = {}
    val_preds = {}
    y_ref = None
    best_epochs = {}

    for name in model_names:
        probs, labels, best_epoch = load_best_validation_probabilities(name)
        if y_ref is None:
            y_ref = labels
        else:
            ensure(np.array_equal(y_ref, labels), "Validation labels mismatch across classifiers.")
        val_probs[name] = normalize_probability_rows(probs)
        val_preds[name] = val_probs[name].argmax(axis=1)
        best_epochs[name] = best_epoch

    candidates = []

    equal_weights = np.asarray(CONFIG["ensemble_weights"], dtype=np.float64)
    equal_probs = weighted_probability_average(val_probs, equal_weights)
    equal_pred = equal_probs.argmax(axis=1)
    candidates.append({
        "name": "Ensemble (Weighted Avg - Equal)",
        "kind": "weighted_avg",
        "weights": equal_weights.tolist(),
        "val_score": ensemble_selection_score(y_ref, equal_pred),
        "val_metrics": classification_metrics(y_ref, equal_pred, equal_probs),
    })

    step = float(CONFIG["ensemble_weight_search_step"])
    values = np.arange(0.0, 1.0 + 1e-9, step)
    best_weighted = None
    for w_vit in values:
        for w_swin in values:
            w_conv = 1.0 - w_vit - w_swin
            if w_conv < -1e-9 or w_conv > 1.0 + 1e-9:
                continue
            weights = np.asarray([w_vit, w_swin, w_conv], dtype=np.float64)
            probs = weighted_probability_average(val_probs, weights)
            pred = probs.argmax(axis=1)
            score = ensemble_selection_score(y_ref, pred)
            if best_weighted is None or score > best_weighted["val_score"]:
                best_weighted = {
                    "name": "Ensemble (Weighted Avg)",
                    "kind": "weighted_avg",
                    "weights": weights.tolist(),
                    "val_score": score,
                    "val_metrics": classification_metrics(y_ref, pred, probs),
                }
    ensure(best_weighted is not None, "Weighted ensemble search failed to produce a candidate.")
    candidates.append(best_weighted)

    mv_pred = majority_vote(
        [val_preds["vit"], val_preds["swin"], val_preds["convnext"]],
        [val_probs["vit"], val_probs["swin"], val_probs["convnext"]],
    )
    candidates.append({
        "name": "Ensemble (Majority Voting)",
        "kind": "majority_vote",
        "val_score": ensemble_selection_score(y_ref, mv_pred),
        "val_metrics": classification_metrics(y_ref, mv_pred, None),
    })

    if CONFIG["ensemble_fit_logreg"]:
        candidates.append(fit_logreg_stacking(val_probs, y_ref))

    selected = max(candidates, key=lambda item: item["val_score"])
    save_json(
        DIRS["metrics"] / "ensemble_selection_summary.json",
        {
            "selection_metric": CONFIG["ensemble_selection_metric"],
            "best_epochs": best_epochs,
            "selected_model": selected["name"],
            "candidates": [
                {
                    "name": cand["name"],
                    "kind": cand["kind"],
                    "val_score": cand["val_score"],
                    "weights": cand.get("weights"),
                    "val_metrics": cand["val_metrics"],
                }
                for cand in candidates
            ],
        },
    )
    return selected, candidates


def apply_ensemble_spec(spec: Dict, probs: Dict[str, np.ndarray], preds: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray | None]:
    if spec["kind"] == "weighted_avg":
        weights = np.asarray(spec["weights"], dtype=np.float64)
        probs_out = weighted_probability_average(probs, weights)
        return probs_out.argmax(axis=1), probs_out

    if spec["kind"] == "majority_vote":
        pred = majority_vote([preds["vit"], preds["swin"], preds["convnext"]], [probs["vit"], probs["swin"], probs["convnext"]])
        return pred, None

    if spec["kind"] == "stacking_logreg":
        x = np.concatenate([probs["vit"], probs["swin"], probs["convnext"]], axis=1)
        logits = x @ np.asarray(spec["coef"], dtype=np.float64).T + np.asarray(spec["intercept"], dtype=np.float64)
        probs_out = softmax_np(logits)
        return probs_out.argmax(axis=1), probs_out

    raise ValueError(f"Unsupported ensemble spec kind: {spec['kind']}")


def save_confusion_matrix(y_true, y_pred, title: str, filename: str):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(CONFIG["expected_num_classes"])))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CONFIG["class_folders"], yticklabels=CONFIG["class_folders"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plot_and_save(fig, DIRS["plots"] / filename)


def save_roc_comparison(y_true: np.ndarray, probs_proposed: np.ndarray, probs_vit_swin: np.ndarray):
    y_true_bin = np.eye(CONFIG["expected_num_classes"])[y_true]
    fpr1, tpr1, _ = roc_curve(y_true_bin.ravel(), probs_proposed.ravel())
    fpr2, tpr2, _ = roc_curve(y_true_bin.ravel(), probs_vit_swin.ravel())
    auc1 = auc(fpr1, tpr1)
    auc2 = auc(fpr2, tpr2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr1, tpr1, label=f"Proposed Ensemble (AUC={auc1:.3f})")
    ax.plot(fpr2, tpr2, label=f"ViT + Swin baseline (AUC={auc2:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Figure 6: ROC curve comparison")
    ax.legend(loc="lower right")
    plot_and_save(fig, DIRS["plots"] / "figure6_roc_comparison.png")


def save_epoch_ensemble_accuracy_curves(weighted_avg_weights: np.ndarray | None = None):
    model_names = ["vit", "swin", "convnext"]
    min_epoch_count = CONFIG["cls_epochs"]
    probs_per_model = {}
    labels_ref = None
    weights = np.asarray(weighted_avg_weights if weighted_avg_weights is not None else CONFIG["ensemble_weights"], dtype=np.float64)

    for name in model_names:
        model_dir = DIRS["epoch_probs"] / name
        epoch_files = sorted(model_dir.glob("val_probs_epoch_*.npy"))
        ensure(len(epoch_files) >= 1, f"No saved epoch probabilities found for {name}")
        probs_per_model[name] = epoch_files
        min_epoch_count = min(min_epoch_count, len(epoch_files))
        labels = np.load(model_dir / "val_labels.npy")
        if labels_ref is None:
            labels_ref = labels
        else:
            ensure(np.array_equal(labels_ref, labels), "Validation labels mismatch across classifiers.")

    vit_acc, swin_acc, conv_acc, wa_acc, mv_acc = [], [], [], [], []
    for epoch in range(min_epoch_count):
        pv = np.load(probs_per_model["vit"][epoch])
        ps = np.load(probs_per_model["swin"][epoch])
        pc = np.load(probs_per_model["convnext"][epoch])

        vit_pred = pv.argmax(axis=1)
        swin_pred = ps.argmax(axis=1)
        conv_pred = pc.argmax(axis=1)
        wa_probs = weights[0] * pv + weights[1] * ps + weights[2] * pc
        wa_pred = wa_probs.argmax(axis=1)
        mv_pred = majority_vote([vit_pred, swin_pred, conv_pred], [pv, ps, pc])

        vit_acc.append(100.0 * accuracy_score(labels_ref, vit_pred))
        swin_acc.append(100.0 * accuracy_score(labels_ref, swin_pred))
        conv_acc.append(100.0 * accuracy_score(labels_ref, conv_pred))
        wa_acc.append(100.0 * accuracy_score(labels_ref, wa_pred))
        mv_acc.append(100.0 * accuracy_score(labels_ref, mv_pred))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(vit_acc, label="ViT")
    ax.plot(swin_acc, label="Swin Transformer")
    ax.plot(conv_acc, label="ConvNeXt")
    ax.plot(wa_acc, label="Ensemble (Weighted Avg)")
    ax.plot(mv_acc, label="Ensemble (Majority Voting)")
    ax.set_title("Figure 5: Accuracy vs Epoch performance evaluation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.legend()
    plot_and_save(fig, DIRS["plots"] / "figure5_accuracy_vs_epoch.png")


def evaluate_classifiers(models_dict: Dict[str, nn.Module], test_df: pd.DataFrame, mask_dirs, device: torch.device):
    log_header("EVALUATE CLASSIFIERS")
    results_rows = []
    y_ref = None
    preds = {}
    probs = {}

    for name, model in models_dict.items():
        y_true, y_pred, y_prob = get_test_predictions(name, model, test_df, mask_dirs["test"], device)
        if y_ref is None:
            y_ref = y_true
        else:
            ensure(np.array_equal(y_ref, y_true), "Test labels mismatch across classifiers.")
        preds[name] = y_pred
        probs[name] = y_prob

        metrics = classification_metrics(y_true, y_pred, y_prob)
        metrics["Model"] = {
            "vit": "ViT",
            "swin": "Swin Transformer",
            "convnext": "ConvNeXt",
        }[name]
        results_rows.append(metrics)
        save_confusion_matrix(y_true, y_pred, f"{metrics['Model']} confusion matrix", f"{name}_confusion_matrix.png")

    selected_spec, ensemble_specs = fit_ensemble_from_validation()
    ensemble_outputs = {}
    weighted_curve_weights = None
    selected_metrics = None

    print(json.dumps({
        "ensemble_selection_metric": CONFIG["ensemble_selection_metric"],
        "selected_ensemble": selected_spec["name"],
        "candidates": [
            {
                "name": spec["name"],
                "kind": spec["kind"],
                "val_score": round(float(spec["val_score"]), 6),
                "weights": spec.get("weights"),
            }
            for spec in ensemble_specs
        ],
    }, indent=2))

    for spec in ensemble_specs:
        pred, prob = apply_ensemble_spec(spec, probs, preds)
        metrics = classification_metrics(y_ref, pred, prob)
        metrics["Model"] = spec["name"]
        metrics["Validation selection score"] = spec["val_score"]
        results_rows.append(metrics)
        ensemble_outputs[spec["name"]] = {"pred": pred, "prob": prob, "metrics": metrics}

        if spec["name"] == "Ensemble (Weighted Avg)":
            weighted_curve_weights = np.asarray(spec["weights"], dtype=np.float64)
            save_confusion_matrix(y_ref, pred, "Weighted average ensemble confusion matrix", "ensemble_weighted_confusion_matrix.png")
            save_json(DIRS["metrics"] / "weighted_ensemble_classification_report.json", classification_report(
                y_ref, pred, target_names=CONFIG["class_folders"], output_dict=True, zero_division=0
            ))
        if spec["name"] == "Ensemble (Majority Voting)":
            save_confusion_matrix(y_ref, pred, "Majority voting ensemble confusion matrix", "ensemble_majority_confusion_matrix.png")
        if spec["name"] == selected_spec["name"]:
            selected_metrics = metrics
            save_confusion_matrix(y_ref, pred, f"{spec['name']} confusion matrix", "ensemble_selected_confusion_matrix.png")
            save_json(
                DIRS["metrics"] / "selected_ensemble_classification_report.json",
                classification_report(y_ref, pred, target_names=CONFIG["class_folders"], output_dict=True, zero_division=0),
            )

    table3 = pd.DataFrame(results_rows)[["Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-score (%)", "AUC", "Validation selection score"]]
    save_df(DIRS["tables"] / "table3_run.csv", table3)

    vit_swin_probs = 0.5 * probs["vit"] + 0.5 * probs["swin"]
    selected_prob = ensemble_outputs[selected_spec["name"]]["prob"]
    roc_probs = selected_prob if selected_prob is not None else ensemble_outputs["Ensemble (Weighted Avg)"]["prob"]
    save_roc_comparison(y_ref, roc_probs, vit_swin_probs)
    save_epoch_ensemble_accuracy_curves(weighted_curve_weights)

    final_pred_df = pd.DataFrame({
        "filename": test_df["filename"],
        "y_true": y_ref,
        "vit_pred": preds["vit"],
        "swin_pred": preds["swin"],
        "convnext_pred": preds["convnext"],
        "ensemble_weighted_pred": ensemble_outputs["Ensemble (Weighted Avg)"]["pred"],
        "ensemble_majority_pred": ensemble_outputs["Ensemble (Majority Voting)"]["pred"],
        "ensemble_selected_pred": ensemble_outputs[selected_spec["name"]]["pred"],
    })
    if "Ensemble (Weighted Avg - Equal)" in ensemble_outputs:
        final_pred_df["ensemble_equal_weighted_pred"] = ensemble_outputs["Ensemble (Weighted Avg - Equal)"]["pred"]
    if "Ensemble (Stacking LogReg)" in ensemble_outputs:
        final_pred_df["ensemble_stacking_pred"] = ensemble_outputs["Ensemble (Stacking LogReg)"]["pred"]
    save_df(DIRS["predictions"] / "ensemble_test_predictions.csv", final_pred_df)

    ensure(selected_metrics is not None, "No selected ensemble metrics were produced.")
    return table3, selected_metrics


# -----------------------------------------------------------------------------
# paper reference tables
# -----------------------------------------------------------------------------
def save_paper_reference_tables():
    # Table 2 from paper
    table2_rows = [
        {"Types of Images": "Adenocarcinoma", "Deep Learning Model": "U-Net", "Dice Coefficient": 0.887, "Jaccard Index": 0.808, "Conformity Coefficient": 0.646, "BF-score": 0.791, "HOG-sim": 0.910},
        {"Types of Images": "Adenocarcinoma", "Deep Learning Model": "MedT", "Dice Coefficient": 0.735, "Jaccard Index": 0.595, "Conformity Coefficient": 0.197, "BF-score": 0.554, "HOG-sim": 0.780},
        {"Types of Images": "Adenocarcinoma", "Deep Learning Model": "SegNet", "Dice Coefficient": 0.865, "Jaccard Index": 0.775, "Conformity Coefficient": 0.646, "BF-score": 0.760, "HOG-sim": 0.895},
        {"Types of Images": "Adenocarcinoma", "Deep Learning Model": "Proposed Methodology", "Dice Coefficient": 0.900, "Jaccard Index": 0.820, "Conformity Coefficient": 0.700, "BF-score": 0.804, "HOG-sim": 0.935},

        {"Types of Images": "High-grade IN", "Deep Learning Model": "U-Net", "Dice Coefficient": 0.895, "Jaccard Index": 0.816, "Conformity Coefficient": 0.747, "BF-score": 0.803, "HOG-sim": 0.918},
        {"Types of Images": "High-grade IN", "Deep Learning Model": "MedT", "Dice Coefficient": 0.824, "Jaccard Index": 0.707, "Conformity Coefficient": 0.556, "BF-score": 0.682, "HOG-sim": 0.840},
        {"Types of Images": "High-grade IN", "Deep Learning Model": "SegNet", "Dice Coefficient": 0.894, "Jaccard Index": 0.812, "Conformity Coefficient": 0.757, "BF-score": 0.800, "HOG-sim": 0.915},
        {"Types of Images": "High-grade IN", "Deep Learning Model": "Proposed Methodology", "Dice Coefficient": 0.910, "Jaccard Index": 0.840, "Conformity Coefficient": 0.780, "BF-score": 0.820, "HOG-sim": 0.940},

        {"Types of Images": "Low-grade IN", "Deep Learning Model": "U-Net", "Dice Coefficient": 0.911, "Jaccard Index": 0.849, "Conformity Coefficient": 0.765, "BF-score": 0.835, "HOG-sim": 0.930},
        {"Types of Images": "Low-grade IN", "Deep Learning Model": "MedT", "Dice Coefficient": 0.889, "Jaccard Index": 0.808, "Conformity Coefficient": 0.730, "BF-score": 0.810, "HOG-sim": 0.905},
        {"Types of Images": "Low-grade IN", "Deep Learning Model": "SegNet", "Dice Coefficient": 0.924, "Jaccard Index": 0.864, "Conformity Coefficient": 0.828, "BF-score": 0.850, "HOG-sim": 0.945},
        {"Types of Images": "Low-grade IN", "Deep Learning Model": "Proposed Methodology", "Dice Coefficient": 0.920, "Jaccard Index": 0.855, "Conformity Coefficient": 0.810, "BF-score": 0.842, "HOG-sim": 0.940},

        {"Types of Images": "Normal", "Deep Learning Model": "U-Net", "Dice Coefficient": 0.411, "Jaccard Index": 0.263, "Conformity Coefficient": -2.199, "BF-score": 0.320, "HOG-sim": 0.450},
        {"Types of Images": "Normal", "Deep Learning Model": "MedT", "Dice Coefficient": 0.676, "Jaccard Index": 0.562, "Conformity Coefficient": -0.615, "BF-score": 0.510, "HOG-sim": 0.720},
        {"Types of Images": "Normal", "Deep Learning Model": "SegNet", "Dice Coefficient": 0.777, "Jaccard Index": 0.684, "Conformity Coefficient": -0.607, "BF-score": 0.610, "HOG-sim": 0.780},
        {"Types of Images": "Normal", "Deep Learning Model": "Proposed Methodology", "Dice Coefficient": 0.760, "Jaccard Index": 0.690, "Conformity Coefficient": -0.650, "BF-score": 0.620, "HOG-sim": 0.770},

        {"Types of Images": "PolyP", "Deep Learning Model": "U-Net", "Dice Coefficient": 0.965, "Jaccard Index": 0.908, "Conformity Coefficient": -1.514, "BF-score": 0.930, "HOG-sim": 0.975},
        {"Types of Images": "PolyP", "Deep Learning Model": "MedT", "Dice Coefficient": 0.771, "Jaccard Index": 0.643, "Conformity Coefficient": 0.305, "BF-score": 0.610, "HOG-sim": 0.760},
        {"Types of Images": "PolyP", "Deep Learning Model": "SegNet", "Dice Coefficient": 0.937, "Jaccard Index": 0.886, "Conformity Coefficient": 0.858, "BF-score": 0.900, "HOG-sim": 0.955},
        {"Types of Images": "PolyP", "Deep Learning Model": "Proposed Methodology", "Dice Coefficient": 0.960, "Jaccard Index": 0.920, "Conformity Coefficient": 0.880, "BF-score": 0.925, "HOG-sim": 0.970},

        {"Types of Images": "Serrated adenoma", "Deep Learning Model": "U-Net", "Dice Coefficient": 0.938, "Jaccard Index": 0.888, "Conformity Coefficient": 0.865, "BF-score": 0.900, "HOG-sim": 0.960},
        {"Types of Images": "Serrated adenoma", "Deep Learning Model": "MedT", "Dice Coefficient": 0.670, "Jaccard Index": 0.509, "Conformity Coefficient": -0.043, "BF-score": 0.500, "HOG-sim": 0.730},
        {"Types of Images": "Serrated adenoma", "Deep Learning Model": "SegNet", "Dice Coefficient": 0.907, "Jaccard Index": 0.832, "Conformity Coefficient": 0.794, "BF-score": 0.870, "HOG-sim": 0.940},
        {"Types of Images": "Serrated adenoma", "Deep Learning Model": "Proposed Methodology", "Dice Coefficient": 0.920, "Jaccard Index": 0.870, "Conformity Coefficient": 0.840, "BF-score": 0.890, "HOG-sim": 0.955},
    ]
    save_csv(DIRS["paper_reference"] / "table2_paper_reference.csv", table2_rows)

    table3_rows = [
        {"Model": "ViT", "Accuracy (%)": 87.6, "Precision (%)": 88.8, "Recall (%)": 85.5, "F1-score (%)": 86.2, "AUC": None},
        {"Model": "Swin Transformer", "Accuracy (%)": 89.4, "Precision (%)": 90.2, "Recall (%)": 87.3, "F1-score (%)": 88.4, "AUC": None},
        {"Model": "ConvNeXt", "Accuracy (%)": 88.5, "Precision (%)": 89.2, "Recall (%)": 86.4, "F1-score (%)": 87.6, "AUC": None},
        {"Model": "Ensemble (Weighted Avg)", "Accuracy (%)": 95.1, "Precision (%)": 93.5, "Recall (%)": 91.3, "F1-score (%)": 94.2, "AUC": 0.95},
        {"Model": "Ensemble (Majority Voting)", "Accuracy (%)": 93.2, "Precision (%)": 92.3, "Recall (%)": 90.1, "F1-score (%)": 91.2, "AUC": None},
    ]
    save_csv(DIRS["paper_reference"] / "table3_paper_reference.csv", table3_rows)

    table4_rows = [
        {"Ref": "[24]", "Year": 2023, "Method/Backbone": "ViT + Swin (dual branch)", "Dataset & Split": "EBHI-Seg 70/10/20", "AUC": 0.94, "F1-score (%)": 93.9, "Accuracy (%)": 94.2},
        {"Ref": "[25]", "Year": 2023, "Method/Backbone": "CNN feature extractor + SVM", "Dataset & Split": "EBHI 5-fold CV", "AUC": None, "F1-score (%)": 94.5, "Accuracy (%)": 95.37},
        {"Ref": "[26]", "Year": 2024, "Method/Backbone": "EfficientNet-B3 + CBAM", "Dataset & Split": "EBHI-Seg 80/10/10", "AUC": 0.943, "F1-score (%)": 94.1, "Accuracy (%)": 94.8},
        {"Ref": "[14]", "Year": 2024, "Method/Backbone": "CCDNet", "Dataset & Split": "NCT-CRC-HE-100K 80/10/10", "AUC": None, "F1-score (%)": 98.6, "Accuracy (%)": 98.6},
        {"Ref": "[35]", "Year": 2024, "Method/Backbone": "MobileViT-UNet", "Dataset & Split": "GlaS-Extended", "AUC": None, "F1-score (%)": 94.4, "Accuracy (%)": None},
        {"Ref": "[36]", "Year": 2023, "Method/Backbone": "SwinCup", "Dataset & Split": "GlaS & CRAG", "AUC": None, "F1-score (%)": 90.0, "Accuracy (%)": None},
        {"Ref": "Proposed", "Year": 2025, "Method/Backbone": "GR-HDUNET -> ViT + Swin + ConvNeXt ensemble", "Dataset & Split": "EBHI-Seg 70/10/20", "AUC": 0.95, "F1-score (%)": 94.2, "Accuracy (%)": 95.1},
    ]
    save_csv(DIRS["paper_reference"] / "table4_paper_reference.csv", table4_rows)


# -----------------------------------------------------------------------------
# final manifest
# -----------------------------------------------------------------------------
def save_run_manifest(seg_tables: List[pd.DataFrame], table3: pd.DataFrame, proposed_metrics: Dict[str, float]):
    summary = {
        "completed_at": now_str(),
        "saved_artifacts": {
            "plots_dir": str(DIRS["plots"]),
            "metrics_dir": str(DIRS["metrics"]),
            "tables_dir": str(DIRS["tables"]),
            "checkpoints_dir": str(DIRS["checkpoints"]),
            "final_models_dir": str(DIRS["models_final"]),
            "masks_train_dir": str(DIRS["masks_train"]),
            "masks_val_dir": str(DIRS["masks_val"]),
            "masks_test_dir": str(DIRS["masks_test"]),
        },
        "proposed_final_classification": proposed_metrics,
    }
    save_json(DIRS["meta"] / "manifest.json", summary)
    all_seg = pd.concat(seg_tables, axis=0)
    save_df(DIRS["tables"] / "table2_run_all_models.csv", all_seg)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    device = setup_environment()
    save_json(DIRS["meta"] / "arguments.json", {"proposed": args.proposed, "all": args.all})
    save_json(DIRS["meta"] / "assumptions.json", ASSUMPTIONS)
    save_paper_reference_tables()

    df = build_index()
    warmup_decode_cache(df)
    save_preprocessing_figure(df)
    splits = make_splits(df)

    # Proposed segmentation
    proposed_generator, proposed_seg_inference_cfg = train_proposed_segmentation(splits["train"], splits["val"], device)
    evaluate_tables = []
    proposed_seg_table = evaluate_segmentation_model(
        "Proposed Methodology",
        proposed_generator,
        splits["test"],
        device,
        inference_cfg=proposed_seg_inference_cfg,
    )
    evaluate_tables.append(proposed_seg_table)

    # Generate masks for all splits for classification
    generate_refined_masks("train", splits["train"], proposed_generator, device, inference_cfg=proposed_seg_inference_cfg)
    generate_refined_masks("val", splits["val"], proposed_generator, device, inference_cfg=proposed_seg_inference_cfg)
    generate_refined_masks("test", splits["test"], proposed_generator, device, inference_cfg=proposed_seg_inference_cfg)
    mask_dirs = {"train": DIRS["masks_train"], "val": DIRS["masks_val"], "test": DIRS["masks_test"]}

    # Baselines if requested
    if args.all:
        unet, unet_inference_cfg = train_baseline_segmentation("U-Net", UNetBaseline(), splits["train"], splits["val"], device)
        segnet, segnet_inference_cfg = train_baseline_segmentation("SegNet", SegNetBaseline(), splits["train"], splits["val"], device)
        medt, medt_inference_cfg = train_baseline_segmentation("MedT", MedTBaseline(), splits["train"], splits["val"], device)

        evaluate_tables.append(evaluate_segmentation_model("U-Net", unet, splits["test"], device, inference_cfg=unet_inference_cfg))
        evaluate_tables.append(evaluate_segmentation_model("SegNet", segnet, splits["test"], device, inference_cfg=segnet_inference_cfg))
        evaluate_tables.append(evaluate_segmentation_model("MedT", medt, splits["test"], device, inference_cfg=medt_inference_cfg))

    # Classification
    vit = train_classifier("vit", build_vit, splits["train"], splits["val"], mask_dirs, device)
    swin = train_classifier("swin", build_swin, splits["train"], splits["val"], mask_dirs, device)
    convnext = train_classifier("convnext", build_convnext, splits["train"], splits["val"], mask_dirs, device)

    models_dict = {"vit": vit, "swin": swin, "convnext": convnext}
    table3, proposed_metrics = evaluate_classifiers(models_dict, splits["test"], mask_dirs, device)

    # Comparison table using current run + paper reference rows
    table4_ref = pd.read_csv(DIRS["paper_reference"] / "table4_paper_reference.csv")
    run_row = {
        "Ref": "This run",
        "Year": datetime.now().year,
        "Method/Backbone": "GR-HDUNET -> ViT + Swin + ConvNeXt ensemble",
        "Dataset & Split": "EBHI-Seg 70/10/20",
        "AUC": proposed_metrics["AUC"],
        "F1-score (%)": proposed_metrics["F1-score (%)"],
        "Accuracy (%)": proposed_metrics["Accuracy (%)"],
    }
    table4_run = pd.concat([table4_ref, pd.DataFrame([run_row])], ignore_index=True)
    save_df(DIRS["tables"] / "table4_paper_reference_plus_this_run.csv", table4_run)

    save_run_manifest(evaluate_tables, table3, proposed_metrics)

    log_header("DONE")
    print(f"Everything saved under: {RUN_ROOT}")
    print("Key outputs:")
    print(f"- Log: {DIRS['logs'] / CONFIG['log_filename']}")
    print(f"- Table 2 run: {DIRS['tables'] / 'table2_run_all_models.csv'}")
    print(f"- Table 3 run: {DIRS['tables'] / 'table3_run.csv'}")
    print(f"- Figure 3: {DIRS['plots'] / 'figure3_preprocessing.png'}")
    print(f"- Figure 5: {DIRS['plots'] / 'figure5_accuracy_vs_epoch.png'}")
    print(f"- Figure 6: {DIRS['plots'] / 'figure6_roc_comparison.png'}")


if __name__ == "__main__":
    main()
