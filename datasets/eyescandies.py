import os
import glob
import random
import yaml
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.mvtec3d_utils import organized_pc_to_depth_map, resize_organized_pc

eyescandies_classes = [
    'CandyCane', 'ChocolateCookie', 'ChocolatePraline', 'Confetto',
    'GummyBear', 'HazelnutTruffle', 'LicoriceSandwich', 'Lollipop',
    'Marshmallow', 'PeppermintCandy'
]

EYESCANDIES_DIR = "/home/pub_766/eyescandies/Eyecandies"


def load_eyescandies(category, k_shot, missing_type, missing_rate=0.3):
    """
    Load Eyescandies dataset with the same interface as load_mvtec3d.

    Returns:
        (train_img_paths, train_depth_paths, train_gt_paths,
         train_labels, train_types, train_missing_idxs),
        (test_img_paths, test_depth_paths, test_gt_paths,
         test_labels, test_types, test_missing_idxs)

    Notes:
        - img_paths  : path to {idx}_image_0.png  (RGB, all spotlights on)
        - depth_paths: path to {idx}_depth.png    (16-bit normalized depth)
        - gt_paths   : 0 for good samples, path to {idx}_mask.png for anomalous
        - labels     : 0=good, 1=anomalous
        - types      : 'good' or 'anomaly'
    """
    assert category in eyescandies_classes, \
        f"Unknown category: {category}. Must be one of {eyescandies_classes}"

    cat_dir = os.path.join(EYESCANDIES_DIR, category)

    # ── Training split (all good) ──────────────────────────────────────────
    train_img_paths, train_depth_paths, train_gt_paths, \
        train_labels, train_types = _load_good_split(
            os.path.join(cat_dir, 'train', 'data')
        )

    # ── Test split (good + anomalous) ──────────────────────────────────────
    test_img_paths, test_depth_paths, test_gt_paths, \
        test_labels, test_types = _load_test_split(
            os.path.join(cat_dir, 'test_public', 'data')
        )

    # ── Missing modality settings ──────────────────────────────────────────
    train_missing = missing_setting(len(train_img_paths), missing_type, missing_rate)
    test_missing  = missing_setting(len(test_img_paths),  missing_type, missing_rate)

    return (
        train_img_paths, train_depth_paths, train_gt_paths,
        train_labels, train_types, train_missing
    ), (
        test_img_paths, test_depth_paths, test_gt_paths,
        test_labels, test_types, test_missing
    )


def _load_good_split(data_dir):
    """Load a split where all samples are good (train / val)."""
    img_paths   = []
    depth_paths = []
    gt_paths    = []
    labels      = []
    types       = []

    # Collect unique sample indices by scanning for image_0 files
    image_files = sorted(glob.glob(os.path.join(data_dir, '*_image_0.png')))

    for img_path in image_files:
        basename = os.path.basename(img_path)          # e.g. 000_image_0.png
        idx = basename.replace('_image_0.png', '')      # e.g. 000

        depth_path = os.path.join(data_dir, f'{idx}_depth.png')
        if not os.path.exists(depth_path):
            continue  # skip incomplete samples

        img_paths.append(img_path)
        depth_paths.append(depth_path)
        gt_paths.append(0)          # good → no mask
        labels.append(0)
        types.append('good')

    return img_paths, depth_paths, gt_paths, labels, types


def _load_test_split(data_dir):
    """Load test split, using metadata.yaml to determine good/anomalous."""
    img_paths   = []
    depth_paths = []
    gt_paths    = []
    labels      = []
    types       = []

    image_files = sorted(glob.glob(os.path.join(data_dir, '*_image_0.png')))

    for img_path in image_files:
        basename = os.path.basename(img_path)
        idx = basename.replace('_image_0.png', '')

        depth_path    = os.path.join(data_dir, f'{idx}_depth.png')
        mask_path     = os.path.join(data_dir, f'{idx}_mask.png')
        metadata_path = os.path.join(data_dir, f'{idx}_metadata.yaml')

        if not os.path.exists(depth_path):
            continue

        # Determine label from metadata
        is_anomalous = False
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path) as f:
                    meta = yaml.safe_load(f)
                # Eyescandies uses 'anomalous: 0/1'
                is_anomalous = bool(meta.get('anomalous', 0))
            except Exception:
                is_anomalous = os.path.exists(mask_path)
        else:
            is_anomalous = os.path.exists(mask_path)

        img_paths.append(img_path)
        depth_paths.append(depth_path)

        if is_anomalous and os.path.exists(mask_path):
            gt_paths.append(mask_path)
            labels.append(1)
            types.append('anomaly')
        else:
            gt_paths.append(0)
            labels.append(0)
            types.append('good')

    return img_paths, depth_paths, gt_paths, labels, types


def missing_setting(length, missing_type, missing_rate):
    """Identical to MVTec3D missing_setting — reused for consistency."""
    tot_missing_indx = [0] * length
    missing_indices = random.sample(range(length), int(length * missing_rate))

    if missing_type == 'img':
        for idx in missing_indices:
            tot_missing_indx[idx] = 1
    elif missing_type == 'depth':
        for idx in missing_indices:
            tot_missing_indx[idx] = 2
    elif missing_type == 'both':
        for idx in missing_indices:
            tot_missing_indx[idx] = 1
        missing_indices = random.sample(missing_indices, len(missing_indices) // 2)
        for idx in missing_indices:
            tot_missing_indx[idx] = 2

    return tot_missing_indx


# ── EyecandiesDataset ──────────────────────────────────────────────────────────
# Subclass of CLIPDataset that reads PNG depth maps instead of TIFF point clouds.
# Converts depth PNG → organized point cloud (H×W×3) to match MVTec3D format.

FOCAL_LENGTH = 711.11  # Fixed camera focal length for all Eyescandies images


def depth_png_to_organized_pc(depth_path, info_depth_path):
    """
    Convert Eyescandies 16-bit depth PNG to organized point cloud (H×W×3).
    Matches the format returned by read_tiff_organized_pc for MVTec3D.
    """
    # Step 1: Read 16-bit depth image
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    # Step 2: Denormalize using info_depth.yaml
    if os.path.exists(info_depth_path):
        with open(info_depth_path) as f:
            info = yaml.safe_load(f)
        mind = info['normalization']['min']
        maxd = info['normalization']['max']
        depth_m = depth_raw / 65535.0 * (maxd - mind) + mind
    else:
        # Fallback: simple 0-1 normalization
        depth_m = depth_raw / 65535.0

    height, width = depth_m.shape

    # Step 3: Build organized point cloud using pinhole camera model
    cx, cy = width / 2.0, height / 2.0
    u = np.arange(width, dtype=np.float32)
    v = np.arange(height, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    # Back-project: X = (u - cx) * Z / f,  Y = (v - cy) * Z / f,  Z = depth
    Z = depth_m
    X = (uu - cx) * Z / FOCAL_LENGTH
    Y = (vv - cy) * Z / FOCAL_LENGTH

    # Stack into H×W×3 organized point cloud (same as MVTec3D TIFF format)
    organized_pc = np.stack([X, Y, Z], axis=2).astype(np.float32)
    return organized_pc


class EyecandiesDataset(Dataset):
    """
    Dataset class for Eyescandies that mirrors CLIPDataset's interface exactly.
    Reads PNG depth maps and converts them to organized point clouds on the fly.
    """

    def __init__(self, category, phase, k_shot, missing_type, missing_rate):
        self.phase = phase
        self.category = category

        train_data, test_data = load_eyescandies(
            category, k_shot, missing_type, missing_rate
        )

        if phase == 'train':
            (self.img_paths, self.pc_paths, self.gt_paths,
             self.labels, self.types, self.missing_indxs) = train_data
        else:
            (self.img_paths, self.pc_paths, self.gt_paths,
             self.labels, self.types, self.missing_indxs) = test_data

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path  = self.img_paths[idx]
        pc_path   = self.pc_paths[idx]   # depth PNG path
        gt        = self.gt_paths[idx]
        label     = self.labels[idx]
        img_type  = self.types[idx]

        # ── RGB image ──────────────────────────────────────────────────────
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # ── Depth PNG → organized point cloud ─────────────────────────────
        idx_str       = os.path.basename(pc_path).replace('_depth.png', '')
        data_dir      = os.path.dirname(pc_path)
        info_depth_path = os.path.join(data_dir, f'{idx_str}_info_depth.yaml')

        organized_pc = depth_png_to_organized_pc(pc_path, info_depth_path)

        # ── Depth map (Z channel repeated 3×) ─────────────────────────────
        depth_map_3channel = np.repeat(
            organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2
        )

        # ── Resize to 240×240 ─────────────────────────────────────────────
        resized_depth_map_3channel = resize_organized_pc(
            depth_map_3channel, target_height=240, target_width=240
        )
        resized_organized_pc = resize_organized_pc(
            organized_pc, target_height=240, target_width=240
        )
        resized_organized_pc = resized_organized_pc.clone().detach().float()

        # ── Ground truth mask ─────────────────────────────────────────────
        if gt == 0:
            gt = np.zeros([img.shape[0], img.shape[0]])
        else:
            gt = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
            gt[gt > 0] = 255

        # ── Resize img and gt ─────────────────────────────────────────────
        img = cv2.resize(img, (240, 240))
        gt  = cv2.resize(gt, (240, 240), interpolation=cv2.INTER_NEAREST)

        img_name = (
            f'{self.category}-{img_type}-'
            f'{os.path.basename(img_path[:-4])}'
        )

        # ── Apply missing modality ─────────────────────────────────────────
        if self.missing_indxs[idx] == 1:
            img = np.zeros_like(img)
        elif self.missing_indxs[idx] == 2:
            resized_depth_map_3channel = np.zeros_like(resized_depth_map_3channel)

        return (
            img,
            torch.Tensor(resized_organized_pc),
            torch.Tensor(resized_depth_map_3channel),
            gt, label, img_name, img_type,
            self.missing_indxs[idx]
        )