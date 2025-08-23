import cv2
import numpy as np
from typing import Tuple


def compute_flow_farneback(prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
    # Dense displacement (px/frame)
    return cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0
    )


def block_weighted_average(vel: np.ndarray, mask: np.ndarray, cell: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-cell weighted average velocity (weights = count of valid pixels)."""
    H, W = vel.shape[:2]
    HH, WW = (H//cell)*cell, (W//cell)*cell
    vel = vel[:HH, :WW, :]
    mask = mask[:HH, :WW].astype(np.float32)

    hh, ww = HH//cell, WW//cell
    vel_r = vel.reshape(hh, cell, ww, cell, 2)
    w_r = mask.reshape(hh, cell, ww, cell)

    vw_sum = (vel_r * w_r[..., None]).sum(axis=(1, 3))  # (hh, ww, 2)
    w_sum  = w_r.sum(axis=(1, 3))                       # (hh, ww)
    return vw_sum, w_sum