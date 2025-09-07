import cv2
import numpy as np
from typing import Tuple


def compute_flow_farneback(prev_gray, gray, k: int = 1):
    levels   = 3 + int(np.ceil(np.log2(k)))      # hasta 5–6
    winsize  = min(41, 15 + 8*(k-1))             # 15→23→31…
    iters    = 3 + (1 if k>=2 else 0)
    poly_n   = 5 if k==1 else 7
    poly_sig = 1.2 if k==1 else 1.5
    return cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=0.5, levels=levels, winsize=winsize,
        iterations=iters, poly_n=poly_n, poly_sigma=poly_sig, flags=0
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