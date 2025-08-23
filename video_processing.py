from pathlib import Path
from typing import List, Optional
import cv2
import os, glob
import numpy as np


def gather_videos(input_dir: Optional[str] = None,
                  patterns: Optional[List[str]] = None,
                  ext: str = ".avi",
                  recursive: bool = True) -> List[str]:
    """
    Collect videos by folder and/or glob patterns.
    - input_dir: folder to sweep (optional)
    - patterns: extra globs like ['/data/*.avi', '/more/*cam1*.avi'] (optional)
    - ext: extension to filter in input_dir (default .avi)
    - recursive: search subfolders when using input_dir
    """
    out = []
    if input_dir:
        ext = ext if ext.startswith(".") else "." + ext
        root = Path(input_dir)
        it = root.rglob(f"*{ext}") if recursive else root.glob(f"*{ext}")
        out.extend(str(p) for p in it if p.is_file() and p.suffix.lower() == ext.lower())

    if patterns:
        for p in patterns:
            if any(ch in p for ch in "*?[]"):
                out.extend(glob.glob(p, recursive=True))
            elif os.path.isfile(p):
                out.append(p)

    # Keep only files that exist
    out = [p for p in out if os.path.isfile(p)]
    # De-dup, stable order
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq


def ensure_fps(cap: cv2.VideoCapture, fallback: float = 30.0) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    return float(fps) if fps and fps > 1e-6 else float(fallback)


def resize_to_width(img: np.ndarray, target_w: Optional[int]) -> np.ndarray:
    if target_w is None: return img
    h, w = img.shape[:2]
    if w == target_w: return img
    scale = target_w / float(w)
    return cv2.resize(img, (target_w, max(1, int(round(h*scale)))), interpolation=cv2.INTER_AREA)
