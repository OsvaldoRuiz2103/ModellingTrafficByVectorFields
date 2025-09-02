from typing import List, Optional

import cv2
import numpy as np
import time

from video_processing import ensure_fps, resize_to_width
from trajectory_analysis import compute_flow_farneback, block_weighted_average
from detectors import YOLOVehicleMaskDetector

def process_videos(videos: List[str],
                   grid: int = 16,
                   min_mag_px_per_frame: float = 0.5,
                   frame_step: int = 1,
                   resize_width: Optional[int] = 960,
                   fps_fallback: float = 30.0):
    """
    Aggregate optical flow from multiple videos into one vector field.
    Shows progress and timing.
    Returns: u, v, weights, mean_speed_px_s, Hc, Wc
    """
    if not videos:
        raise ValueError("No videos provided.")
    
    det = YOLOVehicleMaskDetector(
        model_path=r"models/visdrone/best.onnx",
        classes=("car","van","truck","bus","motor"),
        imgsz=1536, max_det=1000, 
        use_seg="auto",                          # try instance masks, else bbox
        dilation_px=3,                           # optional: expand a bit
        min_area_frac=0.00005                    # optional: drop tiny specks
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    sum_v_cells = None
    sum_w_cells = None
    Hc = Wc = None

    for idx, path in enumerate(videos, 1):
        print(f"\n[INFO] Processing video {idx}/{len(videos)}: {path}")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"[WARN] Could not open {path}, skipping.")
            continue

        fps = ensure_fps(cap, fps_fallback)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0
        start_video = time.time()

        ok, frame = cap.read()
        if not ok:
            print(f"[WARN] Empty video {path}, skipping.")
            cap.release(); continue

        frame = resize_to_width(frame, resize_width)
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if sum_v_cells is None:
            H, W = prev_gray.shape
            HH, WW = (H//grid)*grid, (W//grid)*grid
            Hc, Wc = HH//grid, WW//grid
            sum_v_cells = np.zeros((Hc, Wc, 2), dtype=np.float64)
            sum_w_cells = np.zeros((Hc, Wc), dtype=np.float64)

        while True:
            # Skip frames if needed
            for _ in range(frame_step):
                ok, frame = cap.read()
                if not ok: break
            if not ok: break

            cars_only = det.car_mask(frame, conf=0.7, iou=0.7)

            frame = resize_to_width(frame, resize_width)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            flow = np.zeros((H, W, 2), np.float32)
       
            mask_r = cv2.resize(cars_only.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            cars_only = mask_r.astype(bool)   
            cars_u8   = mask_r * 255          

            cars_u8 = cv2.morphologyEx(cars_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
            _, _, stats, _ = cv2.connectedComponentsWithStats(cars_u8, connectivity=8)

            # stats: [x, y, w, h, area], stats[0] is background
            area_min_px = 64
            boxes = [(int(x), int(y), int(w), int(h))
                    for (x, y, w, h, area) in stats[1:] if area > area_min_px]
            
            if not boxes:
                prev_gray = gray
                processed_frames += 1
                continue

            total_roi_area = sum(w*h for (x, y, w, h) in boxes)
            use_roi = (total_roi_area < 0.4 * H * W) and (len(boxes) < 60)

            if use_roi:
                pad = 16  # ~winsize
                for (x,y,w,h) in boxes:
                    x1 = max(0, x - pad); y1 = max(0, y - pad)
                    x2 = min(W, x + w + pad); y2 = min(H, y + h + pad)
                    roi1 = prev_gray[y1:y2, x1:x2]
                    roi2 = gray[y1:y2, x1:x2]
                    flow_roi = compute_flow_farneback(roi1, roi2)
                    flow[y1:y2, x1:x2] = flow_roi
            else:
                flow = compute_flow_farneback(prev_gray, gray)

            vel  = flow * fps

            mag_pf = np.sqrt((flow**2).sum(axis=2))
            motion_mask = mag_pf >= float(min_mag_px_per_frame)

            mask = motion_mask & cars_only

            vw_sum, w_sum = block_weighted_average(vel, mask, grid)
            sum_v_cells += vw_sum
            sum_w_cells += w_sum


            prev_gray = gray
            processed_frames += 1

            # Show progress each 50 frames
            if processed_frames % 50 == 0 or processed_frames == total_frames:
              #  cv2.imwrite(f"runs/detect/frame_{processed_frames:06d}.png", (cars_only.astype(np.uint8) * 255))
                elapsed = time.time() - start_video
                est_total = elapsed / processed_frames * total_frames
                print(f"  Frame {processed_frames}/{total_frames} "
                      f"({processed_frames/total_frames:.1%}) "
                      f"Elapsed: {elapsed:.1f}s, Est. total: {est_total:.1f}s")

        cap.release()
        print(f"[INFO] Finished video {idx} in {time.time() - start_video:.1f}s, processed {processed_frames} frames")

    if sum_w_cells is None:
        raise RuntimeError("No valid motion found.")

    field = sum_v_cells / np.maximum(sum_w_cells[..., None], 1e-6)
    return field[...,0], field[...,1], sum_w_cells, Hc, Wc


