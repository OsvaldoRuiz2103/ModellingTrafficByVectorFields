# detectors_yolov4.py
from pathlib import Path
import re
import cv2
import numpy as np

# COCO class ids
_COCO = {"person":0, "bicycle":1, "car":2, "motorcycle":3, "airplane":4, "bus":5,
         "train":6, "truck":7, "boat":8, "traffic light":9}

def _resolve(p: str) -> str:
    p = Path(p).expanduser()
    return str(p if p.exists() else (Path.cwd() / p.name))

def _read_cfg_size(cfg_path: str) -> tuple[int, int]:
    """Parse Darknet cfg for width/height; default to 416 if not found."""
    w = h = None
    try:
        with open(cfg_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = [x.strip() for x in line.split("=", 1)]
                if k.lower() == "width":
                    w = int(re.findall(r"\d+", v)[0])
                elif k.lower() == "height":
                    h = int(re.findall(r"\d+", v)[0])
        if not w or not h:
            raise ValueError
    except Exception:
        w = h = 416
    return w, h

class YOLOCarDetector:
    """
    YOLOv4 / YOLOv4-tiny (Darknet cfg+weights via OpenCV DNN).
    API: .car_mask(frame_bgr, conf=0.5) -> boolean mask (H,W)
    """
    def __init__(self,
                 cfg: str,
                 weights: str,
                 use_gpu: bool = False,
                 include_bus_truck: bool = False,
                 nms_thresh: float = 0.45):
        cfg = _resolve(cfg)
        weights = _resolve(weights)

        self.net = cv2.dnn.readNetFromDarknet(cfg, weights)
        self.out_names = self.net.getUnconnectedOutLayersNames()
        self.inp_w, self.inp_h = _read_cfg_size(cfg)

        if use_gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.nms_thresh = float(nms_thresh)
        self.allowed_ids = {_COCO["car"]}
        if include_bus_truck:
            self.allowed_ids |= {_COCO["bus"], _COCO["truck"]}

    def car_mask(self, frame_bgr: np.ndarray, conf: float = 0.5, expand_px: int = 6) -> np.ndarray:
        """
        Returns a boolean mask where True corresponds to detected car pixels.
        conf: threshold on (objectness * class_prob) after parsing YOLO outputs.
        """
        H, W = frame_bgr.shape[:2]

        # 1) Preprocess
        blob = cv2.dnn.blobFromImage(
            frame_bgr, scalefactor=1/255.0, size=(self.inp_w, self.inp_h),
            mean=(0, 0, 0), swapRB=True, crop=False
        )
        self.net.setInput(blob)

        # 2) Forward
        outs = self.net.forward(self.out_names)

        # 3) Parse detections
        boxes, scores, class_ids = [], [], []
        x_scale, y_scale = W / self.inp_w, H / self.inp_h

        for det in outs:              # det shape: (N, 85) typically
            for row in det:
                obj = float(row[4])
                if obj < 1e-6:
                    continue
                class_scores = row[5:]
                cid = int(np.argmax(class_scores))
                score = obj * float(class_scores[cid])
                if score < conf:
                    continue
                cx, cy, bw, bh = row[0:4]
                # Convert center-format (network scale) to top-left (image px)
                px = int((cx - bw/2) * x_scale)
                py = int((cy - bh/2) * y_scale)
                pw = int(bw * x_scale)
                ph = int(bh * y_scale)
                boxes.append([px, py, pw, ph])
                scores.append(score)
                class_ids.append(cid)

        # 4) NMS and masking
        mask = np.zeros((H, W), np.uint8)
        if boxes:
            idxs = cv2.dnn.NMSBoxes(boxes, scores, conf, self.nms_thresh)
            if len(idxs) > 0:
                if isinstance(idxs, np.ndarray):
                    idxs = idxs.flatten().tolist()
                for i in idxs:
                    if class_ids[i] in self.allowed_ids:
                        x, y, w, h = boxes[i]
                        x1, y1 = max(0, x), max(0, y)
                        x2, y2 = min(W - 1, x + w), min(H - 1, y + h)
                        mask[y1:y2+1, x1:x2+1] = 255

        if expand_px > 0 and mask.any():
            k = np.ones((expand_px, expand_px), np.uint8)
            mask = cv2.dilate(mask, k, 1)

        return mask.astype(bool)
