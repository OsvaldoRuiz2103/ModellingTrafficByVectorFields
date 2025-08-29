from typing import List, Optional, Sequence
import cv2
import numpy as np
from ultralytics import YOLO

class YOLOVehicleMaskDetector:
    def __init__(
        self,
        model_path: str,
        classes: Sequence[str] | Sequence[int] = ("car", "van", "truck", "bus", "motor"),
        imgsz: int = 1536,
        max_det: int = 1000,
        device: Optional[str] = None,     # 'cpu' or '0' etc.
        use_seg: str = "auto",            # 'auto' | 'seg' | 'box'
        keep_mask_path: Optional[str] = None,  # binary PNG (white=keep, black=ignore)
        dilation_px: int = 0,             # e.g., 3-7 to slightly expand masks
        min_area_frac: float = 0.0,       # drop tiny dets in mask (fraction of frame area)
    ):
        self.model = YOLO(model_path)
        self.imgsz = int(imgsz)
        self.max_det = int(max_det)
        self.device = device
        self.use_seg = use_seg
        self.dilation_px = int(dilation_px)
        self.min_area_frac = float(min_area_frac)

        # resolve class ids from names/ids in the loaded model
        self.class_ids = self._resolve_class_ids(classes)

        # optional ROI keep-mask
        self.keep_mask = None
        if keep_mask_path:
            m = cv2.imread(str(keep_mask_path), cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise FileNotFoundError(f"keep_mask not found: {keep_mask_path}")
            self.keep_mask = (m > 0).astype(np.uint8) * 255  # 0/255

    def car_mask(self, frame_bgr: np.ndarray, conf: float = 0.1, iou: float = 0.7) -> np.ndarray:
        """
        Returns a binary mask (H, W) where True are vehicle pixels (union of all vehicles).
        """
        H, W = frame_bgr.shape[:2]
        img = frame_bgr

        # apply ROI keep-mask if provided (resize once to frame size)
        if self.keep_mask is not None:
            if self.keep_mask.shape != (H, W):
                km = cv2.resize(self.keep_mask, (W, H), interpolation=cv2.INTER_NEAREST)
            else:
                km = self.keep_mask
            img = cv2.bitwise_and(img, img, mask=km)

        # run model on the numpy image directly
        res = self.model.predict(
            source=img,
            imgsz=self.imgsz,
            max_det=self.max_det,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False,
        )[0]  

        # union mask (uint8 0/255)
        union = np.zeros((H, W), np.uint8)

        # choose seg vs box
        want_seg = self.use_seg in ("auto", "seg")
        have_seg = hasattr(res, "masks") and (res.masks is not None) and (getattr(res.masks, "data", None) is not None)

        if want_seg and have_seg:
            # instance masks (N,h',w') in model scale -> resize to frame
            mdat = res.masks.data.cpu().numpy()
            # apply class filter by indexing detections
            keep_idx = self._keep_indices(res)
            for i in keep_idx:
                m_small = mdat[i]
                m = cv2.resize(m_small, (W, H), interpolation=cv2.INTER_NEAREST)
                if self.min_area_frac > 0 and (m > 0.5).sum() < self.min_area_frac * H * W:
                    continue
                union = np.maximum(union, (m > 0.5).astype(np.uint8) * 255)
        else:
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
                cls = res.boxes.cls.cpu().numpy().astype(int)
                for i, (x1, y1, x2, y2) in enumerate(xyxy):
                    if cls[i] not in self.class_ids: 
                        continue
                    # clamp and fill
                    x1 = max(0, min(x1, W - 1)); x2 = max(0, min(x2, W))
                    y1 = max(0, min(y1, H - 1)); y2 = max(0, min(y2, H))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    if self.min_area_frac > 0 and (x2 - x1) * (y2 - y1) < self.min_area_frac * H * W:
                        continue
                    union[y1:y2, x1:x2] = 255

        if self.dilation_px > 0 and union.any():
            k = np.ones((self.dilation_px, self.dilation_px), np.uint8)
            union = cv2.dilate(union, k, 1)

        return union.astype(bool)

    def _resolve_class_ids(self, wanted: Sequence[str | int]) -> List[int]:
        names = self.model.names
        if isinstance(names, dict):
            name2id = {v: int(k) for k, v in names.items()}
        else:
            name2id = {n: i for i, n in enumerate(names)}
            
        ids: List[int] = []
        for t in wanted:
            if isinstance(t, int) or (isinstance(t, str) and t.isdigit()):
                ids.append(int(t))
            else:
                if t not in name2id:
                    raise ValueError(f"Unknown class name: {t}. Available: {sorted(name2id.keys())}")
                ids.append(name2id[t])
        return sorted(set(ids))

    def _keep_indices(self, res) -> List[int]:
        """indices of detections whose class is in self.class_ids"""
        if res.boxes is None or len(res.boxes) == 0:
            return []
        cls = res.boxes.cls.cpu().numpy().astype(int)
        return [i for i, c in enumerate(cls) if c in self.class_ids]