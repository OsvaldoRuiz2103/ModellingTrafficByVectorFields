import cv2, numpy as np

# VOC class order for MobileNet-SSD
_VOC = ["background","aeroplane","bicycle","bird","boat","bottle","bus","car",
        "cat","chair","cow","diningtable","dog","horse","motorbike","person",
        "pottedplant","sheep","sofa","train","tvmonitor"]

class CarDetector:
    def __init__(self, prototxt_path: str, caffemodel_path: str, use_gpu: bool = False):
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
        if use_gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.allowed_ids = { _VOC.index(cls) for cls in ["car", "bus", "motorbike"] }

    def car_mask(self, frame_bgr, conf: float = 0.4, expand_px: int = 6) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(frame_bgr, 0.007843, (300, 300), 127.5, swapRB=False, crop=False)
        self.net.setInput(blob)
        det = self.net.forward() 

        mask = np.zeros((h, w), np.uint8)
        for i in range(det.shape[2]):
            score = float(det[0, 0, i, 2]); cls = int(det[0, 0, i, 1])
            if score >= conf and cls in self.allowed_ids:
                x1, y1, x2, y2 = (det[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
                x1 = max(0, x1 - expand_px); y1 = max(0, y1 - expand_px)
                x2 = min(w - 1, x2 + expand_px); y2 = min(h - 1, y2 + expand_px)
                mask[y1:y2+1, x1:x2+1] = 255

        if expand_px > 0:
            k = np.ones((expand_px, expand_px), np.uint8)
            mask = cv2.dilate(mask, k, 1)
        return mask.astype(bool)
