from __future__ import annotations

import cv2


def create_background_subtractor(method: str = "mog2"):
    method = method.lower()
    if method == "knn":
        return cv2.createBackgroundSubtractorKNN(detectShadows=True)
    return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=32, detectShadows=True)


def preprocess_mask(frame, subtractor, kernel_size: int = 5):
    fg_mask = subtractor.apply(frame)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
    return fg_mask
