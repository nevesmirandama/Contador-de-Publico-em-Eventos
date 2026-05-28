from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]


def detect_moving_objects(mask, min_area: int = 1200, min_height: int = 80,
                          min_aspect: float = 0.18, max_aspect: float = 1.20) -> List[BBox]:
    """Retorna blobs de movimento com formato mais compatível com corpo humano."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[BBox] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if h < min_height:
            continue
        aspect = w / float(max(h, 1))
        if aspect < min_aspect or aspect > max_aspect:
            continue
        boxes.append((x, y, w, h))
    return merge_boxes(boxes)


def create_hog_detector():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


def detect_people_hog(frame, hog, hit_threshold: float = 0.0,
                      win_stride: tuple[int, int] = (4, 4),
                      padding: tuple[int, int] = (8, 8),
                      scale: float = 1.03) -> List[BBox]:
    """Detecção de pessoa com HOG do OpenCV, com upscaling para alvos pequenos."""
    h, w = frame.shape[:2]
    target_w = 960
    ratio = max(target_w / float(w), 1.0)
    work = frame if ratio == 1.0 else cv2.resize(frame, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)

    rects, _weights = hog.detectMultiScale(
        work,
        hitThreshold=hit_threshold,
        winStride=win_stride,
        padding=padding,
        scale=scale,
        useMeanshiftGrouping=False,
    )

    boxes = []
    for (x, y, rw, rh) in rects:
        x = int(x / ratio)
        y = int(y / ratio)
        rw = int(rw / ratio)
        rh = int(rh / ratio)
        boxes.append((x, y, rw, rh))
    return non_max_suppression(boxes, overlap_thresh=0.35)


def combine_person_detections(frame, mask, hog, min_area: int = 1200) -> List[BBox]:
    """Modo híbrido: prioriza HOG e complementa com blobs de movimento plausíveis.

    Regras:
    - Se o blob de movimento sobrepõe uma pessoa do HOG, ele pode expandir levemente a caixa.
    - Se o HOG falha, ainda usamos blobs com formato de pessoa para não zerar a detecção.
    """
    person_boxes = detect_people_hog(frame, hog)
    motion_boxes = detect_moving_objects(mask, min_area=min_area)

    if not person_boxes and not motion_boxes:
        return []

    if not person_boxes:
        return motion_boxes

    merged: List[BBox] = []
    used_motion = set()
    for p in person_boxes:
        best = p
        best_idx = None
        best_iou = 0.0
        for idx, m in enumerate(motion_boxes):
            score = iou(p, m)
            if score > best_iou:
                best_iou = score
                best_idx = idx
        if best_idx is not None and best_iou >= 0.10:
            best = union_box(p, motion_boxes[best_idx])
            used_motion.add(best_idx)
        merged.append(best)

    # aceita alguns blobs extras de movimento se estiverem grandes e próximos do padrão humano
    for idx, m in enumerate(motion_boxes):
        if idx in used_motion:
            continue
        x, y, w, h = m
        if h >= 120 and 0.20 <= (w / float(h)) <= 0.85:
            merged.append(m)

    return non_max_suppression(merge_boxes(merged), overlap_thresh=0.30)


def merge_boxes(boxes: List[BBox], iou_thresh: float = 0.20) -> List[BBox]:
    changed = True
    boxes = boxes[:]
    while changed and len(boxes) > 1:
        changed = False
        result: List[BBox] = []
        skip = set()
        for i in range(len(boxes)):
            if i in skip:
                continue
            current = boxes[i]
            for j in range(i + 1, len(boxes)):
                if j in skip:
                    continue
                if iou(current, boxes[j]) >= iou_thresh:
                    current = union_box(current, boxes[j])
                    skip.add(j)
                    changed = True
            result.append(current)
        boxes = result
    return boxes


def non_max_suppression(boxes: List[BBox], overlap_thresh: float = 0.35) -> List[BBox]:
    if not boxes:
        return []
    arr = np.array([[x, y, x + w, y + h] for x, y, w, h in boxes], dtype=float)
    pick = []

    x1 = arr[:, 0]
    y1 = arr[:, 1]
    x2 = arr[:, 2]
    y2 = arr[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    out = []
    for i in pick:
        xx1, yy1, xx2, yy2 = arr[i].astype(int)
        out.append((xx1, yy1, xx2 - xx1, yy2 - yy1))
    return out


def union_box(a: BBox, b: BBox) -> BBox:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = min(ax, bx)
    y1 = min(ay, by)
    x2 = max(ax + aw, bx + bw)
    y2 = max(ay + ah, by + bh)
    return (x1, y1, x2 - x1, y2 - y1)


def iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    xA = max(ax, bx)
    yA = max(ay, by)
    xB = min(ax + aw, bx + bw)
    yB = min(ay + ah, by + bh)
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / float(max(union, 1))


def box_centroid(box: BBox) -> tuple[int, int]:
    x, y, w, h = box
    return (x + w // 2, y + h // 2)
