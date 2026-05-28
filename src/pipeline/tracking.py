from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Dict, List, Tuple

from .detection import box_centroid, BBox


@dataclass
class Track:
    track_id: int
    centroid: Tuple[int, int]
    bbox: BBox
    age: int = 0
    misses: int = 0
    last_side: int = 0
    last_count_frame: int = -9999


class CentroidTracker:
    def __init__(self, max_distance: float = 60.0, max_misses: int = 20):
        self.max_distance = max_distance
        self.max_misses = max_misses
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    @staticmethod
    def _distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return hypot(a[0] - b[0], a[1] - b[1])

    def update(self, detections: List[BBox]) -> Dict[int, Track]:
        det_centroids = [box_centroid(box) for box in detections]
        unmatched_dets = set(range(len(detections)))

        for track in list(self.tracks.values()):
            best_idx = None
            best_dist = float("inf")
            for idx, centroid in enumerate(det_centroids):
                if idx not in unmatched_dets:
                    continue
                dist = self._distance(track.centroid, centroid)
                if dist < best_dist and dist <= self.max_distance:
                    best_dist = dist
                    best_idx = idx
            if best_idx is not None:
                track.centroid = det_centroids[best_idx]
                track.bbox = detections[best_idx]
                track.age += 1
                track.misses = 0
                unmatched_dets.remove(best_idx)
            else:
                track.misses += 1
                track.age += 1

        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id].misses > self.max_misses:
                del self.tracks[track_id]

        for idx in unmatched_dets:
            self.tracks[self.next_id] = Track(
                track_id=self.next_id,
                centroid=det_centroids[idx],
                bbox=detections[idx],
            )
            self.next_id += 1

        return self.tracks
