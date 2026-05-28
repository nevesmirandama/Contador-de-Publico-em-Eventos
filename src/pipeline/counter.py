from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

Point = Tuple[int, int]


def point_side(point: Point, a: Point, b: Point) -> int:
    value = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


@dataclass
class CountEvent:
    track_id: int
    direction: str
    centroid: Point


class LineCounter:
    def __init__(self, p1: Point, p2: Point, min_gap_frames: int = 10):
        self.p1 = p1
        self.p2 = p2
        self.min_gap_frames = min_gap_frames
        self.entries = 0
        self.exits = 0

    def check_crossing(self, track, frame_index: int) -> Optional[CountEvent]:
        current_side = point_side(track.centroid, self.p1, self.p2)

        if track.last_side == 0:
            track.last_side = current_side
            return None

        if current_side == 0:
            return None

        # Só conta quando realmente muda de lado e respeita intervalo mínimo
        if current_side != track.last_side and frame_index - track.last_count_frame >= self.min_gap_frames:
            # REGRA AJUSTADA:
            #  1  -> -1 = entrada
            # -1  ->  1 = saída
            if track.last_side == 1 and current_side == -1:
                self.entries += 1
                direction = "entrada"
            elif track.last_side == -1 and current_side == 1:
                self.exits += 1
                direction = "saida"
            else:
                track.last_side = current_side
                return None

            track.last_side = current_side
            track.last_count_frame = frame_index
            return CountEvent(track_id=track.track_id, direction=direction, centroid=track.centroid)

        track.last_side = current_side
        return None
