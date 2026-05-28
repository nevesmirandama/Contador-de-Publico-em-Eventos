from __future__ import annotations

from typing import Optional, Tuple

import cv2

Point = Tuple[int, int]


class LineSelector:
    def __init__(self, window_name: str = "Defina a linha"):
        self.window_name = window_name
        self.start: Optional[Point] = None
        self.end: Optional[Point] = None
        self.drawing = False

    def _callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
            self.end = (x, y)
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.end = (x, y)
            self.drawing = False

    def select(self, frame) -> tuple[Point, Point]:
        clone = frame.copy()
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._callback)
        while True:
            display = clone.copy()
            if self.start and self.end:
                cv2.line(display, self.start, self.end, (0, 255, 255), 2)
                cv2.circle(display, self.start, 4, (0, 255, 0), -1)
                cv2.circle(display, self.end, 4, (0, 0, 255), -1)
            cv2.putText(display, "Clique e arraste. ENTER confirma. R reinicia.",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(10) & 0xFF
            if key == 13 and self.start and self.end:
                break
            if key in (ord('r'), ord('R')):
                self.start, self.end = None, None
        cv2.destroyWindow(self.window_name)
        return self.start, self.end
