from __future__ import annotations

import statistics
import time
import cv2


def main(source="0", duration_sec=60):
    cap = cv2.VideoCapture(0 if source == "0" else source)
    if not cap.isOpened():
        print("Erro: nao foi possivel abrir a fonte.")
        return
    samples = []
    start = time.time()
    last = time.time()
    while time.time() - start < duration_sec:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        dt = now - last
        if dt > 0:
            samples.append(1.0 / dt)
        last = now
    cap.release()
    if not samples:
        print("Sem amostras de FPS.")
        return
    print(f"FPS medio: {statistics.mean(samples):.2f}")
    print(f"FPS min: {min(samples):.2f}")
    print(f"FPS max: {max(samples):.2f}")
    print(f"FPS desvio-padrao: {statistics.pstdev(samples):.2f}")


if __name__ == "__main__":
    main()
