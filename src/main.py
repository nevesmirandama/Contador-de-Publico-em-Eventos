from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from pipeline.counter import LineCounter
from pipeline.detection import (
    combine_person_detections,
    create_hog_detector,
    detect_moving_objects,
    detect_people_hog,
)
from pipeline.preprocessing import create_background_subtractor, preprocess_mask
from pipeline.tracking import CentroidTracker
from ui.line_selector import LineSelector
from utils.csv_logger import CsvEventLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Grupo 5 - Contador de Público em Eventos")
    parser.add_argument("--source", default="0", help="0 para webcam ou caminho para arquivo de vídeo")
    parser.add_argument("--bg", default="mog2", choices=["mog2", "knn"], help="Algoritmo de subtração de fundo")
    parser.add_argument("--detector", default="motion", choices=["motion", "hog", "hybrid"],
                        help="motion=movimento, hog=pessoa, hybrid=pessoa+movimento")
    parser.add_argument("--min-area", type=int, default=1800, help="Área mínima do contorno")
    parser.add_argument("--max-distance", type=float, default=90.0, help="Distância máxima entre centróides")
    parser.add_argument("--max-misses", type=int, default=25, help="Máximo de frames sem detecção")
    parser.add_argument("--csv", default="results/events.csv", help="Caminho do CSV")
    parser.add_argument("--line", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"), help="Linha fixa: x1 y1 x2 y2")
    parser.add_argument("--save-frame-dir", default="results", help="Diretório para salvar frames")
    return parser.parse_args()


def open_capture(source: str):
    return cv2.VideoCapture(0 if source == "0" else source)


def draw_dashboard(frame, counter: LineCounter, fps: float, flow_per_min: float, detector_name: str):
    h, _w = frame.shape[:2]
    cv2.rectangle(frame, (10, 10), (355, 155), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (355, 155), (0, 255, 0), 2)
    cv2.putText(frame, f"Detector: {detector_name}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Entradas: {counter.entries}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Saidas: {counter.exits}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(frame, f"Fluxo/min: {flow_per_min:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (215, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Q=sair  R=linha  S=salvar frame", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def get_boxes(frame, mask, detector_mode: str, hog, min_area: int):
    if detector_mode == "motion":
        return detect_moving_objects(mask, min_area=min_area)
    if detector_mode == "hog":
        return detect_people_hog(frame, hog)
    return combine_person_detections(frame, mask, hog, min_area=min_area)


def warmup_subtractor(cap, subtractor, frames: int = 20):
    # queima os primeiros frames para o fundo não nascer todo branco
    for _ in range(frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        subtractor.apply(frame)


def main():
    args = parse_args()
    cap = open_capture(args.source)
    if not cap.isOpened():
        print("Erro: nao foi possivel abrir a fonte de video.")
        return

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        print("Erro: nao foi possivel ler o primeiro frame.")
        cap.release()
        return

    if args.line:
        p1 = (args.line[0], args.line[1])
        p2 = (args.line[2], args.line[3])
    else:
        p1, p2 = LineSelector().select(first_frame)

    subtractor = create_background_subtractor(args.bg)
    hog = create_hog_detector()
    # volta para o início se for arquivo; em webcam segue do ponto atual
    if args.source != "0":
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    warmup_subtractor(cap, subtractor, frames=12)

    tracker = CentroidTracker(max_distance=args.max_distance, max_misses=args.max_misses)
    counter = LineCounter(p1, p2)
    logger = CsvEventLogger(args.csv)

    save_dir = Path(args.save_frame_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    start_time = time.time()
    last_time = time.time()
    fps = 0.0
    last_mask = None

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Aviso: frame nao recebido. Encerrando captura.")
            break

        frame_idx += 1
        if args.detector == "hog":
            mask = None
            last_mask = None
            boxes = get_boxes(frame, None, args.detector, hog, args.min_area)
        else:
            mask = preprocess_mask(frame, subtractor)
            last_mask = mask
            boxes = get_boxes(frame, mask, args.detector, hog, args.min_area)
        tracks = tracker.update(boxes)

        for track in tracks.values():
            event = counter.check_crossing(track, frame_idx)
            if event:
                logger.log(event.direction, event.track_id, event.centroid)

        cv2.line(frame, p1, p2, (0, 255, 255), 2)

        for track in tracks.values():
            x, y, w, h = track.bbox
            cx, cy = track.centroid
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"Movimento {track.track_id}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        now = time.time()
        delta = now - last_time
        if delta > 0:
            fps = 1.0 / delta
        last_time = now

        elapsed_min = max((now - start_time) / 60.0, 1e-6)
        flow_per_min = (counter.entries + counter.exits) / elapsed_min
        draw_dashboard(frame, counter, fps, flow_per_min, args.detector)

        cv2.imshow("Grupo 5 - Contador de Publico", frame)
        if last_mask is not None:
            cv2.imshow("Mascara", last_mask)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        elif key in (ord("r"), ord("R")):
            ok2, current = cap.read()
            if ok2 and current is not None:
                p1, p2 = LineSelector().select(current)
                counter = LineCounter(p1, p2)
        elif key in (ord("s"), ord("S")):
            out_path = save_dir / f"frame_{frame_idx}.png"
            cv2.imwrite(str(out_path), frame)
            print(f"Frame salvo em: {out_path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finalizado. Entradas={counter.entries} | Saidas={counter.exits}")


if __name__ == "__main__":
    main()
