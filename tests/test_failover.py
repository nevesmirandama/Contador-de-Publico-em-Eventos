from __future__ import annotations

import cv2
import time


def main(source="0"):
    cap = cv2.VideoCapture(0 if source == "0" else source)
    if not cap.isOpened():
        print("Erro: fonte indisponivel.")
        return
    print("Teste de tolerancia a falhas iniciado. Desconecte a webcam e reconecte. Q para sair.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Aviso amigavel: frame nao recebido. Tentando retomar...")
            time.sleep(1.0)
            cap.release()
            cap = cv2.VideoCapture(0 if source == "0" else source)
            continue
        cv2.imshow("Teste Failover", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
