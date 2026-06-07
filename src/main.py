import cv2
import numpy as np
from ultralytics import YOLO

# =====================================================
# CONFIGURAÇÕES
# =====================================================
CAMERA_INDEX = 0

# Modelo de pose.
# Se quiser testar outros modelos, pode trocar por:
# "yolo11n-pose.pt" ou "yolo26n-pose.pt", dependendo da sua versão do Ultralytics.
MODEL_NAME = "yolov8n-pose.pt"

CONFIDENCE = 0.35
KEYPOINT_CONFIDENCE = 0.35

# Quantos frames uma pessoa pode sumir antes de ser considerada fora do ambiente.
# Isso ajuda quando duas pessoas ficam muito próximas e o detector perde uma por instantes.
MAX_FRAMES_DESAPARECIDO = 20

WINDOW_NAME = "Contador por Cabeca e Ombros"

# =====================================================
# ÍNDICES DOS KEYPOINTS COCO
# =====================================================
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6

# =====================================================
# FUNÇÕES AUXILIARES
# =====================================================
def ponto_valido(kpts, idx, min_conf=KEYPOINT_CONFIDENCE):
    """
    Verifica se um keypoint existe e tem confiança suficiente.
    Formato esperado do keypoint: [x, y, conf]
    """
    if kpts is None:
        return False

    if idx >= len(kpts):
        return False

    x, y, conf = kpts[idx]

    if conf < min_conf:
        return False

    if x <= 0 or y <= 0:
        return False

    return True


def obter_ponto_cabeca(kpts):
    """
    Retorna o melhor ponto para representar a cabeça.
    Prioridade:
    1. Nariz
    2. Média dos olhos
    3. Média das orelhas
    """
    if ponto_valido(kpts, NOSE):
        x, y, _ = kpts[NOSE]
        return int(x), int(y)

    olhos = []
    for idx in [LEFT_EYE, RIGHT_EYE]:
        if ponto_valido(kpts, idx):
            x, y, _ = kpts[idx]
            olhos.append((x, y))

    if len(olhos) > 0:
        x = int(np.mean([p[0] for p in olhos]))
        y = int(np.mean([p[1] for p in olhos]))
        return x, y

    orelhas = []
    for idx in [LEFT_EAR, RIGHT_EAR]:
        if ponto_valido(kpts, idx):
            x, y, _ = kpts[idx]
            orelhas.append((x, y))

    if len(orelhas) > 0:
        x = int(np.mean([p[0] for p in orelhas]))
        y = int(np.mean([p[1] for p in orelhas]))
        return x, y

    return None


def validar_formato_cabeca_ombros(kpts):
    """
    Valida se a pessoa tem pelo menos:
    - cabeça detectada;
    - ombro esquerdo;
    - ombro direito.

    Também valida se os ombros estão abaixo da cabeça.
    Isso reduz falsos positivos.
    """
    cabeca = obter_ponto_cabeca(kpts)

    if cabeca is None:
        return False, None, None, None

    if not ponto_valido(kpts, LEFT_SHOULDER):
        return False, None, None, None

    if not ponto_valido(kpts, RIGHT_SHOULDER):
        return False, None, None, None

    head_x, head_y = cabeca

    lx, ly, _ = kpts[LEFT_SHOULDER]
    rx, ry, _ = kpts[RIGHT_SHOULDER]

    lx, ly = int(lx), int(ly)
    rx, ry = int(rx), int(ry)

    # Ombros devem estar abaixo da cabeça.
    if ly <= head_y or ry <= head_y:
        return False, None, None, None

    largura_ombros = abs(rx - lx)

    # Evita aceitar detecções deformadas demais.
    if largura_ombros < 20:
        return False, None, None, None

    # Centro entre cabeça e ombros.
    centro_x = int((head_x + lx + rx) / 3)
    centro_y = int((head_y + ly + ry) / 3)

    return True, (head_x, head_y), (lx, ly), (rx, ry), (centro_x, centro_y)


def desenhar_cabeca_ombros(frame, head, left_shoulder, right_shoulder, centro, track_id):
    """
    Desenha o padrão cabeça + ombros.
    """
    hx, hy = head
    lx, ly = left_shoulder
    rx, ry = right_shoulder
    cx, cy = centro

    # Cabeça
    cv2.circle(frame, (hx, hy), 7, (0, 255, 255), -1)

    # Ombros
    cv2.circle(frame, (lx, ly), 7, (255, 0, 0), -1)
    cv2.circle(frame, (rx, ry), 7, (255, 0, 0), -1)

    # Linhas cabeça/ombros
    cv2.line(frame, (hx, hy), (lx, ly), (0, 255, 0), 2)
    cv2.line(frame, (hx, hy), (rx, ry), (0, 255, 0), 2)
    cv2.line(frame, (lx, ly), (rx, ry), (0, 255, 0), 2)

    # Centro usado para saber se está dentro da área
    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    cv2.putText(
        frame,
        f"ID {track_id}",
        (hx + 10, hy - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )


# =====================================================
# INICIAR MODELO E WEBCAM
# =====================================================
model = YOLO(MODEL_NAME)

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Nao foi possivel abrir a webcam.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ret, frame = cap.read()

if not ret:
    cap.release()
    raise RuntimeError("Nao foi possivel capturar imagem da webcam.")

h, w = frame.shape[:2]

# =====================================================
# ÁREA DO AMBIENTE
# =====================================================
# Ajuste esta região conforme o ambiente real.
# Aqui está usando quase a imagem inteira.
roi_points = np.array([
    [40, 40],
    [w - 40, 40],
    [w - 40, h - 40],
    [40, h - 40]
], dtype=np.int32)

# Dicionário dos IDs rastreados.
# Cada pessoa terá:
# - presente: True/False
# - frames_desaparecido: contador para evitar bug de sumiço rápido
# - ultimo_centro: última posição conhecida
pessoas = {}

while True:
    ret, frame = cap.read()

    if not ret:
        break

    output = frame.copy()

    # Desenhar área do ambiente
    cv2.polylines(
        output,
        [roi_points],
        isClosed=True,
        color=(0, 255, 255),
        thickness=2
    )

    # =====================================================
    # DETECÇÃO + RASTREAMENTO COM POSE
    # =====================================================
    results = model.track(
        source=frame,
        persist=True,
        conf=CONFIDENCE,
        tracker="botsort.yaml",
        verbose=False
    )

    ids_detectados_agora = set()

    if results and len(results) > 0:
        result = results[0]

        # Verifica se existem IDs e keypoints
        if (
            result.boxes is not None
            and result.boxes.id is not None
            and result.keypoints is not None
        ):
            track_ids = result.boxes.id.cpu().numpy().astype(int)

            # xy: coordenadas dos keypoints
            # conf: confiança dos keypoints
            kpts_xy = result.keypoints.xy.cpu().numpy()
            kpts_conf = result.keypoints.conf.cpu().numpy()

            for i, track_id in enumerate(track_ids):
                xy = kpts_xy[i]
                conf = kpts_conf[i]

                # Junta x, y e confiança em uma matriz [17, 3]
                kpts = np.concatenate(
                    [xy, conf.reshape(-1, 1)],
                    axis=1
                )

                validacao = validar_formato_cabeca_ombros(kpts)

                if not validacao[0]:
                    continue

                _, head, left_shoulder, right_shoulder, centro = validacao

                cx, cy = centro

                dentro_roi = cv2.pointPolygonTest(
                    roi_points,
                    (cx, cy),
                    False
                ) >= 0

                if not dentro_roi:
                    continue

                ids_detectados_agora.add(track_id)

                # Se é uma pessoa nova, cadastra.
                if track_id not in pessoas:
                    pessoas[track_id] = {
                        "presente": True,
                        "frames_desaparecido": 0,
                        "ultimo_centro": centro
                    }
                else:
                    pessoas[track_id]["presente"] = True
                    pessoas[track_id]["frames_desaparecido"] = 0
                    pessoas[track_id]["ultimo_centro"] = centro

                desenhar_cabeca_ombros(
                    output,
                    head,
                    left_shoulder,
                    right_shoulder,
                    centro,
                    track_id
                )

    # =====================================================
    # CONTROLE CONTRA BUG DE PESSOAS MUITO PRÓXIMAS
    # =====================================================
    for track_id in list(pessoas.keys()):
        if pessoas[track_id]["presente"]:
            if track_id not in ids_detectados_agora:
                pessoas[track_id]["frames_desaparecido"] += 1

                # Se sumiu por poucos frames, mantém como presente.
                if pessoas[track_id]["frames_desaparecido"] <= MAX_FRAMES_DESAPARECIDO:
                    continue

                # Se sumiu por muitos frames, considera que saiu.
                pessoas[track_id]["presente"] = False

    total_presentes = sum(
        1 for dados in pessoas.values()
        if dados["presente"]
    )

    # =====================================================
    # INFORMAÇÕES NA TELA
    # =====================================================
    cv2.rectangle(output, (10, 10), (470, 120), (0, 0, 0), -1)

    cv2.putText(
        output,
        f"Pessoas no ambiente: {total_presentes}",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        "Validacao: cabeca + ombros",
        (25, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

    cv2.putText(
        output,
        "Pressione 'q' para sair",
        (25, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

    cv2.imshow(WINDOW_NAME, output)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()