import cv2
import time
from ultralytics import YOLO

# Load model ONCE
model = YOLO("yolov8s_1024_best.pt")

CLASS_NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Other Corruption",
    4: "Pothole"
}

def webcam_stream():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    if not cap.isOpened():
        return

    frame_id = 0
    PROCESS_EVERY_N_FRAMES = 3   # 🔥 balanced
    CONFIDENCE = 0.20            # 🔥 lower for live cam
    FPS_LIMIT = 12

    last_time = time.time()
    annotated_frame = None

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_id += 1

            # Run YOLO every N frames
            if frame_id % PROCESS_EVERY_N_FRAMES == 0:
                results = model(
                    frame,
                    imgsz=640,
                    conf=CONFIDENCE,
                    verbose=False
                )[0]

                annotated = frame.copy()

                if len(results.boxes) > 0:
                    for box in results.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])

                        label = f"{CLASS_NAMES[cls_id]} {conf*100:.1f}%"

                        # Bounding box
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Label background
                        (tw, th), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                        )
                        cv2.rectangle(
                            annotated,
                            (x1, y1 - th - 8),
                            (x1 + tw + 6, y1),
                            (0, 255, 0),
                            -1
                        )

                        # Label text
                        cv2.putText(
                            annotated,
                            label,
                            (x1 + 3, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 0),
                            2
                        )
                else:
                    # No detection → show normal frame
                    cv2.putText(
                        annotated,
                        "No road damage detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                annotated_frame = annotated

            # Use last processed frame
            output = annotated_frame if annotated_frame is not None else frame

            # FPS limiter
            elapsed = time.time() - last_time
            time.sleep(max(0, (1 / FPS_LIMIT) - elapsed))
            last_time = time.time()

            _, buffer = cv2.imencode(
                ".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            )

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                buffer.tobytes() +
                b"\r\n"
            )

    finally:
        cap.release()
