# utility/detector.py
import cv2
import numpy as np
import time
import base64
from ultralytics import YOLO

model_s_1024 = YOLO("yolov8s_1024_best.pt")

CLASS_NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Other Corruption",
    4: "Pothole"
}

def detect_damage(image_np, confidence=0.25):
    start = time.time()

    results = model_s_1024(image_np, imgsz=1024, conf=confidence)[0]

    # ✅ Annotated image (same as Gradio)
    annotated = results.plot(line_width=3, labels=True, conf=True)
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # ✅ Encode image to base64
    _, buffer = cv2.imencode(".png", annotated)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        detections.append({
            "label": CLASS_NAMES[cls_id],
            "confidence": round(float(box.conf[0]) * 100, 2),
            "bbox": [int(x) for x in box.xyxy[0]]
        })

    return {
        "total": len(detections),
        "detections": detections,
        "time": round(time.time() - start, 2),
        "image": img_base64  # 🔥 THIS is the key
    }
# # utility/detector.py
# import cv2
# from ultralytics import YOLO

# model_s_1024 = YOLO("yolov8s_1024_best.pt")

# def detect_frame(frame, confidence=0.25):
#     results = model_s_1024(frame, imgsz=1024, conf=confidence)[0]

#     annotated = results.plot(line_width=2, labels=True, conf=True)
#     return annotated
