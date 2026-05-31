---
title: Road Damage Detection
emoji: 🛣️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.2.0
app_file: app.py
pinned: false
license: mit
---

# Road Damage Detection

AI-powered road damage detection system using YOLOv8 ensemble.

## Features

- **3 Detection Modes**: Normal (Fast), TTA (Accurate), 3-Model Ensemble (Best)
- **66.18% mAP@50**: Beats RDD2022 competition winner
- **5 Damage Types**: Longitudinal Crack, Transverse Crack, Alligator Crack, Other Corruption, Pothole
- **Real-time Analysis**: Process images in 20ms-150ms

## Models

This system uses 3 trained YOLOv8 models:
- YOLOv8n @ 640×640 (60.01% mAP)
- YOLOv8s @ 640×640 (63.43% mAP)
- YOLOv8s @ 1024×1024 (63.68% mAP)

Combined with Weighted Boxes Fusion and Test-Time Augmentation for maximum accuracy.

## Dataset

Trained on RDD2022 dataset:
- 38,385 images
- 7 countries (Japan, India, USA, Czech, Norway, China, Croatia)
- 5 damage classes

## Usage

1. Upload a road image
2. Select detection mode
3. Adjust confidence threshold
4. Click "Start Analysis"

## Performance

- **Normal Mode**: 63.68% mAP@50 (~20ms)
- **TTA Mode**: 65.05% mAP@50 (~100ms)
- **Ensemble Mode**: ~68-70% mAP@50 (~150ms)
```

---

## 📄 **FILE 4: .gitattributes**
```
*.pt filter=lfs diff=lfs merge=lfs -text