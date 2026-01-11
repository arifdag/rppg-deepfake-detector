# rPPG Deepfake Detector

This project implements deepfake detection methods based on **remote Photoplethysmography (rPPG)** cues extracted from facial video. It focuses on detecting subtle physiological signs (blood volume changes) that are often disrupted or absent in generated videos.

## Models Implemented

### 1. rPPG Transformer (New)
**Notebook:** `deepfake-detection-transformer.ipynb`

A state-of-the-art approach that uses a hybrid architecture to capture long-range temporal dependencies in physiological signals.

**Architecture:**
- **3D CNN Stem**: Extracts low-level spatial-temporal features from video clips.
- **Spatial Attention**: Focuses on stable skin regions (cheeks/forehead) that contain the strongest pulse signal.
- **Transformer Encoder**: Models global temporal context using self-attention.

```mermaid
graph LR
    Video["Video Clip"] --> Prep["Preprocessing<br/>(Face Detect + Crop)"]
    Prep --> Stem["3D CNN Stem<br/>(Spatial Features)"]
    Stem --> Attn["Spatial Attention"]
    Attn --> Trans["Transformer Encoder<br/>(Temporal Context)"]
    Trans --> CLS["CLS Token"]
    CLS --> Class["Classifier<br/>(Real vs Fake)"]
    style Trans fill:#e0e0e0,stroke:#333,stroke-width:2px,color:black
```

### 2. DeepFakesON-Phys Baseline
**Notebook:** `deepfake-detection-baseline.ipynb`

An implementation inspired by the **DeepFakesON-Phys** model. It uses a pure **CAN-like 3D CNN** to capture spatiotemporal patterns without the Transformer component. This serves as a strong baseline for comparison.

## Pipeline Overview

Both models share a common preprocessing pipeline:

1.  **Face Detection**: Uses MTCNN (or Haar Cascade) to locate faces.
2.  **ROI Cropping**: Extracts skin regions (cheeks/forehead) and resizes to a fixed size (e.g., 112x112).
3.  **Signal Extraction**: Visual cues are fed into the neural network to differentiate between real human pulses and deepfake artifacts.

## Dataset
The models are designed to train and evaluate on the **Celeb-DF v2** dataset.
