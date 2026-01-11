# rPPG Deepfake Detector

A deepfake detection system using **remote Photoplethysmography (rPPG)** and visual features to detect subtle physiological signs that are disrupted or absent in manipulated videos.

## Models

| Model | Test AUC | Test Accuracy | Balanced Acc |
|-------|----------|---------------|--------------|
| Baseline (3D CNN) | 0.7555 | 68.73% | 69.89% |
| Transformer | 0.8169 | 71.62% | 73.43% |
| **Fusion (Visual + rPPG)** | **0.9708** | **90.54%** | **91.05%** |

*Trained and evaluated on Celeb-DF v2 dataset*

### 1. DeepFakesON-Phys Baseline
`notebooks/deepfake-detection-baseline.ipynb`

3D CNN architecture inspired by DeepFakesON-Phys for spatiotemporal pattern detection from rPPG signals.

### 2. rPPG Transformer
`notebooks/deepfake-detection-transformer.ipynb`

Hybrid 3D CNN stem + Transformer encoder for capturing long-range temporal dependencies.

### 3. Visual-rPPG Fusion
`notebooks/deepfake-detection-fusion.ipynb`

Best performing model combining EfficientNet-B2 visual features with POS-based rPPG physiological signals. Uses Focal Loss and Test-Time Augmentation.

---

## Web Application

Flask-based interface for video deepfake detection.

![demo](https://github.com/user-attachments/assets/bc2d05db-c53c-4338-97ba-8f8375973c93)

**Features:**
- Video upload (MP4, AVI, MOV, MKV, WebM)
- Real-time analysis with confidence scores
- GPU acceleration support

### Quick Start

```bash
cd webapp
run.bat          # Windows - creates venv and installs deps automatically
```

Or manually:
```bash
cd webapp
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000` in your browser.

**Note:** Place the trained model checkpoint at `rPPG colab checkpoint/best.pth` (relative to project root).

---

## Project Structure

```
rppg-deepfake-detector/
├── notebooks/
│   ├── deepfake-detection-baseline.ipynb
│   ├── deepfake-detection-transformer.ipynb
│   └── deepfake-detection-fusion.ipynb
├── webapp/
│   ├── app.py
│   ├── requirements.txt
│   ├── run.bat
│   ├── static/
│   └── templates/
├── README.md
└── .gitignore
```

---

## Dataset

**Celeb-DF v2**
- Training: 6,011 videos (712 real, 5,299 fake)
- Test: 518 videos (178 real, 340 fake)
- Uses official test split

---

## Limitations

- Trained on face-swap deepfakes only (Celeb-DF v2)
- May not detect AI-generated videos (Sora, Runway, etc.)
- May not detect fully synthetic faces or audio-only manipulations
- Requires visible face with natural physiological signals
