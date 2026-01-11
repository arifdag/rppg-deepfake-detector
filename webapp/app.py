import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tempfile
import timm
from scipy.signal import butter, filtfilt

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
FRAMES_PER_VIDEO = 8
FPS_TARGET = 25
RPPG_CLIP_LEN = 96
RPPG_STRIDE = 3
RPPG_BPM_BAND = (42, 240)

CHECKPOINT_PATH = Path(__file__).parent.parent / "rPPG colab checkpoint" / "best.pth"
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Model Architecture
class TemporalAttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
        )

    def forward(self, x):
        w = self.attn(x)
        w = torch.softmax(w, dim=1)
        return (x * w).sum(dim=1)


class FusionDetector(nn.Module):
    def __init__(self, backbone_name="tf_efficientnet_b2_ns", rppg_dim=16, dropout=0.4):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features

        self.temporal_pool = TemporalAttentionPool(feat_dim)

        self.rppg_mlp = nn.Sequential(
            nn.Linear(rppg_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim + 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, frames, rppg_feat):
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)
        f = self.backbone(x)
        f = f.view(B, T, -1)
        v = self.temporal_pool(f)
        r = self.rppg_mlp(rppg_feat)
        z = torch.cat([v, r], dim=1)
        logits = self.head(z).squeeze(1)
        return logits


# rPPG Feature Extraction
def _butter_bandpass(low_hz, high_hz, fs, order=3):
    """Create bandpass filter coefficients with proper frequency validation."""
    nyq = 0.5 * fs
    
    # Normalize frequencies
    low = low_hz / nyq
    high = high_hz / nyq
    
    # Clamp to valid range (0 < Wn < 1) with safety margin
    low = max(0.01, min(low, 0.98))
    high = max(0.02, min(high, 0.99))
    
    # Ensure low < high
    if low >= high:
        low = high * 0.5
    
    try:
        b, a = butter(order, [low, high], btype="band")
        return b, a
    except Exception:
        # Fallback: return identity filter coefficients
        return np.array([1.0]), np.array([1.0])


def bandpass_filter(x, fs, low_hz=0.7, high_hz=4.0):
    """Apply bandpass filter with robust error handling."""
    if len(x) < 8:
        return x
    
    # Ensure fs is reasonable (at least 2x the high frequency for Nyquist)
    min_fs = high_hz * 2.1
    if fs < min_fs:
        # Adjust high_hz to be valid for this sampling rate
        high_hz = min(high_hz, fs * 0.45)
        low_hz = min(low_hz, high_hz * 0.5)
    
    try:
        b, a = _butter_bandpass(low_hz, high_hz, fs, order=3)
        filtered = filtfilt(b, a, x)
        return filtered.astype(np.float32)
    except Exception as e:
        print(f"Bandpass filter warning: {e}, returning unfiltered signal")
        # Return normalized but unfiltered signal
        x_norm = x - np.mean(x)
        return x_norm.astype(np.float32)


def pos_rppg_from_rgb_means(rgb_means: np.ndarray, fs: float):
    """Extract rPPG signal using POS method."""
    X = rgb_means.astype(np.float32)
    if X.ndim != 2 or X.shape[1] != 3 or X.shape[0] < 8:
        return np.zeros((max(1, X.shape[0]),), dtype=np.float32)

    # Ensure minimum sampling rate
    fs = max(fs, 5.0)  # At least 5 Hz
    
    mean = X.mean(axis=0, keepdims=True) + 1e-6
    Xn = X / mean

    S1 = Xn[:, 0] - Xn[:, 1]
    S2 = Xn[:, 0] + Xn[:, 1] - 2.0 * Xn[:, 2]

    std_s1 = np.std(S1) + 1e-6
    std_s2 = np.std(S2) + 1e-6
    alpha = std_s1 / std_s2

    h = S1 - alpha * S2
    h = h - h.mean()

    # Use safe frequency range based on actual sampling rate
    low_hz = RPPG_BPM_BAND[0] / 60.0  # 0.7 Hz
    high_hz = min(RPPG_BPM_BAND[1] / 60.0, fs * 0.45)  # Cap at 45% of Nyquist
    
    h_f = bandpass_filter(h, fs=fs, low_hz=low_hz, high_hz=high_hz)
    h_f = (h_f - h_f.mean()) / (np.std(h_f) + 1e-6)
    return h_f.astype(np.float32)


def rppg_feature_vector(signal_1d: np.ndarray, fs: float):
    x = signal_1d.astype(np.float32)
    if len(x) < 16:
        return np.zeros((16,), dtype=np.float32)

    mean = float(np.mean(x))
    std = float(np.std(x))
    mad = float(np.mean(np.abs(x - np.mean(x))))
    zcr = float(np.mean((x[:-1] * x[1:]) < 0))

    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.abs(np.fft.rfft(x)) ** 2

    low_hz = RPPG_BPM_BAND[0] / 60.0
    high_hz = RPPG_BPM_BAND[1] / 60.0
    mask = (freqs >= low_hz) & (freqs <= high_hz)

    spec_band = spec[mask]
    freqs_band = freqs[mask]
    if len(spec_band) < 2:
        peak_hz = 0.0
        peak_power = 0.0
        band_power = float(spec.sum())
    else:
        peak_idx = int(np.argmax(spec_band))
        peak_hz = float(freqs_band[peak_idx])
        peak_power = float(spec_band[peak_idx])
        band_power = float(np.sum(spec_band))

    total_power = float(np.sum(spec) + 1e-9)
    band_ratio = band_power / total_power
    peak_ratio = peak_power / (band_power + 1e-9)

    med = float(np.median(spec_band) + 1e-9) if len(spec_band) else 1e-9
    snr = peak_power / med
    bpm = peak_hz * 60.0

    feats = np.array([
        mean, std, mad, zcr,
        bpm, peak_hz, peak_ratio, band_ratio,
        snr,
        float(np.percentile(x, 5)), float(np.percentile(x, 95)),
        float(np.max(x)), float(np.min(x)),
        float(np.mean(np.diff(x))), float(np.std(np.diff(x))),
        float(n)
    ], dtype=np.float32)
    return feats


# Video Processing
def read_video_frames(video_path: str, max_frames: int = 256, stride: int = 2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    idx = 0
    grabbed = True
    while grabbed and len(frames) < max_frames:
        grabbed, frame = cap.read()
        if not grabbed:
            break
        if idx % stride == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames, fps


def center_crop_square(frame_bgr, size=IMG_SIZE):
    h, w = frame_bgr.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    crop = frame_bgr[y0:y0+s, x0:x0+s]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return crop


def detect_face_cascade(frame_bgr):
    """Simple face detection using Haar cascades (no MTCNN dependency)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return np.array([x, y, x + w, y + h], dtype=np.float32)
    return None


def crop_bbox_rgb(frame_bgr, bbox, out_size=96, margin=0.2):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    x1 = max(0, int(x1 - margin * bw))
    y1 = max(0, int(y1 - margin * bh))
    x2 = min(w, int(x2 + margin * bw))
    y2 = min(h, int(y2 + margin * bh))
    
    if x2 <= x1 or y2 <= y1:
        crop = center_crop_square(frame_bgr, size=out_size)
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    crop = frame_bgr[y1:y2, x1:x2]
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def extract_rppg_features(frames, fs):
    """Extract rPPG features from video frames.
    """
    if len(frames) < 8:
        return np.zeros((16,), dtype=np.float32)

    # Ensure minimum sampling rate for valid filtering
    fs = max(fs, 5.0)
    
    bbox = detect_face_cascade(frames[0])

    rgb_means = []
    for fr in frames:
        if bbox is None:
            roi_rgb = cv2.cvtColor(center_crop_square(fr, size=96), cv2.COLOR_BGR2RGB)
        else:
            roi_rgb = crop_bbox_rgb(fr, bbox, out_size=96, margin=0.2)
        rgb_means.append(roi_rgb.reshape(-1, 3).mean(axis=0))

    rgb_means = np.stack(rgb_means, axis=0).astype(np.float32)
    sig = pos_rppg_from_rgb_means(rgb_means, fs=fs)
    feats = rppg_feature_vector(sig, fs=fs)
    return feats


def prepare_frames_tensor(frames, bbox):
    """Prepare frames tensor for model input."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    processed = []
    for fr in frames:
        if bbox is None:
            face_rgb = cv2.cvtColor(center_crop_square(fr, size=IMG_SIZE), cv2.COLOR_BGR2RGB)
        else:
            face_rgb = crop_bbox_rgb(fr, bbox, out_size=IMG_SIZE, margin=0.2)
        
        out = transform(image=face_rgb)
        processed.append(out["image"])
    
    while len(processed) < FRAMES_PER_VIDEO:
        processed.append(processed[len(processed) % len(processed)])
    
    return torch.stack(processed[:FRAMES_PER_VIDEO], dim=0)


# Model Loading
model = None

def load_model():
    global model
    if model is not None:
        return model
    
    print(f"Loading model from {CHECKPOINT_PATH}...")
    print(f"Using device: {DEVICE}")
    
    model = FusionDetector(backbone_name="tf_efficientnet_b2_ns", rppg_dim=16, dropout=0.4)
    
    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint (epoch {ckpt.get('epoch', 'N/A')}, val_auc={ckpt.get('best_auc', 'N/A')})")
    else:
        print(f"Warning: Checkpoint not found at {CHECKPOINT_PATH}")
    
    model = model.to(DEVICE)
    model.eval()
    return model


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Inference
@torch.no_grad()
def analyze_video(video_path: str, num_passes: int = 5):
    """Analyze video with TTA (Test-Time Augmentation)."""
    model = load_model()
    
    frames, fps = read_video_frames(video_path, max_frames=RPPG_CLIP_LEN, stride=RPPG_STRIDE)
    
    if len(frames) < 4:
        return {"error": "Could not read enough frames from video"}
    
    # Ensure reasonable FPS (fallback to 25 if invalid)
    if fps <= 0 or fps > 120:
        fps = 25.0
    
    # Extract rPPG features with safe fps
    effective_fps = fps / max(1, RPPG_STRIDE)
    rppg_feat = extract_rppg_features(frames, effective_fps)
    rppg_tensor = torch.from_numpy(rppg_feat).float().to(DEVICE).unsqueeze(0)
    
    # Detect face once
    bbox = detect_face_cascade(frames[0])
    
    # Multiple passes with different frame sampling
    probabilities = []
    
    for t in range(num_passes):
        rng = np.random.RandomState(1234 + t)
        idx = rng.choice(len(frames), size=min(FRAMES_PER_VIDEO, len(frames)), replace=len(frames) < FRAMES_PER_VIDEO)
        selected_frames = [frames[i] for i in idx]
        
        # Prepare tensor
        frames_tensor = prepare_frames_tensor(selected_frames, bbox)
        frames_tensor = frames_tensor.unsqueeze(0).to(DEVICE)
        
        # Inference
        with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            logit = model(frames_tensor, rppg_tensor)
            prob = torch.sigmoid(logit).item()
        
        probabilities.append(prob)
    
    # Average probability
    avg_prob = float(np.mean(probabilities))
    std_prob = float(np.std(probabilities))
    
    # Classification (threshold tuned on validation set)
    threshold = 0.4039
    is_fake = avg_prob >= threshold
    
    # Trust score (based on confidence and consistency)
    confidence = abs(avg_prob - 0.5) * 2  # 0-1 scale
    consistency = max(0, 1 - std_prob * 5)  # Higher std = lower consistency
    trust_score = (confidence * 0.7 + consistency * 0.3) * 100
    
    return {
        "prediction": "FAKE" if is_fake else "REAL",
        "fake_probability": round(avg_prob * 100, 2),
        "real_probability": round((1 - avg_prob) * 100, 2),
        "trust_score": round(trust_score, 1),
        "confidence": round(confidence * 100, 1),
        "consistency": round(consistency * 100, 1),
        "num_frames_analyzed": len(frames),
        "fps": round(fps, 1),
        "warning": None  # Will be set if applicable
    }


# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
    
    # Save file temporarily
    filename = secure_filename(file.filename)
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        file.save(temp_path)
        result = analyze_video(temp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/model-info')
def model_info():
    return jsonify({
        "name": "Visual + rPPG Fusion Detector",
        "backbone": "EfficientNet-B2 (NS)",
        "rppg_method": "POS (Plane Orthogonal to Skin)",
        "training_data": "Celeb-DF v2",
        "validation_auc": 0.9958,
        "test_auc": 0.9708,
        "test_accuracy": 0.9054,
        "device": str(DEVICE),
        "checkpoint_loaded": CHECKPOINT_PATH.exists(),
        "limitations": [
            "Trained on face-swap deepfakes (Celeb-DF v2)",
            "May not detect AI-generated videos (Gemini, Sora, etc.)",
            "Best for detecting face manipulation, not fully synthetic content",
            "rPPG features assume natural video with physiological signals"
        ]
    })


if __name__ == '__main__':
    # Pre-load model
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
