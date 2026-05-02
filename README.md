# Multimodal Deepfake Detection

## What this project does and why it matters

This project builds a system that watches a video of a person's face and decides whether it is real or computer-generated (a "deepfake"). Deepfakes are AI-generated videos that can make anyone appear to say or do something they never did — they are increasingly used to spread misinformation, commit fraud, and damage reputations. Most existing detection tools look at only one type of visual evidence and are easy to fool when tested on videos they were not trained on. This project uses three completely different types of evidence simultaneously — what the face looks like, invisible statistical patterns hidden in the image frequencies, and how the face moves across time — trains separate expert models on each, and then compares multiple strategies for combining them. A key goal is measuring whether the combined detector holds up on datasets it was never shown during training, which is the real-world test that most published tools fail.

---

## The three input signals

### 1. Spatial (RGB frames)
This is the "what does the face look like" signal. The model is shown individual face images and learns to spot visual inconsistencies that deepfake generators leave behind — things like unnatural skin texture, lighting that does not match the scene, or blurring around the hairline and ears. Humans can sometimes spot these artifacts too, but they are subtle and easy to miss at normal video speed.

### 2. Frequency (FFT spectrum)
This is the "hidden pattern" signal. Every digital image has a frequency fingerprint — a mathematical signature describing how much fine detail versus broad color is present. When AI generators synthesize a face they produce characteristic blotchy patterns in the frequency domain (GAN grid artifacts, diffusion model fingerprints) that look very different from the clean, natural patterns in a real photograph. This analysis is done using the Fast Fourier Transform (FFT) and reveals evidence that is completely invisible when looking at the image normally.

### 3. Temporal (frame sequences)
This is the "how does it move" signal. Deepfake generators are trained primarily on still images and struggle to produce perfectly smooth, natural motion across time. Small jitters, inconsistent blinks, and unnatural head movements accumulate over a sequence of frames. By feeding the model a clip of 16 consecutive frames rather than a single image, we give it the chance to catch these temporal inconsistencies that no single frame would reveal.

---

## Datasets

| Dataset | Real Videos | Fake Videos | Role | Size |
|---|---|---|---|---|
| FaceForensics++ (c23) | 1,000 | 4,000 | **Training + primary eval only** | ~50 GB |
| Celeb-DF v2 | 590 | 5,639 | Cross-dataset test 1 (no training) | ~2 GB |
| DFDC Preview | ~1,000 | ~4,000 | Cross-dataset test 2 (no training) | ~10 GB |

**Important:** only FF++ is used for training. Celeb-DF v2 and DFDC Preview are held out entirely and used exclusively for cross-dataset generalization evaluation — the model never sees them during training. This tests whether the detector actually learned to recognize deepfakes in general rather than memorizing the specific artifacts of one dataset.

**FF++** contains four manipulation methods: DeepFakes, Face2Face, FaceSwap, and NeuralTextures. Using the compressed (c23) version matches the standard experimental setup in published research and allows direct comparison with Xception and EfficientNet baselines.

**Celeb-DF v2** uses a higher-quality synthesis pipeline than FF++, making its fakes significantly harder to spot. AUC on Celeb-DF is the standard measure of real-world robustness in the literature.

**DFDC Preview** was collected in the wild with diverse subjects, ages, ethnicities, lighting conditions, and compression levels. It is the most realistic and challenging benchmark of the three.

---

## Model comparison strategy and ablation study

### Phase 1 — Unimodal baselines
Three independent models, each trained from scratch on one signal type:
- **Spatial CNN** — custom ResNet-style CNN on 224×224 RGB face crops
- **Frequency CNN** — same architecture, trained on FFT magnitude images
- **Temporal CNN+LSTM** — shared CNN frame encoder → bidirectional LSTM over 16-frame sequences

### Phase 2 — Fusion architectures
Three strategies for combining the baseline models:
- **Late fusion** — average the probability outputs of the three trained baselines (no additional training)
- **Concatenation fusion** — join the 256-dim feature vectors from all three branches and train an MLP classifier end-to-end
- **Cross-attention fusion** — the frequency embedding acts as a Query that attends over per-frame temporal features via multi-head attention, learning which frames contain the most frequency-relevant artifacts

### Phase 3 — Pretrained backbone comparison
Xception (the published SOTA baseline for FF++) and EfficientNetV2 are fine-tuned under identical conditions to benchmark the from-scratch models and isolate the contribution of ImageNet pretraining versus architecture design.

### Ablation study
An **ablation study** is the experiment that answers "what happens if I remove one piece?" Every combination is evaluated: spatial only, frequency only, temporal only, spatial+frequency, spatial+temporal, frequency+temporal, and all three. If removing frequency barely changes accuracy, it contributes nothing unique. If removing it causes a large drop, it is essential. This is the standard scientific method for proving that each component in a complex system is actually doing useful work.

---

## Setup

```bash
git clone <repo_url>
cd deepfake-detection

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Download datasets manually and place under:
```
data/raw/faceforensics/       # ~50 GB — request access at github.com/ondyari/FaceForensics
data/raw/celebdf/             # ~2 GB  — github.com/yuezunli/celeb-deepfakeforensics
data/raw/dfdc_preview/        # ~10 GB — Kaggle deepfake-detection-challenge
```

---

## Preprocessing pipeline

Run scripts in this order. Each script is idempotent — safe to re-run, skips files already processed.

### Step 1 — Extract frames (sample every 5th frame → ~6 fps)
```bash
python preprocessing/extract_frames.py \
    --input_dir  data/raw/faceforensics \
    --output_dir data/processed/frames \
    --frame_interval 5
```

### Step 2 — Crop faces to 224×224
```bash
python preprocessing/crop_faces.py \
    --input_dir  data/processed/frames \
    --output_dir data/processed/faces \
    --target_size 224 \
    --padding 0.2
```

### Step 3 — Compute FFT magnitude spectra
```bash
python preprocessing/compute_fft.py \
    --input_dir  data/processed/faces \
    --output_dir data/processed/fft
```

### Step 4 — Build dataset splits
FF++ → train/val/test (70/15/15 at video level). Celeb-DF and DFDC → separate eval CSVs.
```bash
python preprocessing/build_dataset.py \
    --faces_dir  data/processed/faces \
    --fft_dir    data/processed/fft \
    --output_dir data/splits
```

### Step 5 — Verify dataset quality
```bash
python preprocessing/verify_dataset.py \
    --splits_dir data/splits \
    --n_samples  5
```

---

## Results

Training on Google Colab A100. Cross-dataset columns report AUC with no fine-tuning.

### FF++ test set performance

| Model | AUC-ROC | Accuracy | Deepfakes | Face2Face | FaceSwap | NeuralTextures |
|---|---|---|---|---|---|---|
| Spatial CNN | — | — | — | — | — | — |
| Frequency CNN | — | — | — | — | — | — |
| Temporal CNN+LSTM | — | — | — | — | — | — |
| Late Fusion | — | — | — | — | — | — |
| Concat Fusion | — | — | — | — | — | — |
| Cross-Attention Fusion | — | — | — | — | — | — |
| Xception (SOTA baseline) | — | — | — | — | — | — |
| EfficientNetV2 | — | — | — | — | — | — |

### Cross-dataset generalization (AUC-ROC, no fine-tuning)

| Model | Celeb-DF v2 | DFDC Preview |
|---|---|---|
| Spatial CNN | — | — |
| Frequency CNN | — | — |
| Temporal CNN+LSTM | — | — |
| Late Fusion | — | — |
| Concat Fusion | — | — |
| Cross-Attention Fusion | — | — |
| Xception (SOTA baseline) | — | — |
| EfficientNetV2 | — | — |

---

## Tech stack

| Component | Technology |
|---|---|
| Frame extraction | OpenCV |
| Face detection | MediaPipe |
| Frequency analysis | NumPy FFT |
| Model training | PyTorch 2.x + torchvision |
| Pretrained backbones (Phase 3) | timm (Xception, EfficientNetV2) |
| Data splits | scikit-learn |
| Visualization | Matplotlib |
| Notebooks | Jupyter (Colab) |
| API (planned) | FastAPI + Uvicorn |
| Inference export (planned) | ONNX + onnxruntime |
| Training hardware | Google Colab A100 (40 GB VRAM) |
| Local machine | macOS, Apple Silicon, 16 GB RAM |
