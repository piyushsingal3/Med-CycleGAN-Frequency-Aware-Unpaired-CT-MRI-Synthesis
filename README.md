# CycleGAN — MRI ↔ CT Medical Image Translation

---

## Project Overview

This project implements the CycleGAN architecture (from your documentation) and applies it to translate:
- **Domain A = MRI scans** → **Domain B = CT scans** (and back)

The key insight from the documentation:
> *"CycleGANs address the limitation [of paired data] by using unpaired image data, making them extremely versatile"*

MRI↔CT is a perfect real-world application because **perfectly registered MRI-CT pairs of the same patient are nearly impossible to obtain** — exactly the problem CycleGAN was designed to solve.

---

## Architecture (from your documentation)

### Generator: ResNet-based
```
c7s1-64 → d128 → d256 → [R256 × 9] → u128 → u64 → c7s1-1
  ↑ Encoder ↑           ↑ Transformer ↑    ↑ Decoder ↑
```
- **c7s1-k**: 7×7 Conv-InstanceNorm-ReLU, k filters, stride 1
- **dk**: 3×3 Conv-InstanceNorm-ReLU, k filters, stride 2 (downsample)
- **Rk**: Residual block, two 3×3 convolutions
- **uk**: 3×3 fractional-stride Conv-InstanceNorm-ReLU (upsample)

### Discriminator: PatchGAN (70×70)
```
C64 → C128 → C256 → C512 → Conv(1)
```
- **Ck**: 4×4 Conv-InstanceNorm-LeakyReLU, stride 2
- No InstanceNorm on C64 (first layer)
- Output: probability map (each patch = real or fake)

### Loss Functions (from documentation)
```
L_total = L_GAN(G, Dy) + L_GAN(F, Dx) + λ·L_cyc(G,F) + 0.5λ·L_identity(G,F)
```
- **Adversarial loss** (LSGAN variant for stability)
- **Cycle consistency loss**: F(G(x)) ≈ x  and  G(F(y)) ≈ y
- **Identity loss**: G(y) ≈ y  and  F(x) ≈ x

---

## Project Structure

```
cyclegan_mri_ct/
│
├── train.py          ← Main training script
├── test.py           ← Run inference on new images
├── evaluate.py       ← Compute MAE / PSNR / SSIM / FID
├── prepare_data.py   ← Download & organize datasets
├── models.py         ← Generator + Discriminator architectures
├── dataset.py        ← PyTorch Dataset (MRI/CT/DICOM support)
├── utils.py          ← ReplayBuffer, LR scheduler, checkpoints
├── requirements.txt  ← Python dependencies
│
├── data/             
│   ├── trainA/       ← MRI training images
│   ├── trainB/       ← CT  training images
│   ├── testA/        ← MRI test images
│   └── testB/        ← CT  test images
│
├── checkpoints/      ← Saved model checkpoints
└── outputs/          ← Generated images & test results
    └── samples/      ← Training samples saved every N epochs
```

---

## Complete Roadmap to Run the Project

### STEP 0 — Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify GPU (strongly recommended — CPU training is ~50x slower)
python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

### STEP 1 — Prepare Data

#### Option A: Quick Test with Dummy Data (recommended first)
```bash
python prepare_data.py --mode dummy --data_root data/ --n_train 200 --n_test 50
```
This creates synthetic MRI-like and CT-like images so you can verify the full pipeline runs before downloading real data.

#### Option B: Real Medical Data from Kaggle (recommended for actual project)

**Dataset 1: IXI Brain MRI Dataset**
1. Go to: https://brain-development.org/ixi-dataset/
2. Download `IXI-T1.tar` (T1-weighted MRI, ~4.5 GB)
3. Extract and run:
```bash
python prepare_data.py --mode organize \
    --mri_src /path/to/IXI-T1/ \
    --ct_src  /path/to/ct_scans/ \
    --data_root data/
```

**Dataset 2: Kaggle Gold Standard** (easiest)
```bash
# First setup Kaggle API key:
# 1. https://www.kaggle.com → Account → Create API Token
# 2. Place kaggle.json in ~/.kaggle/

python prepare_data.py --mode download --data_root data/
```

**Recommended Kaggle datasets:**
| Dataset | Link | Domain |
|---------|------|--------|
| Brain MRI | kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset | MRI (trainA) |
| RSNA CT Scans | kaggle.com/datasets/kmader/siim-medical-images | CT (trainB) |
| MRI+CT combined | kaggle.com/datasets/arjunbasandrai/medical-imaging-datasets | Both |

#### Verify your dataset:
```bash
python prepare_data.py --mode verify --data_root data/
```
Expected output:
```
✓ trainA    : 1000+ images  (MRI)
✓ trainB    : 1000+ images  (CT)
✓ testA     : 100+ images
✓ testB     : 100+ images
```

---

### STEP 2 — Verify Architecture

```bash
python models.py
```
Expected output:
```
Generator input:  torch.Size([1, 1, 256, 256])
Generator output: torch.Size([1, 1, 256, 256])
Discriminator patch output: torch.Size([1, 1, 30, 30])
Generator params:     ~11.37M
Discriminator params:  ~2.76M
```

---

### STEP 3 — Train the Model

#### Quick test run (dummy data, CPU):
```bash
python train.py \
    --data_root data/ \
    --n_epochs 5 \
    --batch_size 1 \
    --img_size 128 \
    --n_res_blocks 6 \
    --save_every 2 \
    --sample_every 1
```

#### Full training run (GPU, 256×256):
```bash
python train.py \
    --data_root    data/ \
    --output_dir   outputs/ \
    --n_epochs     200 \
    --decay_epoch  100 \
    --batch_size   1 \
    --lr           0.0002 \
    --img_size     256 \
    --n_res_blocks 9 \
    --lambda_cyc   10.0 \
    --lambda_id    5.0 \
    --save_every   10 \
    --sample_every 5 \
    --n_workers    4
```

#### Resume training:
```bash
python train.py [same args] --resume
```

**Expected training time:**
| Hardware | ~Time per epoch (1000 images) |
|----------|-------------------------------|
| RTX 3080 / 4080 | ~3–5 min |
| T4 (Colab free) | ~8–12 min |
| CPU only | ~3–5 hours |

---

### STEP 4 — Monitor Training

Training samples are saved every 5 epochs to `outputs/samples/`.

Each sample image shows a 3×2 grid:
```
[ Real MRI ] → [ Fake CT  ] → [ Reconstructed MRI ]
[ Real CT  ] → [ Fake MRI ] → [ Reconstructed CT  ]
```

Watch for:
- **Good sign**: Fake CT develops bone-like bright regions; tissue contrast similar to real CT
- **Mode collapse**: All outputs look identical — reduce LR or restart
- **Checkerboard artifacts**: Increase lambda_id or switch to bilinear upsample (already done in this code)

---

### STEP 5 — Run Inference

```bash
# Translate MRI → CT
python test.py \
    --checkpoint checkpoints/checkpoint_epoch_200.pth \
    --data_root  data/ \
    --direction  AtoB \
    --output_dir outputs/test_results/

# Translate CT → MRI
python test.py \
    --checkpoint checkpoints/checkpoint_epoch_200.pth \
    --data_root  data/ \
    --direction  BtoA \
    --output_dir outputs/test_results/
```

---

### STEP 6 — Evaluate

```bash
# Pixel-level metrics (MAE, PSNR, SSIM)
python evaluate.py \
    --real_dir data/testB/ \
    --fake_dir outputs/test_results/fake_CT/

# Include FID (install: pip install pytorch-fid)
python evaluate.py \
    --real_dir data/testB/ \
    --fake_dir outputs/test_results/fake_CT/ \
    --fid
```

**Target metrics for a well-trained model:**
| Metric | Target |
|--------|--------|
| MAE    | < 0.08 |
| PSNR   | > 25 dB |
| SSIM   | > 0.80 |
| FID    | < 60 |

---

## Training Tips

1. **Start small**: Use 128×128, 6 res blocks, 50 epochs to validate the pipeline
2. **Lambda values**: λ_cyc=10, λ_id=5 are the original paper defaults — works well for medical images
3. **Learning rate**: 0.0002 with Adam (β1=0.5, β2=0.999) — from the original CycleGAN paper
4. **Data size**: Aim for 500+ images per domain; 1000+ is better
5. **Google Colab**: Use a T4/A100 GPU runtime — free T4 can train ~100 epochs overnight
6. **DICOM files**: Install `pydicom` if working with real hospital DICOM data

---

## Google Colab Quick Start

```python
# In a Colab cell:
!git clone <your-repo-or-upload-files>
%cd cyclegan_mri_ct
!pip install -r requirements.txt

# Create dummy data to test pipeline
!python prepare_data.py --mode dummy --n_train 300

# Train
!python train.py --n_epochs 50 --sample_every 5 --save_every 10

# View samples
from IPython.display import Image
Image('outputs/samples/epoch_050.png')
```

---

## Key Differences from Standard CycleGAN

| Standard CycleGAN | This Implementation |
|-------------------|---------------------|
| RGB (3-channel) | **Grayscale (1-channel)** for medical images |
| Natural images | **Medical images** (MRI/CT/DICOM) |
| 3 output channels | **1 output channel** |
| BatchNorm | **InstanceNorm** (better for single-image style) |

---

*Original paper: Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks", ICCV 2017*
