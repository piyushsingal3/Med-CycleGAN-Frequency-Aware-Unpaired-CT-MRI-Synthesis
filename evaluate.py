"""
evaluate.py  —  Evaluate CycleGAN translation quality

Metrics:
  • MAE  (Mean Absolute Error)    — pixel-level accuracy
  • SSIM (Structural Similarity)  — perceptual quality
  • PSNR (Peak Signal-to-Noise)   — signal quality
  • FID  (Fréchet Inception Distance) — distribution-level realism
               (requires: pip install pytorch-fid)

Usage (paired test set):
    python evaluate.py --real_dir data/testB/ --fake_dir outputs/test_results/fake_CT/

Usage (FID only):
    python evaluate.py --real_dir data/testB/ --fake_dir outputs/test_results/fake_CT/ --fid
"""

import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms


def load_grayscale(path: str, size: int = 256) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size))
    return np.array(img, dtype=np.float32) / 255.0


def mae(real: np.ndarray, fake: np.ndarray) -> float:
    return np.mean(np.abs(real - fake))


def psnr(real: np.ndarray, fake: np.ndarray) -> float:
    mse_val = np.mean((real - fake) ** 2)
    if mse_val == 0:
        return float("inf")
    return 20 * np.log10(1.0 / np.sqrt(mse_val))


def ssim(real: np.ndarray, fake: np.ndarray, C1=0.01**2, C2=0.03**2) -> float:
    mu1, mu2     = real.mean(), fake.mean()
    sigma1       = real.std()
    sigma2       = fake.std()
    sigma12      = np.mean((real - mu1) * (fake - mu2))
    ssim_val     = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1**2 + mu2**2 + C1) * (sigma1**2 + sigma2**2 + C2))
    return ssim_val


def evaluate_paired(real_dir: str, fake_dir: str, img_size: int = 256):
    """Compute MAE/PSNR/SSIM on paired real vs. fake images."""
    real_files = sorted([f for f in os.listdir(real_dir) if f.endswith(".png")])
    fake_files = sorted([f for f in os.listdir(fake_dir) if f.endswith(".png")])

    common = set(os.path.basename(f) for f in real_files) & \
             set(os.path.basename(f) for f in fake_files)

    if len(common) == 0:
        print(f"[WARNING] No matching filenames between {real_dir} and {fake_dir}")
        print("  Falling back to index-based pairing...")
        n = min(len(real_files), len(fake_files))
        pairs = [(os.path.join(real_dir, real_files[i]),
                  os.path.join(fake_dir, fake_files[i])) for i in range(n)]
    else:
        common = sorted(common)
        pairs  = [(os.path.join(real_dir, f), os.path.join(fake_dir, f))
                  for f in common]

    print(f"Evaluating {len(pairs)} image pairs...")

    mae_scores, psnr_scores, ssim_scores = [], [], []

    for real_path, fake_path in pairs:
        real = load_grayscale(real_path, img_size)
        fake = load_grayscale(fake_path, img_size)

        mae_scores.append(mae(real, fake))
        psnr_scores.append(psnr(real, fake))
        ssim_scores.append(ssim(real, fake))

    print("\n" + "─" * 40)
    print("Evaluation Results (Paired)")
    print("─" * 40)
    print(f"  MAE  : {np.mean(mae_scores):.4f}  ± {np.std(mae_scores):.4f}")
    print(f"  PSNR : {np.mean(psnr_scores):.2f} dB ± {np.std(psnr_scores):.2f}")
    print(f"  SSIM : {np.mean(ssim_scores):.4f}  ± {np.std(ssim_scores):.4f}")
    print("─" * 40)

    return {
        "mae":  np.mean(mae_scores),
        "psnr": np.mean(psnr_scores),
        "ssim": np.mean(ssim_scores),
    }


def evaluate_fid(real_dir: str, fake_dir: str):
    """Compute FID score (requires pytorch-fid package)."""
    try:
        from pytorch_fid import fid_score
    except ImportError:
        print("[FID] pytorch-fid not installed. Run: pip install pytorch-fid")
        return None

    print(f"\nComputing FID: {real_dir} vs {fake_dir}")
    fid = fid_score.calculate_fid_given_paths(
        [real_dir, fake_dir],
        batch_size=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dims=2048
    )
    print(f"  FID Score: {fid:.2f}  (lower = better, 0 = identical distributions)")
    return fid


def main():
    parser = argparse.ArgumentParser(description="Evaluate CycleGAN translation quality")
    parser.add_argument("--real_dir",  type=str, required=True, help="Real CT images folder")
    parser.add_argument("--fake_dir",  type=str, required=True, help="Fake CT images folder")
    parser.add_argument("--img_size",  type=int, default=256)
    parser.add_argument("--fid",       action="store_true",     help="Also compute FID score")
    opt = parser.parse_args()

    results = evaluate_paired(opt.real_dir, opt.fake_dir, opt.img_size)

    if opt.fid:
        fid = evaluate_fid(opt.real_dir, opt.fake_dir)
        if fid is not None:
            results["fid"] = fid

    print("\nInterpretation guide:")
    print("  MAE  < 0.05  → excellent  |  < 0.10  → good  |  > 0.15  → poor")
    print("  PSNR > 30 dB → excellent  |  > 25 dB → good")
    print("  SSIM > 0.90  → excellent  |  > 0.80  → good")
    print("  FID  < 50    → good       |  < 20    → excellent")


if __name__ == "__main__":
    main()
