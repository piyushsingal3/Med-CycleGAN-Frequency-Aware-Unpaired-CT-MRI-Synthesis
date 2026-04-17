"""
dataset.py  —  Medical Image Dataset for CycleGAN (MRI ↔ CT)

Folder structure expected:
    data/
      trainA/   ← MRI training images  (.png / .jpg / .dcm)
      trainB/   ← CT  training images
      testA/    ← MRI test images
      testB/    ← CT  test images

Supports:
  • Standard images (.png, .jpg, .bmp)
  • DICOM files (.dcm) via pydicom (install separately)
  • Grayscale (1-channel) — appropriate for medical imaging
"""

import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Try to import pydicom for DICOM support
try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    print("[dataset.py] pydicom not found — DICOM support disabled. Install with: pip install pydicom")


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
DICOM_EXTENSIONS     = {".dcm", ".dicom"}


def load_image(path: str) -> Image.Image:
    """Load any supported image as a grayscale PIL Image."""
    ext = os.path.splitext(path)[1].lower()

    if ext in DICOM_EXTENSIONS:
        if not DICOM_AVAILABLE:
            raise ImportError("pydicom required for DICOM files. Install: pip install pydicom")
        ds    = pydicom.dcmread(path)
        array = ds.pixel_array.astype(np.float32)

        # Normalize to [0, 255]
        arr_min, arr_max = array.min(), array.max()
        if arr_max > arr_min:
            array = (array - arr_min) / (arr_max - arr_min) * 255.0
        else:
            array = np.zeros_like(array)

        return Image.fromarray(array.astype(np.uint8)).convert("L")

    elif ext in SUPPORTED_EXTENSIONS:
        return Image.open(path).convert("L")   # Grayscale

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def list_images(folder: str) -> list:
    """Return sorted list of all image paths in a folder."""
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    paths = []
    for fname in sorted(os.listdir(folder)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in SUPPORTED_EXTENSIONS | DICOM_EXTENSIONS:
            paths.append(os.path.join(folder, fname))

    if len(paths) == 0:
        raise ValueError(f"No images found in {folder}. "
                         f"Supported: {SUPPORTED_EXTENSIONS | DICOM_EXTENSIONS}")
    return paths


class MedicalImageDataset(Dataset):
    """
    Unpaired MRI ↔ CT dataset for CycleGAN.
    Randomly samples from each domain independently (no pairing required).
    """
    def __init__(self, root: str, transform=None, mode: str = "train"):
        """
        Args:
            root      : Path to dataset root (contains trainA, trainB, testA, testB)
            transform : torchvision transforms to apply
            mode      : 'train' or 'test'
        """
        self.transform = transform
        self.mode      = mode

        # Domain A = MRI, Domain B = CT
        folder_A = os.path.join(root, f"{mode}A")
        folder_B = os.path.join(root, f"{mode}B")

        self.files_A = list_images(folder_A)
        self.files_B = list_images(folder_B)

        print(f"[{mode}] Domain A (MRI): {len(self.files_A)} images")
        print(f"[{mode}] Domain B (CT):  {len(self.files_B)} images")

    def __len__(self):
        # Use the larger domain size (unpaired; B is sampled randomly)
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, idx):
        # Domain A (MRI): sequential
        path_A = self.files_A[idx % len(self.files_A)]

        # Domain B (CT): random (unpaired setting)
        path_B = self.files_B[random.randint(0, len(self.files_B) - 1)]

        img_A = load_image(path_A)
        img_B = load_image(path_B)

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B, "path_A": path_A, "path_B": path_B}


class PairedMedicalDataset(Dataset):
    """
    OPTIONAL: Paired dataset for evaluation/comparison.
    Assumes files in testA/ and testB/ are ordered identically (same patient slices).
    """
    def __init__(self, root: str, transform=None, mode: str = "test"):
        self.transform = transform

        folder_A = os.path.join(root, f"{mode}A")
        folder_B = os.path.join(root, f"{mode}B")

        self.files_A = list_images(folder_A)
        self.files_B = list_images(folder_B)

        if len(self.files_A) != len(self.files_B):
            print(f"[WARNING] Paired dataset has mismatched sizes: "
                  f"A={len(self.files_A)}, B={len(self.files_B)}")

        self.length = min(len(self.files_A), len(self.files_B))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_A = load_image(self.files_A[idx])
        img_B = load_image(self.files_B[idx])

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return {
            "A": img_A, "B": img_B,
            "path_A": self.files_A[idx],
            "path_B": self.files_B[idx]
        }


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Test with dummy data — create temp folders
    import tempfile, shutil
    tmpdir = tempfile.mkdtemp()
    for split in ["trainA", "trainB"]:
        os.makedirs(os.path.join(tmpdir, split))
        for i in range(5):
            img = Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8))
            img.save(os.path.join(tmpdir, split, f"{i:03d}.png"))

    dataset = MedicalImageDataset(tmpdir, transform=transform, mode="train")
    sample  = dataset[0]
    print(f"\nSample A shape: {sample['A'].shape}")
    print(f"Sample B shape: {sample['B'].shape}")
    print(f"Min/Max A: {sample['A'].min():.2f} / {sample['A'].max():.2f}")
    shutil.rmtree(tmpdir)
    print("\n✅ Dataset test passed!")
