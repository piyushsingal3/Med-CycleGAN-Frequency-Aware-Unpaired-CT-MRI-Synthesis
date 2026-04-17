"""
prepare_data.py  —  Download & organize medical imaging datasets for CycleGAN

Supported datasets (all free on Kaggle):
  1. IXI Dataset           — MRI brain images
  2. TCIA CT Scans         — CT brain slices
  3. Brain MRI + CT (combined Kaggle dataset) — easiest to start

How to use:
  Option A (Recommended — download manually):
    1. Go to Kaggle links printed below
    2. Download the zip
    3. Run: python prepare_data.py --mode organize --src /path/to/downloaded/zip

  Option B (Kaggle API — if configured):
    python prepare_data.py --mode download

Run this after:
    python prepare_data.py --mode verify
"""

import os
import shutil
import random
import argparse
import zipfile
import glob
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Dataset info
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = {
    "brain_mri_ct": {
        "kaggle_id"  : "andrewmvd/brain-tumor-mri-dataset",
        "description": "Brain MRI dataset — use as Domain A (MRI)",
        "url"        : "https://www.kaggle.com/datasets/andrewmvd/brain-tumor-mri-dataset"
    },
    "ct_scans": {
        "kaggle_id"  : "kmader/siim-medical-images",
        "description": "CT scans — use as Domain B (CT)",
        "url"        : "https://www.kaggle.com/datasets/kmader/siim-medical-images"
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def print_separator():
    print("─" * 60)


def download_via_kaggle_api(dataset_id: str, output_dir: str):
    """Download dataset using Kaggle API (requires ~/.kaggle/kaggle.json)."""
    import subprocess
    os.makedirs(output_dir, exist_ok=True)
    cmd = f"kaggle datasets download -d {dataset_id} -p {output_dir} --unzip"
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {result.stderr}")
        print("\nSetup Kaggle API:")
        print("  1. Go to https://www.kaggle.com → Account → Create API Token")
        print("  2. Place kaggle.json in ~/.kaggle/")
        print("  3. chmod 600 ~/.kaggle/kaggle.json")
    else:
        print(result.stdout)


def extract_zip(zip_path: str, output_dir: str):
    """Extract a zip file."""
    print(f"Extracting {zip_path} → {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    print("✓ Extraction complete")


def collect_images(src_dir: str, extensions=(".png", ".jpg", ".jpeg", ".dcm")):
    """Recursively collect all image files."""
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(src_dir, "**", f"*{ext}"), recursive=True))
        images.extend(glob.glob(os.path.join(src_dir, "**", f"*{ext.upper()}"), recursive=True))
    return sorted(set(images))


def normalize_and_save(src_path: str, dst_path: str, size: int = 256):
    """Load, normalize, resize, and save as grayscale PNG."""
    ext = os.path.splitext(src_path)[1].lower()

    if ext == ".dcm":
        try:
            import pydicom
            ds    = pydicom.dcmread(src_path)
            array = ds.pixel_array.astype(np.float32)
            a_min, a_max = array.min(), array.max()
            if a_max > a_min:
                array = (array - a_min) / (a_max - a_min) * 255.0
            img = Image.fromarray(array.astype(np.uint8)).convert("L")
        except Exception as e:
            print(f"  [skip DICOM] {src_path}: {e}")
            return False
    else:
        try:
            img = Image.open(src_path).convert("L")
        except Exception as e:
            print(f"  [skip] {src_path}: {e}")
            return False

    img = img.resize((size, size), Image.LANCZOS)
    img.save(dst_path)
    return True


def split_and_organize(images: list, output_dir: str, domain: str,
                        train_ratio: float = 0.85, img_size: int = 256):
    """
    Split images into train/test and copy into:
        output_dir/trainA/ (or trainB/)
        output_dir/testA/  (or testB/)
    """
    random.shuffle(images)
    n_train   = int(len(images) * train_ratio)
    train_set = images[:n_train]
    test_set  = images[n_train:]

    train_dir = os.path.join(output_dir, f"train{domain}")
    test_dir  = os.path.join(output_dir, f"test{domain}")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    def process_set(image_list, dst_dir, label):
        count = 0
        for i, src in enumerate(image_list):
            fname = f"{i:05d}.png"
            dst   = os.path.join(dst_dir, fname)
            if normalize_and_save(src, dst, size=img_size):
                count += 1
            if (i + 1) % 100 == 0:
                print(f"  {label}: {i+1}/{len(image_list)}")
        return count

    print(f"\nProcessing Domain {domain} — Train set ({len(train_set)} images):")
    n_train_ok = process_set(train_set, train_dir, "Train")

    print(f"\nProcessing Domain {domain} — Test set ({len(test_set)} images):")
    n_test_ok  = process_set(test_set,  test_dir,  "Test")

    print(f"\n✓ Domain {domain}: {n_train_ok} train / {n_test_ok} test saved")
    return n_train_ok, n_test_ok


def create_dummy_dataset(output_dir: str, n_train: int = 100, n_test: int = 20,
                          img_size: int = 256):
    """
    Create a dummy dataset with synthetic MRI-like and CT-like images.
    Useful for testing the pipeline before downloading real data.
    """
    print(f"\n🔧 Creating dummy dataset in: {output_dir}")
    print(f"   {n_train} train + {n_test} test images per domain\n")

    for split in ["trainA", "trainB", "testA", "testB"]:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    n_per_domain = {"trainA": n_train, "trainB": n_train,
                    "testA":  n_test,  "testB":  n_test}

    for split, n in n_per_domain.items():
        is_mri = split.endswith("A")
        for i in range(n):
            # MRI-like: smooth Gaussian blobs
            # CT-like: sharp edges with bone-like regions
            if is_mri:
                arr = np.random.normal(128, 40, (img_size, img_size)).clip(0, 255)
            else:
                arr = np.random.uniform(0, 255, (img_size, img_size))
                # Add bright "bone" regions
                cx, cy = img_size // 2, img_size // 2
                rr, cc = np.ogrid[:img_size, :img_size]
                mask = (rr - cx)**2 + (cc - cy)**2 < (img_size // 4)**2
                arr[mask] = np.random.uniform(180, 255)

            img = Image.fromarray(arr.astype(np.uint8), mode="L")
            img.save(os.path.join(output_dir, split, f"{i:05d}.png"))

        print(f"  ✓ {split}: {n} synthetic images")

    print("\n✅ Dummy dataset ready! Use --data_root", output_dir, "to train.")


def verify_dataset(data_root: str):
    """Check that the dataset is properly organized."""
    print_separator()
    print("Verifying dataset structure...")
    print_separator()

    required = ["trainA", "trainB", "testA", "testB"]
    all_ok   = True

    for split in required:
        path = os.path.join(data_root, split)
        if not os.path.exists(path):
            print(f"  ✗ MISSING: {path}")
            all_ok = False
        else:
            files = [f for f in os.listdir(path) if f.endswith(".png")]
            print(f"  ✓ {split:10s}: {len(files)} images")

    if all_ok:
        print("\n✅ Dataset structure OK!")
    else:
        print("\n❌ Dataset incomplete. Run prepare_data.py --mode organize first.")

    print_separator()
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare MRI/CT dataset for CycleGAN")
    parser.add_argument("--mode",      type=str, default="dummy",
                        choices=["dummy", "download", "organize", "verify"],
                        help="dummy=create fake data | download=Kaggle API | organize=from zip | verify=check structure")
    parser.add_argument("--data_root", type=str, default="data/",   help="Output data directory")
    parser.add_argument("--mri_src",   type=str, default="",        help="[organize mode] Path to MRI images/zip")
    parser.add_argument("--ct_src",    type=str, default="",        help="[organize mode] Path to CT images/zip")
    parser.add_argument("--img_size",  type=int, default=256)
    parser.add_argument("--n_train",   type=int, default=200,       help="[dummy] Images per domain per split")
    parser.add_argument("--n_test",    type=int, default=50)
    opt = parser.parse_args()

    print_separator()
    print("CycleGAN MRI↔CT — Data Preparation")
    print_separator()

    if opt.mode == "dummy":
        create_dummy_dataset(opt.data_root, opt.n_train, opt.n_test, opt.img_size)

    elif opt.mode == "download":
        print("\nAvailable datasets:")
        for name, info in DATASETS.items():
            print(f"\n  {name}:")
            print(f"    {info['description']}")
            print(f"    URL: {info['url']}")

        raw_dir = os.path.join(opt.data_root, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        print("\nDownloading MRI dataset...")
        download_via_kaggle_api(DATASETS["brain_mri_ct"]["kaggle_id"],
                                os.path.join(raw_dir, "mri"))

        print("\nDownloading CT dataset...")
        download_via_kaggle_api(DATASETS["ct_scans"]["kaggle_id"],
                                os.path.join(raw_dir, "ct"))

        # Organize downloaded files
        mri_images = collect_images(os.path.join(raw_dir, "mri"))
        ct_images  = collect_images(os.path.join(raw_dir, "ct"))

        print(f"\nFound {len(mri_images)} MRI and {len(ct_images)} CT images")
        split_and_organize(mri_images, opt.data_root, "A", img_size=opt.img_size)
        split_and_organize(ct_images,  opt.data_root, "B", img_size=opt.img_size)

    elif opt.mode == "organize":
        if not opt.mri_src or not opt.ct_src:
            print("ERROR: --mri_src and --ct_src are required for organize mode")
            return

        # Handle zip files
        mri_dir = opt.mri_src
        ct_dir  = opt.ct_src

        if opt.mri_src.endswith(".zip"):
            mri_dir = os.path.join(opt.data_root, "raw_mri")
            extract_zip(opt.mri_src, mri_dir)

        if opt.ct_src.endswith(".zip"):
            ct_dir = os.path.join(opt.data_root, "raw_ct")
            extract_zip(opt.ct_src, ct_dir)

        mri_images = collect_images(mri_dir)
        ct_images  = collect_images(ct_dir)

        print(f"\nFound {len(mri_images)} MRI and {len(ct_images)} CT images")
        split_and_organize(mri_images, opt.data_root, "A", img_size=opt.img_size)
        split_and_organize(ct_images,  opt.data_root, "B", img_size=opt.img_size)

    elif opt.mode == "verify":
        verify_dataset(opt.data_root)

    print_separator()


if __name__ == "__main__":
    main()
