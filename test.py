"""
test.py  —  Run inference with trained CycleGAN

Translates:
  • All MRI images in testA/ → fake CT  (saved to outputs/fake_CT/)
  • All CT  images in testB/ → fake MRI (saved to outputs/fake_MRI/)

Usage:
    python test.py --checkpoint checkpoints/checkpoint_epoch_200.pth
                   --data_root data/
                   --output_dir outputs/test_results/
                   --direction AtoB        # MRI → CT
"""

import os
import argparse
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from models import Generator
from dataset import list_images, load_image


def get_args():
    parser = argparse.ArgumentParser(description="CycleGAN Inference")
    parser.add_argument("--checkpoint",  type=str, required=True,   help="Path to .pth checkpoint file")
    parser.add_argument("--data_root",   type=str, default="data/", help="Root folder with testA/ testB/")
    parser.add_argument("--output_dir",  type=str, default="outputs/test_results/")
    parser.add_argument("--direction",   type=str, default="AtoB",  choices=["AtoB", "BtoA"],
                        help="AtoB = MRI→CT,  BtoA = CT→MRI")
    parser.add_argument("--img_size",    type=int, default=256)
    parser.add_argument("--n_res_blocks",type=int, default=9)
    return parser.parse_args()


def run_inference(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Select correct generator based on direction
    if opt.direction == "AtoB":
        src_folder = os.path.join(opt.data_root, "mripaired")
        out_folder = os.path.join(opt.output_dir, "fake_CT")
        model_key  = "G_state"    # G: MRI → CT
        print("Direction: MRI → CT")
    else:
        src_folder = os.path.join(opt.data_root, "ctpaired")
        out_folder = os.path.join(opt.output_dir, "fake_MRI")
        model_key  = "F_state"    # F: CT → MRI
        print("Direction: CT → MRI")

    os.makedirs(out_folder, exist_ok=True)

    # Load generator
    G = Generator(in_channels=1, out_channels=1, n_res_blocks=opt.n_res_blocks).to(device)
    ckpt = torch.load(opt.checkpoint, map_location=device)
    G.load_state_dict(ckpt[model_key])
    G.eval()
    print(f"Loaded generator from: {opt.checkpoint}")

    # Transform
    transform = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Run inference
    image_paths = list_images(src_folder)
    print(f"Found {len(image_paths)} images in {src_folder}\n")

    with torch.no_grad():
        for i, path in enumerate(image_paths):
            fname    = os.path.basename(path)
            img      = load_image(path)
            tensor   = transform(img).unsqueeze(0).to(device)   # [1,1,H,W]
            fake     = G(tensor)

            # Denormalize: [-1,1] → [0,1]
            fake_img = fake * 0.5 + 0.5

            out_path = os.path.join(out_folder, fname)
            save_image(fake_img, out_path)

            if (i + 1) % 20 == 0 or i == 0:
                print(f"  [{i+1}/{len(image_paths)}] {fname} → {out_path}")

    print(f"\n✅ Inference complete! Results saved to: {out_folder}")


if __name__ == "__main__":
    opt = run_inference(get_args())
