import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from pytorch_msssim import ssim
import os
import time
import argparse
import random
from datetime import timedelta

from models import Generator, Discriminator
from dataset import MedicalImageDataset
from losses import GradientDifferenceLoss, FrequencyDomainLoss

# ── Missing Utils embedded directly to prevent import crashes ──
class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []
    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch
    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

def save_checkpoint(dir_path, epoch, G, F, D_A, D_B, opt_G, opt_DA, opt_DB):
    torch.save({
        "G_state": G.state_dict(), "F_state": F.state_dict(),
        "DA_state": D_A.state_dict(), "DB_state": D_B.state_dict(),
        "opt_G": opt_G.state_dict(), "opt_DA": opt_DA.state_dict(), "opt_DB": opt_DB.state_dict()
    }, f"{dir_path}/checkpoint_epoch_{epoch:03d}.pth")

def load_checkpoint(dir_path, G, F, D_A, D_B, opt_G, opt_DA, opt_DB):
    import glob
    checkpoints = sorted(glob.glob(f"{dir_path}/checkpoint_epoch_*.pth"))
    if not checkpoints: return 0
    ckpt = torch.load(checkpoints[-1])
    G.load_state_dict(ckpt["G_state"]); F.load_state_dict(ckpt["F_state"])
    D_A.load_state_dict(ckpt["DA_state"]); D_B.load_state_dict(ckpt["DB_state"])
    opt_G.load_state_dict(ckpt["opt_G"])
    opt_DA.load_state_dict(ckpt["opt_DA"]); opt_DB.load_state_dict(ckpt["opt_DB"])
    epoch = int(checkpoints[-1].split("_")[-1].split(".")[0])
    print(f"Resumed from epoch {epoch}")
    return epoch

class ETATracker:
    def __init__(self, total_epochs, start_epoch=0):
        self.total = total_epochs
        self.start = start_epoch
        self.t0 = time.time()
        self.history = []
    def update(self, epoch_time):
        self.history.append(epoch_time)
        avg = sum(self.history[-5:]) / len(self.history[-5:])
        remaining = self.total - self.start - len(self.history)
        return str(timedelta(seconds=int(remaining * avg))), avg

# ── Argument Parser ──
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="outputs/")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--decay_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--n_res_blocks", type=int, default=9)
    parser.add_argument("--lambda_cyc", type=float, default=10.0)
    parser.add_argument("--lambda_id", type=float, default=5.0)
    parser.add_argument("--alpha_ssim", type=float, default=0.84)
    parser.add_argument("--lambda_edge", type=float, default=5.0)
    parser.add_argument("--lambda_fft", type=float, default=2.0)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--sample_every", type=int, default=5)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_compile", action="store_true")
    return parser.parse_args()

# ── Main Training Loop ──
def train(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and not opt.no_amp
    
    print("\n" + "=" * 60)
    print("  CycleGAN CT(A) -> MRI(B) Clinical Training Initialized")
    print("=" * 60 + "\n")

    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(opt.checkpoint_dir, exist_ok=True)
    os.makedirs(f"{opt.output_dir}/samples", exist_ok=True)

    G = Generator(in_channels=1, out_channels=1, n_res_blocks=opt.n_res_blocks).to(device)
    F = Generator(in_channels=1, out_channels=1, n_res_blocks=opt.n_res_blocks).to(device)
    D_A = Discriminator(in_channels=1).to(device)
    D_B = Discriminator(in_channels=1).to(device)

    G.apply(weights_init_normal); F.apply(weights_init_normal)
    D_A.apply(weights_init_normal); D_B.apply(weights_init_normal)

    criterion_GAN = nn.MSELoss()
    criterion_L1 = nn.L1Loss()
    criterion_edge = GradientDifferenceLoss().to(device)
    criterion_fft = FrequencyDomainLoss(high_freq_weight=2.0).to(device)

    optimizer_G = optim.Adam(list(G.parameters()) + list(F.parameters()), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_DA = optim.Adam(D_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_DB = optim.Adam(D_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    lr_sched_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, 0, opt.decay_epoch).step)
    lr_sched_DA = optim.lr_scheduler.LambdaLR(optimizer_DA, lr_lambda=LambdaLR(opt.n_epochs, 0, opt.decay_epoch).step)
    lr_sched_DB = optim.lr_scheduler.LambdaLR(optimizer_DB, lr_lambda=LambdaLR(opt.n_epochs, 0, opt.decay_epoch).step)

    scaler_G = GradScaler(enabled=use_amp)
    scaler_DA = GradScaler(enabled=use_amp)
    scaler_DB = GradScaler(enabled=use_amp)

    fake_CT_buf = ReplayBuffer()
    fake_MRI_buf = ReplayBuffer()

    start_epoch = 0
    if opt.resume: start_epoch = load_checkpoint(opt.checkpoint_dir, G, F, D_A, D_B, optimizer_G, optimizer_DA, optimizer_DB)

    transform_train = transforms.Compose([
        transforms.Resize(int(opt.img_size * 1.12)),
        transforms.RandomCrop(opt.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    train_ds = MedicalImageDataset(opt.data_root, transform_train, mode="train")
    train_dl = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_workers, pin_memory=True, drop_last=True)
    n_batches = len(train_dl)
    eta = ETATracker(opt.n_epochs, start_epoch)

    for epoch in range(start_epoch, opt.n_epochs):
        epoch_t0 = time.time()
        G.train(); F.train(); D_A.train(); D_B.train()
        
        for i, batch in enumerate(train_dl):
            real_CT = batch["A"].to(device, non_blocking=True)
            real_MRI = batch["B"].to(device, non_blocking=True)

            with torch.no_grad(): patch_shape = D_B(real_MRI).shape
            real_lbl = torch.ones(patch_shape, device=device)
            fake_lbl = torch.zeros(patch_shape, device=device)

            optimizer_G.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                id_MRI = G(real_MRI)
                id_CT = F(real_CT)
                loss_id = (criterion_L1(id_MRI, real_MRI) + criterion_L1(id_CT, real_CT)) * opt.lambda_id
                loss_id_edge = (criterion_edge(id_MRI, real_MRI) + criterion_edge(id_CT, real_CT)) * opt.lambda_edge * 0.5

                fake_MRI = G(real_CT)
                fake_CT = F(real_MRI)
                loss_adv = criterion_GAN(D_B(fake_MRI), real_lbl) + criterion_GAN(D_A(fake_CT), real_lbl)

                recov_CT = F(fake_MRI)
                recov_MRI = G(fake_CT)
                
                loss_cyc_L1 = criterion_L1(recov_CT, real_CT) + criterion_L1(recov_MRI, real_MRI)
                loss_cyc_ssim = (1 - ssim(recov_CT, real_CT, data_range=2.0, size_average=True)) + (1 - ssim(recov_MRI, real_MRI, data_range=2.0, size_average=True))
                loss_cyc_gdl = criterion_edge(recov_CT, real_CT) + criterion_edge(recov_MRI, real_MRI)
                loss_cyc_fft = criterion_fft(recov_CT, real_CT) + criterion_fft(recov_MRI, real_MRI)

                loss_cyc = opt.lambda_cyc * (
                    opt.alpha_ssim * loss_cyc_ssim + 
                    (1 - opt.alpha_ssim) * loss_cyc_L1 + 
                    opt.lambda_edge / opt.lambda_cyc * loss_cyc_gdl + 
                    opt.lambda_fft / opt.lambda_cyc * loss_cyc_fft
                )

                loss_G_total = loss_adv + loss_cyc + loss_id + loss_id_edge

            scaler_G.scale(loss_G_total).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()

            optimizer_DA.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                fake_CT_stored = fake_CT_buf.push_and_pop(fake_CT.detach())
                loss_DA = 0.5 * (criterion_GAN(D_A(real_CT), real_lbl) + criterion_GAN(D_A(fake_CT_stored.detach()), fake_lbl))
            scaler_DA.scale(loss_DA).backward()
            scaler_DA.step(optimizer_DA)
            scaler_DA.update()

            optimizer_DB.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                fake_MRI_stored = fake_MRI_buf.push_and_pop(fake_MRI.detach())
                loss_DB = 0.5 * (criterion_GAN(D_B(real_MRI), real_lbl) + criterion_GAN(D_B(fake_MRI_stored.detach()), fake_lbl))
            scaler_DB.scale(loss_DB).backward()
            scaler_DB.step(optimizer_DB)
            scaler_DB.update()

            if (i + 1) % opt.log_every == 0:
                print(f"  [E{epoch+1:03d} {i+1:04d}/{n_batches}] G:{loss_G_total.item():.3f} DA:{loss_DA.item():.3f}")

        epoch_time = time.time() - epoch_t0
        eta_str, _ = eta.update(epoch_time)
        print(f"\n{'─'*55}\nEpoch {epoch+1:03d}/{opt.n_epochs} | {epoch_time:.1f}s/epoch | ETA: {eta_str}\n{'─'*55}\n")

        if (epoch + 1) % opt.sample_every == 0:
            G.eval(); F.eval()
            with torch.no_grad():
                grid = torch.cat([real_CT[:1], fake_MRI[:1], recov_CT[:1], real_MRI[:1], fake_CT[:1], recov_MRI[:1]], dim=0)
                save_image(grid * 0.5 + 0.5, f"{opt.output_dir}/samples/epoch_{epoch+1:03d}.png", nrow=3, padding=2)

        if (epoch + 1) % opt.save_every == 0:
            save_checkpoint(opt.checkpoint_dir, epoch + 1, G, F, D_A, D_B, optimizer_G, optimizer_DA, optimizer_DB)

        lr_sched_G.step(); lr_sched_DA.step(); lr_sched_DB.step()

# ── The execution block you were missing ──
if __name__ == "__main__":
    opt = get_args()
    train(opt)