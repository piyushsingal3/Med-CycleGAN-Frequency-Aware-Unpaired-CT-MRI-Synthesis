"""
losses.py — Custom Loss Functions for Clinical CycleGAN CT↔MRI
Novel Contributions:
  1. GradientDifferenceLoss      : Sobel edge loss for bone/boundary sharpness
  2. FrequencyDomainLoss (NEW)   : Weighted FFT magnitude loss; penalizes
                                   high-frequency synthesis errors (bone,
                                   calcifications) 2x harder than soft tissue.
                                   Clinically motivated: CT↔MRI differ most
                                   in their high-frequency content.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientDifferenceLoss(nn.Module):
    """
    L1 loss on Sobel edge maps of generated vs real images.
    Forces the generator to preserve spatial gradients (bone
    boundaries, organ edges) — not just pixel intensity.
    """
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1.,  0.,  1.],
                                 [-2.,  0.,  2.],
                                 [-1.,  0.,  1.]]).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1., -2., -1.],
                                 [ 0.,  0.,  0.],
                                 [ 1.,  2.,  1.]]).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        fake_dx = F.conv2d(fake, self.kernel_x, padding=1)
        fake_dy = F.conv2d(fake, self.kernel_y, padding=1)
        real_dx = F.conv2d(real, self.kernel_x, padding=1)
        real_dy = F.conv2d(real, self.kernel_y, padding=1)
        return F.l1_loss(fake_dx, real_dx) + F.l1_loss(fake_dy, real_dy)


class FrequencyDomainLoss(nn.Module):
    """
    Weighted FFT Magnitude Loss.

    Motivation: CT and MRI differ most in their high-frequency spectrum.
    Bone, trabecular structure, and calcifications in CT manifest as sharp
    high-frequency components that MRI lacks. Standard pixel/SSIM losses
    are biased toward low-frequency (brightness, contrast) errors and
    under-penalize high-freq synthesis failures.

    This loss splits the 2D frequency spectrum into a low-freq (soft tissue)
    band and a high-freq (bone/edge detail) band, weighting the high-freq
    component more heavily during training.

    Args:
        high_freq_weight (float): Multiplier for high-frequency error.
                                  Default 2.0. Increase for sharper bone.
        low_freq_radius_ratio (float): Fraction of image size defining the
                                       low-frequency centre region. Default 0.25.
    """
    def __init__(self, high_freq_weight: float = 2.0,
                 low_freq_radius_ratio: float = 0.25):
        super().__init__()
        self.high_freq_weight = high_freq_weight
        self.low_freq_radius_ratio = low_freq_radius_ratio
        # Cache the mask between forward calls if image size is constant
        self._mask_cache: dict = {}

    def _get_masks(self, h: int, w: int,
                   device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (h, w, device)
        if key not in self._mask_cache:
            cy, cx = h // 2, w // 2
            Y, X = torch.meshgrid(
                torch.arange(h, device=device, dtype=torch.float32),
                torch.arange(w, device=device, dtype=torch.float32),
                indexing='ij'
            )
            dist = torch.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
            radius = min(h, w) * self.low_freq_radius_ratio
            low_mask  = (dist < radius).float()            # soft tissue band
            high_mask = 1.0 - low_mask                     # bone / fine detail band
            self._mask_cache[key] = (low_mask, high_mask)
        return self._mask_cache[key]

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        h, w = fake.shape[-2], fake.shape[-1]

        # ── CRITICAL FIX: Cast to float32 to prevent AMP / float16 crashes in FFT ──
        fake_fp32 = fake.to(torch.float32)
        real_fp32 = real.to(torch.float32)

        # 2D FFT → shift DC to centre → magnitude spectrum
        fake_mag = torch.abs(torch.fft.fftshift(
            torch.fft.fft2(fake_fp32, norm='ortho'), dim=(-2, -1)))
        real_mag = torch.abs(torch.fft.fftshift(
            torch.fft.fft2(real_fp32, norm='ortho'), dim=(-2, -1)))

        low_mask, high_mask = self._get_masks(h, w, fake.device)

        loss_low  = F.l1_loss(fake_mag * low_mask,  real_mag * low_mask)
        loss_high = F.l1_loss(fake_mag * high_mask, real_mag * high_mask)

        return loss_low + self.high_freq_weight * loss_high