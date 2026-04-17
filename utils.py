import torch
from pytorch_msssim import ssim as _ssim


def calculate_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error. Lower is better."""
    return torch.mean(torch.abs(pred - target)).item()


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor,
                   data_range: float = 2.0) -> float:
    """
    Peak Signal-to-Noise Ratio. Higher is better.
    data_range=2.0 because images are normalised to [-1, 1].
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return (20 * torch.log10(torch.tensor(data_range) / torch.sqrt(mse))).item()


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor,
                   data_range: float = 2.0) -> float:
    """Structural Similarity Index. Higher is better."""
    return _ssim(pred, target, data_range=data_range, size_average=True).item()