"""
models.py — Generator & Discriminator for Clinical CycleGAN CT↔MRI

Architecture Choices:
  • Generator   : CBAM-augmented ResNet (9 blocks @ 256px)
                  Each residual block has Channel + Spatial Attention (CBAM)
                  to focus the network on anatomically relevant structures.
  • Discriminator: PatchGAN with Spectral Normalization on every Conv layer
                  for stable adversarial training at 256px.
"""
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────
#  CBAM Attention Modules
# ─────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel recalibration.
    Uses both average AND max pooling for richer statistics.
    """
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    """
    Spatial gating: 'where' to attend in the feature map.
    kernel_size=7 provides a wide receptive field for anatomical structures.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class AttentionResidualBlock(nn.Module):
    """
    Standard ResBlock + CBAM (Channel then Spatial attention gate).
    The attention reweights features BEFORE the skip connection,
    helping the generator focus on bone/tissue boundaries.
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )
        self.ca = ChannelAttention(in_features)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_block(x)
        out = out * self.ca(out)   # channel gate
        out = out * self.sa(out)   # spatial gate
        return x + out             # residual connection


# ─────────────────────────────────────────────────────────
#  Generator  (Encoder → CBAM-ResBlocks → Decoder)
# ─────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    ResNet generator with CBAM attention in every residual block.
    
    Args:
        in_channels  : 1 for grayscale medical images
        out_channels : 1 for grayscale output
        n_res_blocks : 9 for 256px (architectural requirement)
        ngf          : base feature count (64)
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 n_res_blocks: int = 9, ngf: int = 64):
        super().__init__()

        # ── Initial Conv ──
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]

        # ── Encoder (2× downsampling) ──
        in_f = ngf
        for _ in range(2):
            out_f = in_f * 2
            layers += [
                nn.Conv2d(in_f, out_f, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_f),
                nn.ReLU(inplace=True)
            ]
            in_f = out_f  # 256 channels after 2 downs

        # ── Bottleneck: CBAM-augmented Residual Blocks ──
        for _ in range(n_res_blocks):
            layers.append(AttentionResidualBlock(in_f))

        # ── Decoder (2× upsampling) ──
        for _ in range(2):
            out_f = in_f // 2
            layers += [
                nn.ConvTranspose2d(in_f, out_f, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_f),
                nn.ReLU(inplace=True)
            ]
            in_f = out_f

        # ── Output Conv ──
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ─────────────────────────────────────────────────────────
#  Discriminator  (70×70 PatchGAN + Spectral Norm)
# ─────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    70×70 PatchGAN discriminator with Spectral Normalization.
    
    Spectral Norm constrains the Lipschitz constant of each layer,
    significantly stabilising adversarial training at 256px without
    needing gradient penalty or careful learning-rate tuning.
    """
    def __init__(self, in_channels: int = 1, ndf: int = 64):
        super().__init__()

        def sn_block(in_f, out_f, stride=2, norm=True):
            """SpectralNorm Conv → optional InstanceNorm → LeakyReLU"""
            layers = [nn.utils.spectral_norm(
                nn.Conv2d(in_f, out_f, 4, stride=stride, padding=1, bias=not norm)
            )]
            if norm:
                layers.append(nn.InstanceNorm2d(out_f, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *sn_block(in_channels, ndf,    norm=False),   # 256 → 128
            *sn_block(ndf,         ndf*2),                # 128 → 64
            *sn_block(ndf*2,       ndf*4),                # 64  → 32
            *sn_block(ndf*4,       ndf*8,  stride=1),     # 32  → 31
            nn.utils.spectral_norm(
                nn.Conv2d(ndf*8, 1, 4, stride=1, padding=1)  # 31  → 30 (patch output)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)