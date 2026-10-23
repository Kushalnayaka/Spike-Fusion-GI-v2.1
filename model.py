"""Compact retinal/SNN + ECA-CNN + bidirectional selective-scan classifier."""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage: torch.Tensor, threshold: float) -> torch.Tensor:
        ctx.save_for_backward(voltage)
        ctx.threshold = threshold
        return (voltage >= threshold).to(voltage.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (voltage,) = ctx.saved_tensors
        # Smooth triangular surrogate around threshold.
        grad = (1.0 - (voltage - ctx.threshold).abs()).clamp_min(0.0)
        return grad_output * grad, None


def _fixed_kernels() -> torch.Tensor:
    k = 7
    coords = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    r2 = xx.square() + yy.square()
    kernels = []
    for sigma in (0.9, 1.8):
        g1 = torch.exp(-r2 / (2 * sigma * sigma))
        sigma2 = sigma * 1.7
        g2 = torch.exp(-r2 / (2 * sigma2 * sigma2))
        dog = g1 / g1.sum() - g2 / g2.sum()
        kernels.append(dog)
    for theta in (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4):
        xr = xx * math.cos(theta) + yy * math.sin(theta)
        yr = -xx * math.sin(theta) + yy * math.cos(theta)
        gab = torch.exp(-(xr.square() + 0.7 * yr.square()) / (2 * 2.0**2)) * torch.cos(2 * math.pi * xr / 4.0)
        gab = gab - gab.mean()
        kernels.append(gab / gab.abs().sum().clamp_min(1e-6))
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8
    sobel_y = sobel_x.t()
    sobel_x = F.pad(sobel_x, (2, 2, 2, 2))
    sobel_y = F.pad(sobel_y, (2, 2, 2, 2))
    kernels.extend([sobel_x, sobel_y])
    return torch.stack(kernels)[:, None]


class RetinalEncoder(nn.Module):
    """Ten deterministic channels: two DoG, four Gabor, two opponent, two Sobel."""
    def __init__(self):
        super().__init__()
        self.register_buffer("kernels", _fixed_kernels(), persistent=True)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        gray = image[:, 0:1] * 0.299 + image[:, 1:2] * 0.587 + image[:, 2:3] * 0.114
        gray = F.pad(gray, (3, 3, 3, 3), mode="reflect")
        spatial = F.conv2d(gray, self.kernels)
        red_green = image[:, 0:1] - image[:, 1:2]
        blue_yellow = image[:, 2:3] - (image[:, 0:1] + image[:, 1:2]) * 0.5
        opponent = torch.cat((red_green, blue_yellow), dim=1)
        return torch.cat((spatial, opponent), dim=1)


class SNNBranch(nn.Module):
    def __init__(self, steps: int = 4, threshold: float = 0.7,
                 widths: Tuple[int, int] = (24, 32), out_channels: int = 32):
        super().__init__()
        self.steps = steps
        self.threshold = threshold
        self.retina = RetinalEncoder()
        hidden, spiking = widths
        self.conv1 = nn.Conv2d(10, hidden, 3, stride=2, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, spiking, 3, stride=2, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(spiking)
        self.project = nn.Conv2d(spiking, out_channels, 1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        retinal = self.retina(image)
        current = F.avg_pool2d(retinal, 4, 4)
        current = F.silu(self.norm1(self.conv1(current)))
        current = F.silu(self.norm2(self.conv2(current)))
        membrane = torch.zeros_like(current)
        spikes = []
        for _ in range(self.steps):
            membrane = membrane * 0.65 + current * 0.65
            spike = SurrogateSpike.apply(membrane, self.threshold)
            membrane = membrane - spike * self.threshold
            spikes.append(spike)
        spike_train = torch.stack(spikes, dim=1)
        rate = spike_train.mean(dim=1)
        feature = self.norm3(self.project(rate))
        return F.adaptive_avg_pool2d(F.silu(feature), (7, 7)), retinal, spike_train


class ECA(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.adaptive_avg_pool2d(x, 1).squeeze(-1).transpose(1, 2)
        y = torch.sigmoid(self.conv(y)).transpose(1, 2).unsqueeze(-1)
        return x * y


class InvertedBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, expansion: int = 2):
        super().__init__()
        hidden = cin * expansion
        self.use_residual = stride == 1 and cin == cout
        self.expand = nn.Sequential(nn.Conv2d(cin, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.SiLU())
        self.depthwise = nn.Sequential(nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False), nn.BatchNorm2d(hidden), nn.SiLU())
        self.project = nn.Sequential(nn.Conv2d(hidden, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.eca = ECA(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.eca(self.project(self.depthwise(self.expand(x))))
        return x + y if self.use_residual else y


class CNNBranch(nn.Module):
    def __init__(self, widths: Tuple[int, int, int, int] = (16, 24, 32, 48)):
        super().__init__()
        a, b, c, d = widths
        self.out_channels = d
        self.stem = nn.Sequential(nn.Conv2d(3, a, 3, 2, 1, bias=False), nn.BatchNorm2d(a), nn.SiLU())
        self.blocks = nn.Sequential(
            InvertedBlock(a, b, 2), InvertedBlock(b, c, 2),
            InvertedBlock(c, d, 2), InvertedBlock(d, d, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x))


class SelectiveScan1D(nn.Module):
    """Small input-selective diagonal SSM implemented only with torch operators."""
    def __init__(self, dim: int, state_dim: int = 8, expand: float = 1.5):
        super().__init__()
        self.dim = dim
        self.inner = int(dim * expand)
        self.state_dim = state_dim
        self.rank = max(4, dim // 12)
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, self.inner * 2)
        self.local = nn.Conv1d(self.inner, self.inner, 3, padding=2, groups=self.inner)
        self.param_proj = nn.Linear(self.inner, self.rank + state_dim * 2)
        self.dt_proj = nn.Linear(self.rank, self.inner)
        self.out_proj = nn.Linear(self.inner, dim)
        self.a_log = nn.Parameter(torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32).repeat(self.inner, 1)))
        self.skip = nn.Parameter(torch.ones(self.inner))

    def _scan(self, u: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        batch, length, _ = u.shape
        dt_raw, b_raw, c_raw = torch.split(params, [self.rank, self.state_dim, self.state_dim], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))
        decay_rate = torch.exp(self.a_log).to(dtype=u.dtype)
        state = torch.zeros(batch, self.inner, self.state_dim, device=u.device, dtype=u.dtype)
        outputs = []
        for t in range(length):
            delta = dt[:, t].unsqueeze(-1)
            decay = torch.exp(-delta * decay_rate.unsqueeze(0))
            b_t = b_raw[:, t].unsqueeze(1)
            c_t = c_raw[:, t].unsqueeze(1)
            drive = (1.0 - decay) * u[:, t].unsqueeze(-1) * b_t
            state = decay * state + drive
            y = (state * c_t).sum(dim=-1) + self.skip.unsqueeze(0) * u[:, t]
            outputs.append(y)
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        u, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        u = self.local(u.transpose(1, 2))[..., :x.shape[1]].transpose(1, 2)
        u = F.silu(u)
        params = self.param_proj(u)
        y_forward = self._scan(u, params)
        y_reverse = self._scan(u.flip(1), params.flip(1)).flip(1)
        y = 0.5 * (y_forward + y_reverse) * F.silu(gate)
        return residual + self.out_proj(y)


class EdgeTokenMixer(nn.Module):
    """Export-friendly convolutional token block used as an SSM deployment fallback."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.pointwise = nn.Conv1d(dim, dim, 1)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = y.transpose(1, 2)
        y = self.pointwise(F.silu(self.depthwise(y))).transpose(1, 2)
        x = x + y
        return x + self.ff(self.norm(x))


class GatedFusion(nn.Module):
    def __init__(self, snn_channels: int = 32, cnn_channels: int = 48, dim: int = 128):
        super().__init__()
        in_channels = snn_channels + cnn_channels
        self.project = nn.Sequential(nn.Conv2d(in_channels, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.SiLU())
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, max(8, dim // 8), 1), nn.SiLU(), nn.Conv2d(max(8, dim // 8), dim, 1), nn.Sigmoid())

    def forward(self, snn: torch.Tensor, cnn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fused = self.project(torch.cat((snn, cnn), dim=1))
        fused = fused * (0.5 + self.gate(fused))
        tokens = fused.flatten(2).transpose(1, 2)
        return tokens, fused


class SpikeFusion(nn.Module):
    def __init__(self, num_classes: int = 8, image_size: int = 224, token_dim: int = 128,
                 ssm_depth: int = 3, state_dim: int = 16, snn_steps: int = 4,
                 sequence_mode: str = "mamba", snn_widths: Tuple[int, int] = (24, 32),
                 snn_out_channels: int = 32, cnn_widths: Tuple[int, int, int, int] = (16, 24, 32, 48)):
        super().__init__()
        if image_size != 224:
            raise ValueError("This configuration uses a 7x7 token grid from 224x224 inputs")
        if sequence_mode not in {"mamba", "edge"}:
            raise ValueError("sequence_mode must be 'mamba' or 'edge'")
        self.num_classes = num_classes
        self.image_size = image_size
        self.token_dim = token_dim
        self.sequence_mode = sequence_mode
        self.snn_out_channels = snn_out_channels
        self.cnn_widths = tuple(cnn_widths)
        self.snn = SNNBranch(steps=snn_steps, widths=tuple(snn_widths), out_channels=snn_out_channels)
        self.cnn = CNNBranch(widths=tuple(cnn_widths))
        self.fusion = GatedFusion(snn_out_channels, self.cnn.out_channels, token_dim)
        block = SelectiveScan1D if sequence_mode == "mamba" else EdgeTokenMixer
        self.sequence = nn.ModuleList([block(token_dim, state_dim) if sequence_mode == "mamba" else block(token_dim) for _ in range(ssm_depth)])
        self.norm = nn.LayerNorm(token_dim)
        self.head = nn.Linear(token_dim, num_classes)

    def forward_with_aux(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        snn_map, retinal, spike_train = self.snn(image)
        cnn_map = self.cnn(image)
        tokens, fused_map = self.fusion(snn_map, cnn_map)
        for block in self.sequence:
            tokens = block(tokens)
        pooled = self.norm(tokens.mean(dim=1))
        logits = self.head(pooled)
        return logits, {"cnn_map": cnn_map, "snn_map": snn_map, "spike_train": spike_train,
                        "retinal_map": retinal, "fused_map": fused_map, "tokens": tokens}

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.forward_with_aux(image)[0]

    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def profile_config(profile: str = "mamba", num_classes: int = 8) -> Dict:
    if profile in {"mamba", "edge"}:
        return {"profile": profile, "num_classes": num_classes, "image_size": 224,
                "token_dim": 128, "ssm_depth": 3, "state_dim": 16, "snn_steps": 4,
                "sequence_mode": profile if profile == "mamba" else "edge",
                "snn_widths": [24, 32], "snn_out_channels": 32,
                "cnn_widths": [16, 24, 32, 48]}
    if profile == "micro":
        return {"profile": profile, "num_classes": num_classes, "image_size": 224,
                "token_dim": 48, "ssm_depth": 1, "state_dim": 8, "snn_steps": 3,
                "sequence_mode": "edge", "snn_widths": [12, 16],
                "snn_out_channels": 16, "cnn_widths": [8, 12, 16, 24]}
    raise ValueError("profile must be 'mamba', 'edge', or 'micro'")


def build_model(profile: str = "mamba", num_classes: int = 8, **overrides) -> SpikeFusion:
    config = profile_config(profile, num_classes)
    config.pop("profile", None)
    config.update(overrides)
    return SpikeFusion(**config)


def build_model_from_config(config: Dict) -> SpikeFusion:
    fields = ("num_classes", "image_size", "token_dim", "ssm_depth", "state_dim", "snn_steps",
              "sequence_mode", "snn_widths", "snn_out_channels", "cnn_widths")
    values = {name: config[name] for name in fields if name in config}
    return SpikeFusion(**values)
