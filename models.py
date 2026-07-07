# models.py
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer
from datasets import load_dataset


@dataclass
class TrainConfig:
    # --- Structural Adjustments ---
    image_size: int = 256
    latent_channels: int = 4
    text_dim: int = 512
    max_length: int = 32
    base_ch: int = 128

    # --- VRAM & Training Speed Tuning ---
    batch_size: int = 32
    lr: float = 5e-5
    num_epochs: int = 20
    num_workers: int = 10
    timesteps: int = 1000
    guidance_drop_prob: float = 0.1
    save_every: int = 2
    out_dir: str = "checkpoints"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def group_norm(ch: int, groups: int = 32):
    groups = min(groups, ch)
    while ch % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, ch)


class Flickr30kParquetDataset(Dataset):
    def __init__(self, split: str, image_size: int, tokenizer: CLIPTokenizer, max_length: int = 64):
        self.ds = load_dataset(
            "parquet",
            data_files={"train": "/mnt/d/local_flickr30k/data/000*.parquet"},
            split=split
        )
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"].convert("RGB")
        captions = item["caption"]
        caption = random.choice(captions) if isinstance(captions, (list, tuple)) else str(captions)
        image = self.transform(image)
        tokens = self.tokenizer(
            caption, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        return image, tokens["input_ids"].squeeze(0), tokens["attention_mask"].squeeze(0), caption


class DDPMScheduler:
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02, device="cuda"):
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device)

    def add_noise(self, x_start: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t])[:, None, None, None]
        return sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise, noise


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class CrossAttention2d(nn.Module):
    def __init__(self, ch: int, text_dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner = heads * dim_head
        self.norm = group_norm(ch)
        self.to_q = nn.Conv2d(ch, inner, 1, bias=False)
        self.to_kv = nn.Linear(text_dim, inner * 2, bias=False)
        self.proj = nn.Conv2d(inner, ch, 1)

    def forward(self, x, text_tokens):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        q = self.to_q(x).reshape(b, self.heads, self.dim_head, h * w).transpose(2, 3)
        kv = self.to_kv(text_tokens).chunk(2, dim=-1)
        k = kv[0].reshape(b, -1, self.heads, self.dim_head).transpose(1, 2)
        v = kv[1].reshape(b, -1, self.heads, self.dim_head).transpose(1, 2)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * (self.dim_head ** -0.5), dim=-1)
        out = torch.matmul(attn, v).transpose(2, 3).reshape(b, -1, h, w)
        return self.proj(out) + x_in


class ResnetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.norm1 = group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.norm2 = group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class PowerfulUNetDenoiser(nn.Module):
    """An advanced deep U-Net structure engineered for powerful feature estimation."""

    def __init__(self, in_channels: int = 4, text_dim: int = 512, model_channels: int = 128):
        super().__init__()
        time_dim = model_channels * 2
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(model_channels),
            nn.Linear(model_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Downsampling Stack
        self.down1 = ResnetBlock(model_channels, model_channels, time_dim)
        self.down2 = nn.Sequential(nn.Conv2d(model_channels, model_channels * 2, 3, stride=2, padding=1))
        self.down3 = ResnetBlock(model_channels * 2, model_channels * 2, time_dim)

        # Mid Bottleneck (Where Cross-Attention Learns Semantics)
        self.mid_block = ResnetBlock(model_channels * 2, model_channels * 2, time_dim)
        self.attn = CrossAttention2d(model_channels * 2, text_dim)

        # Upsampling Stack
        self.up1 = ResnetBlock(model_channels * 4, model_channels * 2, time_dim)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up3 = ResnetBlock(model_channels * 3, model_channels, time_dim)

        self.conv_out = nn.Sequential(
            group_norm(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, in_channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(timesteps)

        x1 = self.conv_in(x)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2)
        x4 = self.down3(x3, t_emb)

        h = self.mid_block(x4, t_emb)
        h = self.attn(h, text_tokens)

        h = self.up1(torch.cat([h, x4], dim=1), t_emb)
        h = self.up2(h)
        h = self.up3(torch.cat([h, x2], dim=1), t_emb)

        return self.conv_out(h)