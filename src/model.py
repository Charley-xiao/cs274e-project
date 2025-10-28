from __future__ import annotations
from typing import Optional, List, Tuple, Dict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal embedding for times t in [0,1].
    t: (B,1,1,1) or (B,1)
    return: (B, dim)
    """
    if t.dim() == 4:
        t = t.squeeze(-1).squeeze(-1)  # (B,1) -> (B,)
    t = t.view(-1)  # (B,)

    half = dim // 2
    # Use a log-spaced frequency spectrum
    freqs = torch.exp(
        torch.linspace(0, math.log(10000), steps=half, device=t.device, dtype=t.dtype)
    )
    args = t[:, None] * freqs[None, :] * 2.0 * math.pi
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb  # (B, dim)


def _n_groups(c: int) -> int:
    return min(32, c)


class ResBlock(nn.Module):
    """
    Simple ResNet block with additive conditioning.
    - Two convs with GroupNorm + SiLU.
    - Conditioning vector 'cond' (B, cond_dim) is linearly projected to out_channels
      and added before the second conv.
    """
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.norm1 = nn.GroupNorm(_n_groups(in_ch), in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.cond_proj = nn.Linear(cond_dim, out_ch)

        self.norm2 = nn.GroupNorm(_n_groups(out_ch), out_ch)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        # Inject conditioning (additive FiLM-like bias)
        c = self.cond_proj(cond)[:, :, None, None]
        h = h + c
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class ConditionalUNet(nn.Module):
    """
    Small conditional U-Net for 64x64 images.
    - Conditioning vector = t_mlp(timestep_embedding) + class_emb(y_proc)
      where y_proc replaces -1 with the learned null class (index K).
    - Forward API: v = model(x_t, t, y)
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        cond_dim: int = 256,
        num_classes: int = 10,   # EuroSAT RGB
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.cond_dim = cond_dim
        self.num_classes = num_classes
        self.null_class_idx = num_classes  # for y == -1
        self.dropout = dropout

        # Input stem
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Time + class conditioning
        self.time_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.class_emb = nn.Embedding(num_classes + 1, cond_dim)  # +1 for null token

        # Down path
        ch = base_channels
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skip_channels: List[int] = []
        for i, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(ch, out_ch, cond_dim, dropout))
                ch = out_ch
                skip_channels.append(ch)
            if i != len(channel_mult) - 1:
                self.downsamples.append(Downsample(ch))
            else:
                self.downsamples.append(nn.Identity())

        # Middle
        self.mid1 = ResBlock(ch, ch, cond_dim, dropout)
        self.mid2 = ResBlock(ch, ch, cond_dim, dropout)

        # Up path
        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                # concatenate skip: ch (current) + skip_ch
                skip_ch = skip_channels.pop()
                self.up_blocks.append(ResBlock(ch + skip_ch, out_ch, cond_dim, dropout))
                ch = out_ch
            if i != 0:
                self.upsamples.append(Upsample(ch))
            else:
                self.upsamples.append(nn.Identity())

        # Output head
        self.out_norm = nn.GroupNorm(_n_groups(ch), ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        # Kaiming for convs, xavier for linears; embeddings default
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_cond(self, t: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Build conditioning vector (B, cond_dim) from time and class.
        y: (B,) with values in {-1,...,K-1}; -1 is mapped to null_class_idx.
        """
        B = t.shape[0]
        t_emb = timestep_embedding(t, self.cond_dim)           # (B, cond_dim)
        t_feat = self.time_mlp(t_emb)                          # (B, cond_dim)

        if y is None:
            y_proc = torch.full((B,), self.null_class_idx, device=t.device, dtype=torch.long)
        else:
            y_proc = y.clone().to(device=t.device, dtype=torch.long)
            y_proc = torch.where(y_proc < 0, torch.full_like(y_proc, self.null_class_idx), y_proc)
            # Safety clamp
            y_proc = y_proc.clamp_(0, self.num_classes)

        y_feat = self.class_emb(y_proc)                        # (B, cond_dim)
        cond = t_feat + y_feat
        return cond

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x:  (B,C,H,W)   normalized images at interpolation point
        t:  (B,1,1,1)   time in [0,1]
        y:  (B,) Long   class indices; -1 means unconditional token
        returns: velocity field v(x,t,y) with same shape as x
        """
        cond = self._make_cond(t, y)  # (B, cond_dim)

        # Down path with skips
        h = self.in_conv(x)
        skips: List[torch.Tensor] = []
        down_idx = 0
        for i, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[down_idx](h, cond)
                down_idx += 1
                skips.append(h)
            h = self.downsamples[i](h)

        # Middle
        h = self.mid1(h, cond)
        h = self.mid2(h, cond)

        # Up path (note: reverse skips)
        up_idx = 0
        for i, mult in reversed(list(enumerate(self.channel_mult))):
            for _ in range(self.num_res_blocks):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.up_blocks[up_idx](h, cond)
                up_idx += 1
            h = self.upsamples[len(self.channel_mult) - 1 - i](h)

        # Output head
        h = self.out_conv(self.out_act(self.out_norm(h)))
        return h



def create_model(cfg: Dict, num_classes: int) -> nn.Module:
    """
    Factory used by train/sample code.
    cfg keys (all optional):
      - in_channels: int (default 3)
      - base_channels: int (default 64)
      - channel_mult: list[int] (default [1,2,2,4])
      - num_res_blocks: int (default 2)
      - cond_dim: int (default 256)
      - dropout: float (default 0.0)
    """
    in_channels   = cfg.get("in_channels", 3)
    base_channels = cfg.get("base_channels", 64)
    channel_mult  = tuple(cfg.get("channel_mult", (1, 2, 2, 4)))
    num_res_blocks = cfg.get("num_res_blocks", 2)
    cond_dim      = cfg.get("cond_dim", 256)
    dropout       = cfg.get("dropout", 0.0)

    return ConditionalUNet(
        in_channels=in_channels,
        base_channels=base_channels,
        channel_mult=channel_mult,
        num_res_blocks=num_res_blocks,
        cond_dim=cond_dim,
        num_classes=num_classes,
        dropout=dropout,
    )
