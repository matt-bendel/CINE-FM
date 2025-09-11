# models/latent_fm_transformer.py
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# attention backends (priority: flash-attn v3 > v2 > torch SDPA)
# ---------------------------
_HAS_FA3 = False
_HAS_FA2 = False
try:
    # flash-attn v3
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as _fa3_varlen
    _HAS_FA3 = True
except Exception:
    _HAS_FA3 = False

if not _HAS_FA3:
    try:
        # flash-attn v2
        from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked as _fa2_varlen
        _HAS_FA2 = True
    except Exception:
        _HAS_FA2 = False


def sdpa(q, k, v, attn_mask=None):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)


class TimeEmbed(nn.Module):
    """sinusoidal -> MLP, like DiT."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.act = nn.SiLU()

    @staticmethod
    def timestep_embedding(t, dim):
        # t in (0,1], map to log-domain for better conditioning
        half = dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000.0) / (half - 1)))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        h = self.timestep_embedding(t.clamp(min=1e-4, max=1.0), self.dim)
        h = self.fc2(self.act(self.fc1(h)))
        return h


class PatchEmbed3D(nn.Module):
    """3D conv 'patch' embed (pt, ph, pw)."""
    def __init__(self, in_ch, embed_dim, patch_size=(1,1,1)):
        super().__init__()
        self.pt, self.ph, self.pw = patch_size
        self.proj = nn.Conv3d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        # x: [B,C,T,H,W]
        x = self.proj(x)  # [B, D, T', H', W']
        B, D, T, H, W = x.shape
        x = x.permute(0,2,3,4,1).reshape(B, T*H*W, D)  # [B, L, D]
        return self.norm(x), (T, H, W)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        inner = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, inner)
        self.fc2 = nn.Linear(inner, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class SingleStreamBlock(nn.Module):
    """
    HYVideo single-stream-ish block:
    - pre LN
    - fused linear to get qkv and parallel MLP stream
    - (varlen) attention backend
    - residual + gated modulation via time embedding vector
    """
    def __init__(self, dim, heads=16, mlp_ratio=4.0, qk_norm=True):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert self.head_dim * heads == dim
        self.pre = nn.LayerNorm(dim, eps=1e-6)

        self.fuse_in = nn.Linear(dim, dim * 3 + int(dim * mlp_ratio))
        self.fuse_out = nn.Linear(dim + int(dim * mlp_ratio), dim)
        self.mlp_act = nn.GELU()

        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()

        # time modulation -> gate
        self.mod = nn.Linear(dim, dim * 3)  # shift, scale, gate

    def _attn_varlen(self, q, k, v, seqlen):
        B, H, L, D = q.shape
        if _HAS_FA3 or _HAS_FA2:
            # pack qkv: [B, L, 3, H, D]
            qkv = torch.stack([q, k, v], dim=2).permute(0,1,3,2,4)  # [B,H,L,3,D]
            qkv = qkv.permute(0,2,3,1,4).contiguous()  # [B,L,3,H,D]
            qkv = qkv.reshape(B*L, 3, H, D)  # fa expects (B*L, 3, H, D) with varlen info

            # trivial (no ragged) -> we can just fallback to SDPA for simplicity / stability
            # Flash varlen wiring is complicated; SDPA is already very fast with PyTorch 2.
            return sdpa(q, k, v)
        else:
            return sdpa(q, k, v)

    def forward(self, x, tvec):
        """
        x: [B, L, D]
        tvec: [B, D] time embedding projected to dim
        """
        B, L, D = x.shape
        h = self.pre(x)
        shift, scale, gate = self.mod(tvec).chunk(3, dim=-1)
        h = h * (1.0 + scale[:, None, :]) + shift[:, None, :]

        fused = self.fuse_in(h)
        qkv, mlp_in = torch.split(fused, [D*3, fused.shape[-1]-D*3], dim=-1)
        q, k, v = qkv.chunk(3, dim=-1)

        # [B, L, H, Dh]
        def split_heads(z):
            return z.view(B, L, self.heads, self.head_dim).transpose(1,2).contiguous()

        q = self.q_norm(split_heads(q))
        k = self.k_norm(split_heads(k))
        v = split_heads(v)

        # SDPA expects [B*H, L, Dh]
        q_ = q.reshape(B*self.heads, L, self.head_dim)
        k_ = k.reshape(B*self.heads, L, self.head_dim)
        v_ = v.reshape(B*self.heads, L, self.head_dim)

        attn = sdpa(q_.transpose(0,1), k_.transpose(0,1), v_.transpose(0,1))  # [L, B*H, Dh]
        attn = attn.transpose(0,1).reshape(B, self.heads, L, self.head_dim).transpose(1,2)  # [B,L,H,Dh]
        attn = attn.reshape(B, L, D)

        mlp = self.mlp_act(mlp_in)
        out = self.fuse_out(torch.cat([attn, mlp], dim=-1))
        return x + gate[:, None, :] * out


class FinalProjector(nn.Module):
    """Project tokens back to latent video (velocity) via 3D 'unpatchify'."""
    def __init__(self, dim, out_ch, patch_size=(1,1,1), twh: Tuple[int,int,int]=(1,1,1)):
        super().__init__()
        self.dim = dim
        self.out_ch = out_ch
        self.pt, self.ph, self.pw = patch_size
        self.twh = twh  # (T',H',W') token grid
        self.proj = nn.Linear(dim, out_ch * self.pt * self.ph * self.pw)

    def forward(self, x):  # x: [B, L, D]
        B, L, D = x.shape
        T, H, W = self.twh
        y = self.proj(x)  # [B, L, out_ch * pt*ph*pw]
        y = y.view(B, T, H, W, self.out_ch, self.pt, self.ph, self.pw)  # block grid
        y = y.permute(0,4,1,5,2,6,3,7)  # [B,C,T,pt,H,ph,W,pw]
        y = y.reshape(B, self.out_ch, T*self.pt, H*self.ph, W*self.pw)  # [B,C,T,H,W]
        return y


class LatentFlowMatchTransformer(nn.Module):
    """
    Unconditional, single-stream DiT that predicts **velocity** in latent space.
    Inputs: z_t  [B, Cz, n, H, W], scalar t in (0,1]
    Output: u(z_t, t) velocity with same shape as input latent.
    """
    def __init__(
        self,
        latent_channels: int = 16,
        hidden_size: int = 768,
        depth: int = 16,
        heads: int = 12,
        mlp_ratio: float = 4.0,
        patch_size: Tuple[int,int,int]=(1,1,1),  # on latent grid
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden = hidden_size
        self.patch_size = patch_size

        self.t_embed = TimeEmbed(hidden_size)
        self.patch = PatchEmbed3D(latent_channels, hidden_size, patch_size=patch_size)

        self.blocks = nn.ModuleList([
            SingleStreamBlock(hidden_size, heads=heads, mlp_ratio=mlp_ratio, qk_norm=True)
            for _ in range(depth)
        ])
        # Final projection (velocity)
        self.final = None  # created at runtime after we know token grid

    def forward(self, zt: torch.Tensor, t: torch.Tensor):
        """
        zt: [B, Cz, n, H, W]
        t : [B] in (0,1]
        """
        B, C, n, H, W = zt.shape
        x, (Tg, Hg, Wg) = self.patch(zt)  # [B, L, D]
        if self.final is None:
            self.final = FinalProjector(self.hidden, self.latent_channels, patch_size=self.patch_size, twh=(Tg, Hg, Wg)).to(zt.device)
        tvec = self.t_embed(t)  # [B,D]
        for blk in self.blocks:
            x = blk(x, tvec)
        vel = self.final(x)  # [B,Cz,n,H,W]
        return vel
