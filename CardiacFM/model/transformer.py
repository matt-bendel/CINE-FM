# models/latent_fm_transformer.py
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Attention backends (priority: flash-attn v3 > flash-attn v2 > sage-attn v2 > SDPA)
# -----------------------------------------------------------------------------
_HAS_FA3 = False
_HAS_FA2 = False
_HAS_SAGE2 = False

_fa_func = None
_sage_attn = None

# Detect FA3 by its varlen symbol; still use flash_attn_func (dense) at runtime.
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as _fa3_varlen  # noqa: F401
    from flash_attn import flash_attn_func as _fa_func
    _fa_func  # silence linter if import succeeded
    _fa_func_available = True
    _HAS_FA3 = True
except Exception:
    _fa_func_available = False

# If FA3 isn't present, try FA2 (still exposes flash_attn_func)
if not _HAS_FA3:
    try:
        from flash_attn import flash_attn_func as _fa_func
        _fa_func  # noqa: F401
        _fa_func_available = True
        _HAS_FA2 = True
    except Exception:
        _fa_func_available = False

# Sage-Attn v2
if not (_HAS_FA3 or _HAS_FA2):
    try:
        from sageattention import sageattn as _sage_attn
        _sage_attn  # noqa: F401
        _HAS_SAGE2 = True
    except Exception:
        _HAS_SAGE2 = False


def _attn_blh(q_blh, k_blh, v_blh):
    """
    Backend-chooser for attention.
    Inputs/Outputs are [B, L, H, Dh] (BLHD).
    """
    # flash-attn v3 or v2
    if _fa_func_available:
        # flash_attn_func expects [B, L, H, Dh]
        return _fa_func(q_blh, k_blh, v_blh, dropout_p=0.0, softmax_scale=None, causal=False)

    # sage-attn v2
    if _HAS_SAGE2:
        try:
            # Common builds accept BLHD; fall back to another layout if needed.
            return _sage_attn(q_blh, k_blh, v_blh, tensor_layout="BLHD")
        except Exception:
            # Last try – some wheels expect "NHD" (we map [B,L,H,D] -> [L*B,H,D]).
            B, L, H, Dh = q_blh.shape
            q_nhd = q_blh.reshape(B * L, H, Dh)
            k_nhd = k_blh.reshape(B * L, H, Dh)
            v_nhd = v_blh.reshape(B * L, H, Dh)
            out_nhd = _sage_attn(q_nhd, k_nhd, v_nhd, tensor_layout="NHD")
            return out_nhd.reshape(B, L, H, Dh)

    # PyTorch SDPA fallback
    # Convert to [L, B*H, Dh] for scaled_dot_product_attention
    B, L, H, Dh = q_blh.shape
    q_lbh = q_blh.permute(1, 0, 2, 3).reshape(L, B * H, Dh)
    k_lbh = k_blh.permute(1, 0, 2, 3).reshape(L, B * H, Dh)
    v_lbh = v_blh.permute(1, 0, 2, 3).reshape(L, B * H, Dh)
    out_lbh = F.scaled_dot_product_attention(q_lbh, k_lbh, v_lbh, dropout_p=0.0, is_causal=False)
    return out_lbh.reshape(L, B, H, Dh).permute(1, 0, 2, 3).contiguous()


# -----------------------------------------------------------------------------
# Time embedding (unchanged)
# -----------------------------------------------------------------------------
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
        half = dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000.0) / (half - 1)))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        h = self.timestep_embedding(t, self.dim)
        h = self.fc2(self.act(self.fc1(h)))
        return h


# -----------------------------------------------------------------------------
# 3D patch embed (unchanged)
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# MLP
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        inner = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, inner)
        self.fc2 = nn.Linear(inner, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# -----------------------------------------------------------------------------
# RoPE for 3D grids (T/H/W), HYVideo-style
# -----------------------------------------------------------------------------
def _even(x: int) -> int:
    return x if (x % 2 == 0) else (x - 1)

class HunyuanVideoRotaryPosEmbed(nn.Module):
    """
    Produces per-token cos/sin for 3D positions.
    rope_dim = (DT, DY, DX) (each even); sum == head_dim
    Returns per-sample: [L, 2*Dh] where Dh = DT+DY+DX (first Dh = cos, last Dh = sin).
    """
    def __init__(self, rope_dim: Tuple[int,int,int], theta: float = 256.0):
        super().__init__()
        DT, DY, DX = rope_dim
        assert DT >= 0 and DY >= 0 and DX >= 0
        self.DT, self.DY, self.DX = int(DT), int(DY), int(DX)
        self.theta = float(theta)

    @torch.no_grad()
    def _axis_freqs(self, dim: int, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if dim == 0:
            s = pos.new_zeros((0,) + tuple(pos.shape))
            return s, s
        freqs = 1.0 / (self.theta ** (torch.arange(0, dim, 2, device=pos.device, dtype=torch.float32)[: (dim // 2)] / dim))
        freqs = torch.outer(freqs, pos.reshape(-1)).unflatten(-1, pos.shape).repeat_interleave(2, dim=0)  # [dim, ...]
        return freqs.cos(), freqs.sin()

    @torch.no_grad()
    def forward(self, T: int, H: int, W: int, device) -> torch.Tensor:
        t = torch.arange(0, T, device=device, dtype=torch.float32)
        y = torch.arange(0, H, device=device, dtype=torch.float32)
        x = torch.arange(0, W, device=device, dtype=torch.float32)
        GT, GY, GX = torch.meshgrid(t, y, x, indexing="ij")

        FCT, FST = self._axis_freqs(self.DT, GT)
        FCY, FSY = self._axis_freqs(self.DY, GY)
        FCX, FSX = self._axis_freqs(self.DX, GX)

        freqs = torch.cat([FCT, FCY, FCX, FST, FSY, FSX], dim=0)  # [2*Dh, T, H, W]
        freqs = freqs.flatten(1).transpose(0, 1)  # [L, 2*Dh]
        return freqs


def _apply_rope_qk(q: torch.Tensor, k: torch.Tensor, rope_freqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q,k: [B, H, L, Dh]
    rope_freqs: [L, 2*Dh]  (first Dh are cos, last Dh are sin)
    """
    B, H, L, Dh = q.shape
    assert rope_freqs.shape[0] == L, f"RoPE length mismatch: {rope_freqs.shape[0]} vs {L}"
    assert rope_freqs.shape[1] == 2 * Dh, f"RoPE dim mismatch: {rope_freqs.shape[1]} vs {2*Dh}"
    cos, sin = rope_freqs.chunk(2, dim=-1)  # [L,Dh]
    cos = cos.view(1, 1, L, Dh)
    sin = sin.view(1, 1, L, Dh)

    def _rotate(x):
        x_ = x.unflatten(-1, (-1, 2))
        x1, x2 = x_.unbind(-1)
        xr = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return xr

    q_rot = _rotate(q)
    k_rot = _rotate(k)
    q = (q * cos) + (q_rot * sin)
    k = (k * cos) + (k_rot * sin)
    return q, k


# -----------------------------------------------------------------------------
# Transformer block w/ RoPE and pluggable attention backend
# -----------------------------------------------------------------------------
class SingleStreamBlock(nn.Module):
    """
    HYVideo-lean block:
    - pre LN
    - fused linear to get qkv and parallel MLP stream
    - attention backend (FA3 > FA2 > SAGE2 > SDPA) + RoPE applied to Q/K
    - residual + gated modulation via time embedding vector
    """
    def __init__(self, dim, heads=16, mlp_ratio=4.0, qk_norm=True):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert self.head_dim * heads == dim, "hidden size must be divisible by heads"
        assert self.head_dim % 2 == 0, "head_dim must be even to use RoPE"

        self.pre = nn.LayerNorm(dim, eps=1e-6)

        self.fuse_in = nn.Linear(dim, dim * 3 + int(dim * mlp_ratio))
        self.fuse_out = nn.Linear(dim + int(dim * mlp_ratio), dim)
        self.mlp_act = nn.GELU()

        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()

        # time modulation -> gate
        self.mod = nn.Linear(dim, dim * 3)  # shift, scale, gate

    def forward(self, x, tvec, rope_freqs=None):
        """
        x: [B, L, D]
        tvec: [B, D] time embedding projected to dim
        rope_freqs: [L, 2*Dh] (Dh=head_dim) or None
        """
        B, L, D = x.shape
        h = self.pre(x)
        shift, scale, gate = self.mod(tvec).chunk(3, dim=-1)
        h = h * (1.0 + scale[:, None, :]) + shift[:, None, :]

        fused = self.fuse_in(h)
        qkv, mlp_in = torch.split(fused, [D*3, fused.shape[-1]-D*3], dim=-1)
        q, k, v = qkv.chunk(3, dim=-1)

        # [B, L, H, Dh] <-> [B, H, L, Dh]
        def split_heads_blh(z):
            return z.view(B, L, self.heads, self.head_dim)

        def split_heads_bhl(z):
            return z.view(B, L, self.heads, self.head_dim).transpose(1, 2).contiguous()

        q_bhl = self.q_norm(split_heads_bhl(q))  # [B,H,L,Dh]
        k_bhl = self.k_norm(split_heads_bhl(k))
        v_bhl = split_heads_bhl(v)

        # RoPE (expects [B,H,L,Dh])
        if rope_freqs is not None:
            q_bhl, k_bhl = _apply_rope_qk(q_bhl, k_bhl, rope_freqs)

        # Backend attention operates on [B, L, H, Dh]
        q_blh = q_bhl.transpose(1, 2).contiguous()
        k_blh = k_bhl.transpose(1, 2).contiguous()
        v_blh = v_bhl.transpose(1, 2).contiguous()

        attn_blh = _attn_blh(q_blh, k_blh, v_blh)  # [B,L,H,Dh]
        attn = attn_blh.transpose(1, 2).reshape(B, L, D)  # -> [B,L,D]

        mlp = self.mlp_act(mlp_in)
        out = self.fuse_out(torch.cat([attn, mlp], dim=-1))
        return x + gate[:, None, :] * out


# -----------------------------------------------------------------------------
# Project tokens back to latent grid
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Main transformer (unchanged API; now with RoPE + backend-priority attention)
# -----------------------------------------------------------------------------
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
        # --- RoPE params ---
        use_rope: bool = True,
        rope_theta: float = 256.0,
        rope_axes_dim: Optional[Tuple[int,int,int]] = None,  # (DT, DY, DX); if None -> split evenly from head_dim
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden = hidden_size
        self.patch_size = patch_size

        self.heads = int(heads)
        self.head_dim = hidden_size // self.heads
        assert self.head_dim * self.heads == hidden_size, "hidden_size must be divisible by heads"
        assert self.head_dim % 2 == 0, "head_dim must be even to use RoPE"

        # infer rope dims if not provided
        if rope_axes_dim is None:
            d = self.head_dim
            dt = _even(d // 3)
            dy = _even((d - dt) // 2)
            dx = d - dt - dy
            if dx % 2 == 1:
                if dy >= 2:
                    dy -= 1
                elif dt >= 2:
                    dt -= 1
                dx += 1
            rope_axes_dim = (dt, dy, dx)
        else:
            dt, dy, dx = rope_axes_dim
            assert (dt + dy + dx) == self.head_dim, f"sum(rope_axes_dim) must equal head_dim ({self.head_dim})"
            assert dt % 2 == 0 and dy % 2 == 0 and dx % 2 == 0, "each rope_axes_dim must be even"

        self.use_rope = bool(use_rope)
        self.rope = HunyuanVideoRotaryPosEmbed(rope_axes_dim, theta=rope_theta) if self.use_rope else None

        self.t_embed = TimeEmbed(hidden_size)
        self.patch = PatchEmbed3D(latent_channels, hidden_size, patch_size=patch_size)

        self.blocks = nn.ModuleList([
            SingleStreamBlock(hidden_size, heads=heads, mlp_ratio=mlp_ratio, qk_norm=True)
            for _ in range(depth)
        ])
        # Final projection (velocity)
        self.final = None  # created at runtime after we know token grid

    def _rope_for_grid(self, Tg: int, Hg: int, Wg: int, device) -> Optional[torch.Tensor]:
        if not self.use_rope:
            return None
        return self.rope(Tg, Hg, Wg, device=device)  # [L, 2*Dh]

    def forward(self, zt: torch.Tensor, t: torch.Tensor):
        """
        zt: [B, Cz, n, H, W]
        t : [B] in (0,1]
        """
        B, C, n, H, W = zt.shape
        x, (Tg, Hg, Wg) = self.patch(zt)  # [B, L, D]
        if self.final is None:
            self.final = FinalProjector(self.hidden, self.latent_channels, patch_size=self.patch_size, twh=(Tg, Hg, Wg)).to(zt.device)

        rope_freqs = self._rope_for_grid(Tg, Hg, Wg, zt.device)  # [L, 2*Dh] or None

        tvec = self.t_embed(t)  # [B,D]
        for blk in self.blocks:
            x = blk(x, tvec, rope_freqs=rope_freqs)
        vel = self.final(x)  # [B,Cz,n,H,W]
        return vel
