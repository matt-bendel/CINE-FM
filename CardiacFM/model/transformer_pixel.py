# models/latent_fm_transformer.py
import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# -----------------------------------------------------------------------------
# Attention backends (priority: flash-attn v3 > flash-attn v2 > sage-attn v2 > SDPA)
# -----------------------------------------------------------------------------
_HAS_FA3 = False
_HAS_FA2 = False
_HAS_SAGE2 = False

_fa_func = None
_sage_attn = None

# -----------------------------------------------------------------------------
# Attention backends (priority: FA3 dense > FA3 varlen > FA2 dense > SDPA)
# -----------------------------------------------------------------------------
import torch
import torch.nn.functional as F

_HAS_FA3_DENSE = False
_HAS_FA3_VARLEN = False
_HAS_FA2 = False

_fa3_dense = None
_fa3_varlen = None
_fa3_varlen_qkvpacked = None
_fa2_dense = None

# FA3 dense
# FA3 varlen (either separate or qkvpacked)
if not _HAS_FA3_DENSE:
    try:
        from flash_attn_3.flash_attn_interface import flash_attn_varlen_func as _fa3_varlen
        _HAS_FA3_VARLEN = True
    except Exception:
        try:
            from flash_attn_3.flash_attn_interface import flash_attn_varlen_qkvpacked_func as _fa3_varlen_qkvpacked
            _HAS_FA3_VARLEN = True
        except Exception:
            pass

# FA2 dense (NOTE: correct path is flash_attn.flash_attn_interface)
if not (_HAS_FA3_DENSE or _HAS_FA3_VARLEN):
    try:
        from flash_attn.flash_attn_interface import flash_attn_func as _fa2_dense
        _HAS_FA2 = True
    except Exception:
        pass


def _fa3_varlen_dense_wrapper(q_blh, k_blh, v_blh):
    """
    Use FA3 varlen kernels to service dense BLHD input.
    q/k/v: [B, L, H, Dh] (bf16/fp16) -> returns same shape
    """
    B, L, H, Dh = q_blh.shape
    device = q_blh.device
    cu = torch.arange(0, (B + 1) * L, step=L, device=device, dtype=torch.int32)

    if _fa3_varlen is not None:
        q = q_blh.reshape(B * L, H, Dh)
        k = k_blh.reshape(B * L, H, Dh)
        v = v_blh.reshape(B * L, H, Dh)
        o = _fa3_varlen(
            q, k, v,
            cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=L, max_seqlen_k=L,
            dropout_p=0.0, softmax_scale=None, causal=False
        )
        return o.reshape(B, L, H, Dh)

    # qkvpacked fallback
    q = q_blh.reshape(B * L, H, Dh)
    k = k_blh.reshape(B * L, H, Dh)
    v = v_blh.reshape(B * L, H, Dh)
    qkv = torch.stack([q, k, v], dim=1)  # [B*L, 3, H, Dh]
    o = _fa3_varlen_qkvpacked(
        qkv, cu_seqlens=cu, max_seqlen=L,
        dropout_p=0.0, softmax_scale=None, causal=False
    )
    return o.reshape(B, L, H, Dh)


# Detect FA3 by its varlen symbol; still use flash_attn_func (dense) at runtime.
# if not _HAS
# try:
#     from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as _fa3_varlen  # noqa: F401
#     from flash_attn import flash_attn_func as _fa_func
#     _fa_func  # silence linter if import succeeded
#     _fa_func_available = True
#     _HAS_FA3 = True
# except Exception:
#     _fa_func_available = False

# If FA3 isn't present, try FA2 (still exposes flash_attn_func)

def _attn_blh(q_blh, k_blh, v_blh):
    """
    Inputs/Outputs are [B, L, H, Dh] (BLHD).
    """
    # FA3 dense → FA3 varlen → FA2 → SDPA
    if _HAS_FA3_DENSE:
        return _fa3_dense(q_blh, k_blh, v_blh, dropout_p=0.0, softmax_scale=None, causal=False)

    if _HAS_FA3_VARLEN:
        return _fa3_varlen_dense_wrapper(q_blh, k_blh, v_blh)

    if _HAS_FA2:
        return _fa2_dense(q_blh, k_blh, v_blh, dropout_p=0.0, softmax_scale=None, causal=False)

    # PyTorch SDPA (use CUDA flash/mem-efficient kernels when available)
    B, L, H, Dh = q_blh.shape
    q_lbh = q_blh.permute(1, 0, 2, 3).reshape(L, B * H, Dh)
    k_lbh = k_blh.permute(1, 0, 2, 3).reshape(L, B * H, Dh)
    v_lbh = v_blh.permute(1, 0, 2, 3).reshape(L, B * H, Dh)
    # Prefer flash; fall back to mem-efficient; avoid math kernel on CUDA
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False):
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
        h = h.to(self.fc1.weight.dtype)
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
        # FramePack uses gelu-approximate / linear-silu variants; keep GELU-approx for parity
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(inner, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# -----------------------------------------------------------------------------
# RMSNorm (for Q/K parity)
# -----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., D]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        y = x * rms
        if self.weight is not None:
            y = y * self.weight
        return y

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

def _apply_rope_qk(q, k, rope_freqs):
    # q,k: [B,H,L,Dh], rope_freqs: [L, 2*Dh] (fp32)
    B,H,L,Dh = q.shape
    cos, sin = rope_freqs.chunk(2, dim=-1)            # [L,Dh], fp32
    cos = cos.view(1,1,L,Dh)
    sin = sin.view(1,1,L,Dh)

    def _rotate(x):
        x_ = x.float().unflatten(-1, (-1,2))
        x1, x2 = x_.unbind(-1)
        xr = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return xr

    qf = q.float(); kf = k.float()
    q_out = (qf * cos) + (_rotate(qf) * sin)
    k_out = (kf * cos) + (_rotate(kf) * sin)
    return q_out.to(q.dtype), k_out.to(k.dtype)


# -----------------------------------------------------------------------------
# AdaLN (FramePack-style) + Transformer block w/ RoPE and pluggable attention
# -----------------------------------------------------------------------------
class AdaLayerNormZeroSingle(nn.Module):
    """FramePack-style AdaLN Zero (single-stream).
    Given token states x:[B,L,D] and embedding emb:[B,D],
    returns modulated normalized states and a single gate tensor.
    """
    def __init__(self, embedding_dim: int, bias: bool = True):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        # elementwise_affine=False so all scale/shift comes from conditioning
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B,L,D], emb: [B,D]
        B, L, D = x.shape
        xn = self.norm(x)
        # produce shift/scale/gate from embedding and broadcast across sequence
        emb = self.linear(self.silu(emb)).unsqueeze(1)  # [B,1,3D]
        shift, scale, gate = emb.chunk(3, dim=-1)       # [B,1,D] each
        x_mod = xn * (1 + scale) + shift               # [B,L,D]
        return x_mod, gate                              # gate will be broadcast later


class SingleStreamBlock(nn.Module):
    """
    Single-stream Transformer block with **AdaLayerNormZero** gating (FramePack-style):
      1) AdaLNZero -> (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
      2) Attention on modulated tokens; add residual scaled by gate_msa
      3) Post-attn LN -> modulate with (shift_mlp, scale_mlp) -> FeedForward; add residual scaled by gate_mlp
    Also applies RoPE to Q/K and chooses attention backend with FA3>FA2>SAGE2>SDPA.
    """
    def __init__(self, dim, heads=16, mlp_ratio=4.0, qk_norm: str = "rms_norm"):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert self.head_dim * heads == dim, "hidden size must be divisible by heads"
        assert self.head_dim % 2 == 0, "head_dim must be even to use RoPE"

        # AdaLN Zero (produces 6 chunks)
        class AdaLayerNormZero(nn.Module):
            def __init__(self, embedding_dim: int, bias: bool = True):
                super().__init__()
                self.silu = nn.SiLU()
                self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)
                self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
            def forward(self, x: torch.Tensor, emb: torch.Tensor):
                # x:[B,L,D], emb:[B,D]
                x = self.norm(x)
                emb = self.linear(self.silu(emb)).unsqueeze(1)  # [B,1,6D]
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)
                x_msa = x * (1 + scale_msa) + shift_msa
                return x_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        self.adaln = AdaLayerNormZero(dim)

        # Attention path
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj_attn = nn.Linear(dim, dim)
        # Q/K norm parity with FramePack
        if isinstance(qk_norm, str) and qk_norm.lower() in ("rms_norm", "rms"):
            self.q_norm = RMSNorm(self.head_dim, eps=1e-6, affine=False)
            self.k_norm = RMSNorm(self.head_dim, eps=1e-6, affine=False)
        elif (isinstance(qk_norm, str) and qk_norm.lower() in ("layer_norm", "ln")) or (qk_norm is True):
            self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False, eps=1e-6)
            self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # MLP path
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio)

        nn.init.zeros_(self.proj_attn.weight)
        nn.init.zeros_(self.proj_attn.bias)
        nn.init.zeros_(self.mlp.fc2.weight)
        nn.init.zeros_(self.mlp.fc2.bias)
        nn.init.zeros_(self.adaln.linear.weight)
        nn.init.zeros_(self.adaln.linear.bias)

    def forward(self, x, tvec, rope_freqs=None):
        B, L, D = x.shape

        # 1) AdaLN for attention
        x_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln(x, tvec)

        # 2) Attention
        qkv = self.qkv(x_msa)
        q, k, v = qkv.chunk(3, dim=-1)
        def to_bhl(z):
            return z.view(B, L, self.heads, self.head_dim).transpose(1, 2).contiguous()
        q_bhl = self.q_norm(to_bhl(q))
        k_bhl = self.k_norm(to_bhl(k))
        v_bhl = to_bhl(v)
        q_bhl, k_bhl = _apply_rope_qk(q_bhl, k_bhl, rope_freqs)
        # backend expects BLH
        q_blh = q_bhl.transpose(1, 2).contiguous().to(torch.bfloat16)
        k_blh = k_bhl.transpose(1, 2).contiguous().to(torch.bfloat16)
        v_blh = v_bhl.transpose(1, 2).contiguous().to(torch.bfloat16)
        attn_blh = _attn_blh(q_blh, k_blh, v_blh)
        attn = attn_blh.transpose(1, 2).reshape(B, L, D)
        attn = self.proj_attn(attn)

        h = x + gate_msa * attn  # residual 1

        # 3) Post-attn LN -> modulate -> MLP -> gated residual
        h2 = self.norm2(h)
        h2 = h2 * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(h2)
        out = h + gate_mlp * mlp_out  # residual 2
        return out

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
        x = x.to(self.proj.weight.dtype)
        y = self.proj(x)  # [B, L, out_ch * pt*ph*pw]
        y = y.view(B, T, H, W, self.out_ch, self.pt, self.ph, self.pw)  # block grid
        y = y.permute(0,4,1,5,2,6,3,7)  # [B,C,T,pt,H,ph,W,pw]
        y = y.reshape(B, self.out_ch, T*self.pt, H*self.ph, W*self.pw)  # [B,C,T,H,W]
        return y


class AdaLayerNormContinuous(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.fc = nn.Linear(dim, 2 * dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D], emb: [B, D]
        scale, shift = self.fc(self.act(emb)).chunk(2, dim=-1)  # [B, D] each
        x = self.norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x

# -----------------------------------------------------------------------------
# Main transformer (unchanged API; now with RoPE + backend-priority attention)
# + TeaCache support
# -----------------------------------------------------------------------------
class FlowMatchTransformer(nn.Module):
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
        rope_axes_dim: Optional[Tuple[int,int,int]] = (8, 32, 32),  # (DT, DY, DX); if None -> split evenly from head_dim
        # --- Parity knobs ---
        qk_norm: str = "rms_norm",
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

        self.rope = HunyuanVideoRotaryPosEmbed(rope_axes_dim, theta=rope_theta)

        self.t_embed = TimeEmbed(hidden_size)
        self.patch = PatchEmbed3D(latent_channels, hidden_size, patch_size=patch_size)

        self.blocks = nn.ModuleList([
            SingleStreamBlock(hidden_size, heads=heads, mlp_ratio=mlp_ratio, qk_norm=qk_norm)
            for _ in range(depth)
        ])
        # Final projection (velocity)
        self.norm_out = AdaLayerNormContinuous(self.hidden)
        self.final = FinalProjector(self.hidden, self.latent_channels, patch_size=self.patch_size, twh=(8, 32, 32))  # created at runtime after we know token grid

        # ---------------- TeaCache state ----------------
        self._tc_enabled = False
        self._tc_cnt = 0
        self._tc_num_steps = 1
        self._tc_rel_l1_thresh = 0.15
        # default poly from a reference calibration; override via initialize_teacache(...)
        self._tc_poly_coeffs = [7.33226126e+02, -4.01131952e+02, 6.75869174e+01, -3.14987800e+00, 9.61237896e-02]
        self._tc_prev_mod_inp: Optional[torch.Tensor] = None
        self._tc_prev_residual: Optional[torch.Tensor] = None
        self._tc_accum: float = 0.0

    # ---------------- TeaCache helpers ----------------
    @torch.no_grad()
    def initialize_teacache(self, enable_teacache: bool = True, num_steps: int = 25, rel_l1_thresh: float = 0.15, poly_coeffs: Optional[List[float]] = None):
        """
        Enable + configure TeaCache behavior. Call once before an inference trajectory.
        - num_steps: total diffusion/flow steps in your sampler cycle
        - rel_l1_thresh: larger => more skipping (faster, potentially lower quality)
        - poly_coeffs: list of poly coeffs (np.polyfit order-4 recommended) mapping raw rel-L1
                       of the first-block modulated input -> an accumulated "distance" unit.
        """
        self._tc_enabled = enable_teacache
        self._tc_cnt = 0
        self._tc_num_steps = int(max(1, num_steps))
        self._tc_rel_l1_thresh = float(rel_l1_thresh)
        if poly_coeffs is not None:
            self._tc_poly_coeffs = list(map(float, poly_coeffs))
        self._tc_prev_mod_inp = None
        self._tc_prev_residual = None
        self._tc_accum = 0.0

    @torch.no_grad()
    def _tc_poly(self, x: float) -> float:
        # Horner's rule; coeffs are high->low order
        c = self._tc_poly_coeffs
        y = 0.0
        for a in c:
            y = y * x + a
        return float(y)

    @torch.no_grad()
    def _tc_compute_modulated_input(self, tokens: torch.Tensor, tvec: torch.Tensor) -> torch.Tensor:
        b0 = self.blocks[0]
        h, _gate = b0.adaln(tokens, tvec)
        return h

    @torch.no_grad()
    def _tc_should_compute(self, curr_mod_inp: torch.Tensor) -> bool:
        if self._tc_cnt == 0 or self._tc_cnt == self._tc_num_steps - 1 or self._tc_prev_mod_inp is None:
            # always compute on the first and last steps
            self._tc_accum = 0.0
            return True
        # relative L1 between successive modulated inputs
        num = (curr_mod_inp - self._tc_prev_mod_inp).abs().mean().item()
        den = (self._tc_prev_mod_inp.abs().mean().item() + 1e-12)
        curr_rel_l1 = num / den
        self._tc_accum += self._tc_poly(curr_rel_l1)
        if self._tc_accum >= self._tc_rel_l1_thresh:
            self._tc_accum = 0.0
            return True
        return False

    @torch.no_grad()
    def _tc_update_counters(self):
        self._tc_cnt += 1
        if self._tc_cnt == self._tc_num_steps:
            self._tc_cnt = 0

    # --------------------------------------------------
    def _rope_for_grid(self, Tg: int, Hg: int, Wg: int, device) -> Optional[torch.Tensor]:
        return self.rope(Tg, Hg, Wg, device=device)  # [L, 2*Dh]

    def forward(self, zt: torch.Tensor, t: torch.Tensor):
        """
        zt: [B, Cz, n, H, W]
        t : [B] in (0,1]
        """
        B, C, n, H, W = zt.shape
        x, (Tg, Hg, Wg) = self.patch(zt)  # [B, L, D]

        rope_freqs = self._rope_for_grid(Tg, Hg, Wg, zt.device)

        tvec = self.t_embed(t).to(x.dtype)  # [B,D]

        # ---------------- TeaCache path ----------------
        if self._tc_enabled and len(self.blocks) > 0 and not self.training:
            with torch.no_grad():
                curr_mod_inp = self._tc_compute_modulated_input(x, tvec)
                should_calc = self._tc_should_compute(curr_mod_inp)
                self._tc_prev_mod_inp = curr_mod_inp
                self._tc_update_counters()

            if not should_calc and (self._tc_prev_residual is not None):
                # reuse previous residual
                x = x + self._tc_prev_residual
            else:
                ori_x = x
                for blk in self.blocks:
                    x = blk(x, tvec, rope_freqs=rope_freqs)
                self._tc_prev_residual = x - ori_x
        else:
            for blk in self.blocks:
                x = blk(x, tvec, rope_freqs=rope_freqs)

        vel = self.final(x)  # [B,Cz,n,H,W]
        return vel
        