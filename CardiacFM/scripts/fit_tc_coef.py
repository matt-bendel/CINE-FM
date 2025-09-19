#!/usr/bin/env python3
"""
Fit TeaCache rescale polynomial coefficients for LatentFlowMatchTransformer.

It collects pairs (x_i, y_i):
  x_i = rel_L1( mod_inp(t_i) - mod_inp(t_{i-1}) )
  y_i = rel_L1( vel(t_i)      - vel(t_{i-1}) )
And fits y ≈ P(x) (degree 4).

Usage example:
  python teacache_fit.py \
    --ckpt none \
    --latent-ch 16 --n 2 --H 64 --W 64 \
    --steps 25 --batch 2 \
    --outfile teacache_poly.json

If --ckpt is 'none', random weights are used (for smoke tests). Prefer your trained weights.
"""
import argparse
import json
import numpy as np
import torch

# import your model class
from models.latent_fm_transformer import LatentFlowMatchTransformer


def rel_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    num = (a - b).abs().mean().item()
    den = (b.abs().mean().item() + 1e-12)
    return num / den


@torch.no_grad()
def compute_mod_inp(model: LatentFlowMatchTransformer, zt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    x, (Tg, Hg, Wg) = model.patch(zt)
    tvec = model.t_embed(t).to(x.dtype)
    # Use the first block's AdaLNZero output (pre-attention modulated tokens)
    b0 = model.blocks[0]
    h, *_ = b0.adaln(x, tvec)
    return h


@torch.no_grad()
def forward_vel(model: LatentFlowMatchTransformer, zt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return model(zt, t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default='/storage/matt_models/latent_fm/flowmatch/step_0100000/state.pt', help='path to torch ckpt or "none"')
    p.add_argument('--latent-ch', type=int, default=16)
    p.add_argument('--n', type=int, default=4)
    p.add_argument('--H', type=int, default=40)
    p.add_argument('--W', type=int, default=40)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--outfile', type=str, default='teacache_poly.json')
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = LatentFlowMatchTransformer(latent_channels=args.latent_ch).to(device)
    model.eval()

    if args.ckpt.lower() != 'none':
        sd = torch.load(args.ckpt, map_location='cpu')
        sd = sd.get('model', sd)
        model.load_state_dict(sd, strict=False)
        print('[TeaCacheFit] Loaded weights from', args.ckpt)

    B, C, n, H, W = args.batch, args.latent_ch, args.n, args.H, args.W
    zt = torch.randn(B, C, n, H, W, device=device, dtype=torch.float32)

    # timesteps: uniform in (0,1], sorted ascending (or your scheduler’s t grid)
    ts = torch.linspace(1.0/args.steps, 1.0, args.steps, device=device)

    xs, ys = [], []

    # warm first
    t_prev = ts[0].expand(B)
    mod_prev = compute_mod_inp(model, zt, t_prev)
    vel_prev = forward_vel(model, zt, t_prev)

    for i in range(1, args.steps):
        t_cur = ts[i].expand(B)
        mod_cur = compute_mod_inp(model, zt, t_cur)
        vel_cur = forward_vel(model, zt, t_cur)

        x = rel_l1(mod_cur, mod_prev)
        y = rel_l1(vel_cur, vel_prev)
        xs.append(x)
        ys.append(y)

        mod_prev = mod_cur
        vel_prev = vel_cur

    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)

    # robust guard: discard NaNs/infs if any
    m = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[m], ys[m]

    # degree-4 poly fit (high->low order coefficients)
    coeffs = np.polyfit(xs, ys, deg=4).tolist()

    out = {
        'poly_order': 4,
        'coeffs_high_to_low': coeffs,
        'num_points': int(xs.size)
    }
    with open(args.outfile, 'w') as f:
        json.dump(out, f, indent=2)

    print('[TeaCacheFit] Saved coeffs to', args.outfile)
    print('[TeaCacheFit] Coeffs (high->low):')
    for c in coeffs:
        print(f'  {c:.10e}')

    # quick advice on threshold
    p10 = np.percentile(ys, 10)
    p50 = np.percentile(ys, 50)
    p90 = np.percentile(ys, 90)
    print(f'[TeaCacheFit] y (relΔvel) percentiles: 10%={p10:.4e} 50%={p50:.4e} 90%={p90:.4e}')
    print('[TeaCacheFit] Start with rel_l1_thresh in [0.10, 0.15] and tune for your sampler.')

if __name__ == '__main__':
    main()