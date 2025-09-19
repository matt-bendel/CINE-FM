# CardiacFM/sampler/flow_match_uni_pc.py
# Better Flow Matching UniPC by Lvmin Zhang
# (c) 2025
# CC BY-SA 4.0

import time
import torch
from tqdm.auto import tqdm


def expand_dims(v, dims):
    return v[(...,) + (None,) * (dims - 1)]


def _safe_solve(R32, b32, eps=1e-6):
    # Try exact solve; if singular, Tikhonov regularize.
    try:
        return torch.linalg.solve(R32, b32)
    except Exception:
        I = torch.eye(R32.size(-1), device=R32.device, dtype=R32.dtype)
        return torch.linalg.solve(R32 + eps * I, b32)


class FlowMatchUniPC:
    def __init__(self, model, extra_args=None, variant='bh1'):
        self.model = model
        self.variant = variant
        self.extra_args = extra_args if isinstance(extra_args, dict) else {}

    def model_fn(self, x, t):
        out = self.model(x, t, **self.extra_args)
        return out.to(x.dtype)

    def update_fn(self, x, model_prev_list, t_prev_list, t, order):
        """
        Time-domain scalars in float32 for stability; multiply into bf16 tensors only at the edges.
        """
        assert order <= len(model_prev_list)
        dims   = x.dim()
        device = x.device
        xdtype = x.dtype

        # Promote times to float32
        t_prev_0_f32 = t_prev_list[-1].to(torch.float32)
        t_f32        = t.to(torch.float32)

        lambda_prev0 = -torch.log(t_prev_0_f32)   # [B], fp32
        lambda_t     = -torch.log(t_f32)          # [B], fp32
        h            = (lambda_t - lambda_prev0)  # [B], fp32
        hh           = (-h[0])                    # scalar fp32

        # φ terms in fp32
        h_phi_1 = torch.expm1(hh)
        denom   = hh if abs(float(hh)) > 1e-6 else (hh.sign() * 1e-6 + (hh == 0) * 1e-6)
        h_phi_k = h_phi_1 / denom - 1.0

        if self.variant == 'bh1':
            B_h = hh
        elif self.variant == 'bh2':
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError('Bad variant!')

        # Prior model (tensor dtype stays as xdtype)
        model_prev_0 = model_prev_list[-1].to(xdtype)

        # Build rks, D1s with fp32 rks but D1s in xdtype
        rks, D1s_list = [], []
        for i in range(1, order):
            t_prev_i_f32 = t_prev_list[-(i + 1)].to(torch.float32)
            model_prev_i = model_prev_list[-(i + 1)].to(xdtype)

            lambda_previ = -torch.log(t_prev_i_f32)
            rk = ((lambda_previ - lambda_prev0) / h)[0]  # scalar fp32
            rks.append(rk)
            D1s_list.append(((model_prev_i - model_prev_0) / rk.to(xdtype)).to(xdtype))

        rks.append(torch.tensor(1.0, device=device, dtype=torch.float32))
        rks = torch.stack(rks)  # [order], fp32

        # Build R, b in fp32 and solve in fp32
        R_rows, b_list = [], []
        factorial_i = 1.0
        for i in range(1, order + 1):
            R_rows.append(torch.pow(rks, i - 1))   # [order], fp32
            b_list.append(h_phi_k * factorial_i / B_h)
            factorial_i *= (i + 1)
            denom2 = hh if abs(float(hh)) > 1e-6 else denom
            h_phi_k = h_phi_k / denom2 - 1.0 / factorial_i

        R32 = torch.stack(R_rows)            # [order, order], fp32
        b32 = torch.stack(b_list)            # [order], fp32

        use_predictor = len(D1s_list) > 0
        D1s = torch.stack(D1s_list, dim=1) if use_predictor else None  # [..., order-1], xdtype

        if use_predictor:
            if order == 2:
                rhos_p32 = torch.tensor([0.5], device=device, dtype=torch.float32)
            else:
                rhos_p32 = _safe_solve(R32[:-1, :-1], b32[:-1])
        else:
            rhos_p32 = None

        rhos_c32 = torch.tensor([0.5], device=device, dtype=torch.float32) if order == 1 else _safe_solve(R32, b32)

        # Predictor step (cast scalars right before multiply)
        scale_xt   = (t_f32 / t_prev_0_f32)  # [B], fp32
        x_t_ = expand_dims(scale_xt, dims).to(xdtype) * x - expand_dims(h_phi_1, dims).to(xdtype) * model_prev_0

        if use_predictor:
            rhos_p = rhos_p32.to(xdtype)
            pred_res = torch.tensordot(D1s, rhos_p, dims=([1], [0]))
        else:
            pred_res = 0

        x_t = x_t_ - expand_dims(B_h, dims).to(xdtype) * pred_res

        # Model eval
        model_t = self.model_fn(x_t, t).to(xdtype)

        # Corrector
        rhos_c = rhos_c32.to(xdtype)
        if D1s is not None:
            corr_res = torch.tensordot(D1s, rhos_c[:-1], dims=([1], [0]))
        else:
            corr_res = 0

        D1_t = (model_t - model_prev_0)
        x_t  = x_t_ - expand_dims(B_h, dims).to(xdtype) * (corr_res + rhos_c[-1] * D1_t)

        return x_t, model_t

    def sample(self, x, sigmas, callback=None, disable_pbar=False):
        total = len(sigmas) - 1
        order = min(3, total)
        model_prev_list, t_prev_list = [], []

        pbar = None
        if not disable_pbar:
            pbar = tqdm(total=total, desc=f"UniPC({self.variant})", dynamic_ncols=True, leave=False)

        last = time.perf_counter()
        for i in range(total):
            # Keep t in float32, expanded to batch
            vec_t = sigmas[i].to(torch.float32).expand(x.shape[0])

            if i == 0:
                model_prev_list = [self.model_fn(x, vec_t).to(x.dtype)]
                t_prev_list     = [vec_t]
            elif i < order:
                init_order = i
                x, model_x = self.update_fn(x, model_prev_list, t_prev_list, vec_t, init_order)
                model_prev_list.append(model_x.to(x.dtype))
                t_prev_list.append(vec_t)
            else:
                x, model_x = self.update_fn(x, model_prev_list, t_prev_list, vec_t, order)
                model_prev_list.append(model_x.to(x.dtype))
                t_prev_list.append(vec_t)

            model_prev_list = model_prev_list[-order:]
            t_prev_list     = t_prev_list[-order:]

            if callback is not None:
                callback({'x': x, 'i': i, 'denoised': model_prev_list[-1]})

            if pbar is not None:
                now = time.perf_counter()
                pbar.set_postfix({'sec/it': f'{(now - last):.3f}', 't': f'{float(vec_t[0]):.6f}'})
                last = now
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        return model_prev_list[-1]


def sample_unipc(model, noise, sigmas, extra_args=None, callback=None, disable=False, variant='bh1'):
    assert variant in ['bh1', 'bh2']
    return FlowMatchUniPC(model, extra_args=(extra_args or {}), variant=variant).sample(
        noise, sigmas=sigmas, callback=callback, disable_pbar=disable
    )

