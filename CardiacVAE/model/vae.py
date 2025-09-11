# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = [
    'CardiacVAE',
]

CACHE_T = 2

class CausalConv3d(nn.Conv3d):
    """
    Causal 3D conv with:
      • reflect padding in space (H, W)
      • causal padding in time (left only)
    If there is no temporal cache (or it's shorter than needed), we zero-pad the
    remaining temporal-left context. If cache is longer than needed, we trim it.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Conv3d stores padding as (pt, ph, pw). We'll pad manually.
        pt, ph, pw = self.padding
        self._pt = int(pt)
        self._ph = int(ph)
        self._pw = int(pw)

        # Causal left pad in time: match your previous logic (2 * pt).
        # (For kt=3, pt=1 => left pad 2.)
        self._t_left = 2 * self._pt

        # Disable internal conv padding — we control it explicitly.
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        # ---- 1) Spatial reflect pad (H, W) ----
        if self._ph > 0 or self._pw > 0:
            # F.pad expects (wL, wR, hT, hB, dF, dB)
            x = F.pad(x, (self._pw, self._pw, self._ph, self._ph, 0, 0), mode='reflect')

        # ---- 2) Temporal causal pad (left only) ----
        need_left = self._t_left

        if need_left > 0 and cache_x is not None:
            cache_x = cache_x.to(x.device)
            if self._ph > 0 or self._pw > 0:
                cache_x = F.pad(cache_x, (self._pw, self._pw, self._ph, self._ph, 0, 0), mode='reflect')
            # Trim cache to at most what we need (use the most recent frames)
            if cache_x.shape[2] > need_left:
                cache_x = cache_x[:, :, -need_left:, :, :]
            x = torch.cat([cache_x, x], dim=2)
            need_left -= cache_x.shape[2]

        if need_left > 0:
            # 0-pad any remaining temporal-left context (when cache is missing/short)
            x = F.pad(x, (0, 0, 0, 0, need_left, 0), mode='constant', value=0.0)

        return super().forward(x)


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):

    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] != 'Rep':
                        # cache last frame of last two chunk
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x
                        ],
                                            dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = torch.cat([
                            torch.zeros_like(cache_x).to(cache_x.device),
                            cache_x
                        ],
                                            dim=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -1:, :, :].clone()
                    # if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx]!='Rep':
                    #     # cache last frame of last two chunk
                    #     cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        #conv_weight.data[:,:,-1,1,1] = init_matrix * 0.5
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  #* 0.5
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        #init_matrix = repeat(init_matrix, 'o ... -> (o 2) ...').permute(1,0,2).contiguous().reshape(c1,c2)
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3,
                                         -1).permute(0, 1, 3,
                                                     2).contiguous().chunk(
                                                         3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity


class Encoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[
                    i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        # self.temperal_upsample = temperal_downsample[::-1]
        self.temperal_upsample = [False, True]

        self.temporal_downsample_factor = 2
        if self.temperal_downsample[1]:
            self.temporal_downsample_factor = 4
        if self.temperal_downsample[2]:
            self.temporal_downsample_factor = 8

        # modules
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def _encode_one(self, x, scale):
        """
        x: [B, C=2 or 3, T, H, W]
        Returns: mu, log_var of shape [B, z_dim, n, H/8, W/8]
        where n = 1 + (T-1)//temporal_downsample_factor
        """
        self.clear_cache()

        B, C, T, H, W = x.shape
        f = self.temporal_downsample_factor            # 2, 4, or 8 depending on config
        assert (T - 1) % f == 0, f"Input T={T} not compatible with causal factor f={f} (need T=2*(n-1)+1)."
        n = 1 + (T - 1) // f

        # stream: 1 frame, then (n-1) chunks of length f
        # chunk 0: 1 frame
        self._enc_conv_idx = [0]
        out = self.encoder(
            x[:, :, :1, :, :],
            feat_cache=self._enc_feat_map,
            feat_idx=self._enc_conv_idx
        )

        # subsequent chunks: f frames each
        for i in range(1, n):
            self._enc_conv_idx = [0]
            start = 1 + (i - 1) * f
            stop  = 1 + i * f
            out_i = self.encoder(
                x[:, :, start:stop, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx
            )
            out = torch.cat([out, out_i], dim=2)       # concat along time

        # project to (mu, logvar)
        mu, log_var = self.conv1(out).chunk(2, dim=1)

        # apply robust channel scaling (mean/std in scale)
        if isinstance(scale[0], torch.Tensor):
            mean   = scale[0].detach().clone().view(1, self.z_dim, 1, 1, 1)
            invstd = scale[1].detach().clone().view(1, self.z_dim, 1, 1, 1)
            mu = (mu - mean) * invstd
        else:
            mu = (mu - scale[0]) * scale[1]

        self.clear_cache()
        return mu, log_var

    def _decode_one(self, z, scale):
        """
        z: [B, z_dim, n, H/8, W/8]  (latent sequence length n)
        Returns: x_hat [B, C_in, T, H, W] with T = 2*(n-1)+1
        """
        self.clear_cache()

        # invert robust scaling
        if isinstance(scale[0], torch.Tensor):
            mean   = scale[0].detach().clone().view(1, self.z_dim, 1, 1, 1)
            invstd = scale[1].detach().clone().view(1, self.z_dim, 1, 1, 1)
            z = z / invstd + mean
        else:
            z = z / scale[1] + scale[0]

        n = z.shape[2]
        x = self.conv2(z)

        # decode each latent frame causally (one latent frame at a time)
        outs = []
        for i in range(n):
            self._conv_idx = [0]
            out_i = self.decoder(
                x[:, :, i:i+1, :, :],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx
            )
            outs.append(out_i)

        out = torch.cat(outs, dim=2)  # concatenates the reconstructed time stream
        self.clear_cache()
        return out

    # ---------- public API (now accept tensor or list) ----------
    def encode(self, xs, scale):
        """
        xs: Tensor [B, C, T, H, W] OR list of Tensors [C, T, H, W] or [B, C, T, H, W] (varying sizes allowed).
        Returns:
          - if input was Tensor: (mu, log_var)
          - if input was list: list of (mu, log_var) with per-item shapes
        """
        if isinstance(xs, (list, tuple)):
            out = []
            for x in xs:
                if x.dim() == 4:  # [C,T,H,W] -> add batch dim
                    x = x.unsqueeze(0)
                mu, log_var = self._encode_one(x, scale)
                out.append((mu, log_var))
            return out
        else:
            # single tensor path
            return self._encode_one(xs, scale)

    def decode(self, zs, scale):
        """
        zs: Tensor [B, z_dim, n, H/8, W/8] OR list of Tensors with varying (n,H,W).
        Returns:
          - if input was Tensor: x_hat Tensor
          - if input was list: list of x_hat Tensors
        """
        if isinstance(zs, (list, tuple)):
            outs = []
            for z in zs:
                if z.dim() == 4:  # [z_dim, n, H/8, W/8] -> add batch dim
                    z = z.unsqueeze(0)
                out = self._decode_one(z, scale)
                outs.append(out)
            return outs
        else:
            return self._decode_one(zs, scale)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        #cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


class CardiacVAE(WanVAE_):
    """
    Causal 3D VAE for cardiac CINE MRI (complex, 2 channels: real/imag).
    - Spatial compression: 8x
    - Temporal compression: 2x (causal; 3 input frames -> 2 latent frames)
    - Latents are encouraged to be Gaussian across channels at each (t,h,w),
      while allowing spatial/temporal correlations.
    """
    def __init__(self,
                 in_channels: int = 2,     # complex MRI: [real, imag]
                 z_dim: int = 16,
                 dim: int = 128,
                 dim_mult=(1, 2, 4),
                 num_res_blocks: int = 3,
                 attn_scales=(128),
                 dropout: float = 0.0):
        # Only 2x temporal downsample (first stage); spatial 4x via dim_mult
        temperal_downsample = [True, False, False]

        super().__init__(dim=dim,
                         z_dim=z_dim,
                         dim_mult=list(dim_mult),
                         num_res_blocks=num_res_blocks,
                         attn_scales=list(attn_scales),
                         temperal_downsample=temperal_downsample,
                         dropout=dropout)

        # --- Input/Output channel overrides (keep causal convs) ---
        # encoder first conv: accept 2ch complex input (not 3ch RGB)
        self.encoder.conv1 = CausalConv3d(in_channels, dim, kernel_size=3, padding=1)
        # decoder final conv: output 2ch complex reconstruction
        self.decoder.head[-1] = CausalConv3d(dim, in_channels, kernel_size=3, padding=1)

        # Robust per-channel scaling placeholders (set with set_scale(...))
        self.register_buffer("scale_mean", torch.zeros(z_dim))
        self.register_buffer("scale_invstd", torch.ones(z_dim))

    # --- Robust dataset scaling for z (optional) ---
    @torch.no_grad()
    def set_scale(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Provide robust per-channel mean/std for the latent scaling used by WanVAE_.
        mean,std: shape [z_dim]
        """
        device = next(self.parameters()).device
        self.scale_mean = mean.to(device)
        self.scale_invstd = (1.0 / std.clamp_min(1e-3)).to(device)

    # Override encode/decode thin wrappers to pass scale and apply whiteners
    def encode(self, xs):
        """
        Accepts Tensor or list of Tensors. Applies parent streaming encode with scaling,
        then applies 1x1x1 whitening to (mu, log_var).
        """
        scale = [self.scale_mean, self.scale_invstd]

        if isinstance(xs, (list, tuple)):
            pairs = super().encode(xs, scale=scale)  # list of (mu, log_var)
            out = []
            for (mu, log_var) in pairs:
                out.append((mu, log_var))
            return out
        else:
            mu, log_var = super().encode(xs, scale=scale)
            return mu, log_var

    def decode(self, zs):
        """
        Accepts Tensor or list of Tensors and uses parent decode with scaling.
        (No whitening here.)
        """
        scale = [self.scale_mean, self.scale_invstd]
        return super().decode(zs, scale=scale)

    # Add this to CardiacVAE
    @torch.no_grad()
    def encode_raw_mu(self, x):
        # Bypass scaling: pass scale=[0,1] and skip whiteners
        # We call the parent encode directly and DO NOT apply self.mu_whitener/logv_whitener
        mu, log_var = super(CardiacVAE, self).encode(x, scale=[torch.zeros(self.z_dim, device=x.device),
                                                            torch.ones(self.z_dim, device=x.device)])
        return mu  # [B, C=z_dim, T', H', W']

    def forward(self, xs, op: str | None = None):
        """
        DDP-safe entry point.

        - op == "encode": returns same as self.encode(xs)
        - op == "decode": returns same as self.decode(xs)
        - op is None (default): full pass -> returns (xhats, mus, logvs, zs)
            * If xs is a list (ragged), each of these is a list with matching lengths.
            * If xs is a tensor, each is a tensor.
        """
        if op == "encode":
            return self.encode(xs)
        if op == "decode":
            return self.decode(xs)

        # Full end-to-end pass (single forward for DDP)
        pairs = self.encode(xs)  # tensor or list of (mu, logv)

        # Reparameterize (preserve list/tensor structure)
        def _reparam(mu, logv):
            logv = logv.clamp(-30, 20)
            std = torch.exp(0.5 * logv)
            return mu + std * torch.randn_like(std)

        if isinstance(pairs, (list, tuple)):
            mus, logvs = zip(*pairs)
            zs = [_reparam(mu, logv) for (mu, logv) in pairs]
            xhats = self.decode(zs)
            return xhats, list(mus), list(logvs), zs
        else:
            mu, logv = pairs
            z = _reparam(mu, logv)
            xhat = self.decode(z)
            return xhat, mu, logv, z
