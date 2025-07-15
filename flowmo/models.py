"""Model code for FlowMo.

Sources: https://github.com/feizc/FluxMusic/blob/main/train.py
https://github.com/black-forest-labs/flux/tree/main/src/flux
"""

import ast
import itertools
import math
from dataclasses import dataclass
from typing import List, Tuple

import einops
import torch
from einops import rearrange, repeat
from mup import MuReadout
from torch import Tensor, nn

from flowmo import lookup_free_quantize

MUP_ENABLED = True


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    b, h, l, d = q.shape
    q, k = apply_rope(q, k, pe)

    if torch.__version__ == "2.0.1+cu117":  # tmp workaround
        if d != 64:
            print("MUP is broken in this setting! Be careful!")
            x = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
            )
    else:
        x = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=8.0 / d if MUP_ENABLED else None,
        )
    assert x.shape == q.shape
    x = rearrange(x, "B H L D -> B L (H D)")
    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)],
        dim=-1,
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]

    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def _get_diagonal_gaussian(parameters):
    mean, logvar = torch.chunk(parameters, 2, dim=1)
    logvar = torch.clamp(logvar, -30.0, 20.0)
    return mean, logvar


def _sample_diagonal_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    x = mean + std * torch.randn(mean.shape, device=mean.device)
    return x


def _kl_diagonal_gaussian(mean, logvar):
    var = torch.exp(logvar)
    return 0.5 * torch.sum(torch.pow(mean, 2) + var - 1.0 - logvar, dim=1).mean()


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(t.device)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

        self.lin.weight[dim * 2 : dim * 3].data[:] = 0.0
        self.lin.bias[dim * 2 : dim * 3].data[:] = 0.0
        self.lin.weight[dim * 5 : dim * 6].data[:] = 0.0
        self.lin.bias[dim * 5 : dim * 6].data[:] = 0.0

    def forward(self, vec: Tensor) -> Tuple[ModulationOut, ModulationOut]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(
            self.multiplier, dim=-1
        )
        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = False,
    ):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias
        )
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias
        )

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor):
        pe_single, pe_double = pe
        p = 1
        if vec is None:
            img_mod1, img_mod2 = ModulationOut(0, 1 - p, 1), ModulationOut(0, 1 - p, 1)
            txt_mod1, txt_mod2 = ModulationOut(0, 1 - p, 1), ModulationOut(0, 1 - p, 1)
        else:
            img_mod1, img_mod2 = self.img_mod(vec)
            txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (p + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (p + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe_double)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp(
            (p + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
        )

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp(
            (p + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
        )
        return img, txt


class LastLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        readout_zero_init=False,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        if MUP_ENABLED:
            self.linear = MuReadout(
                hidden_size,
                patch_size * patch_size * out_channels,
                bias=True,
                readout_zero_init=readout_zero_init,
            )
        else:
            self.linear = nn.Linear(
                hidden_size, patch_size * patch_size * out_channels, bias=True
            )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: Tensor, vec) -> Tensor:
        if vec is None:
            pass
        else:
            shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
            x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.norm_final(x)
        x = self.linear(x)
        return x


@dataclass
class FluxParams:
    in_channels: int
    patch_size: int
    context_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    axes_dim: List[int]
    theta: int
    qkv_bias: bool


DIT_ZOO = dict(
    dit_xl_4=dict(
        hidden_size=1152,
        mlp_ratio=4.0,
        num_heads=16,
        axes_dim=[8, 28, 28],
        theta=10_000,
        qkv_bias=True,
    ),
    dit_l_4=dict(
        hidden_size=1024,
        mlp_ratio=4.0,
        num_heads=16,
        axes_dim=[8, 28, 28],
        theta=10_000,
        qkv_bias=True,
    ),
    dit_b_4=dict(
        hidden_size=768,
        mlp_ratio=4.0,
        num_heads=12,
        axes_dim=[8, 28, 28],
        theta=10_000,
        qkv_bias=True,
    ),
    dit_s_4=dict(
        hidden_size=384,
        mlp_ratio=4.0,
        num_heads=6,
        axes_dim=[8, 28, 28],
        theta=10_000,
        qkv_bias=True,
    ),
    dit_mup_test=dict(
        hidden_size=768,
        mlp_ratio=4.0,
        num_heads=12,
        axes_dim=[8, 28, 28],
        theta=10_000,
        qkv_bias=True,
    ),
)


def prepare_idxs(img, code_length, patch_size):
    bs, c, h, w = img.shape

    img_ids = torch.zeros(h // patch_size, w // patch_size, 3, device=img.device)
    img_ids[..., 1] = (
        img_ids[..., 1] + torch.arange(h // patch_size, device=img.device)[:, None]
    )
    img_ids[..., 2] = (
        img_ids[..., 2] + torch.arange(w // patch_size, device=img.device)[None, :]
    )
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    txt_ids = (
        torch.zeros((bs, code_length, 3), device=img.device)
        + torch.arange(code_length, device=img.device)[None, :, None]
    )
    return img_ids, txt_ids


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams, name="", lsg=False):
        super().__init__()

        self.name = name
        self.lsg = lsg
        self.params = params
        self.in_channels = params.in_channels
        self.patch_size = params.patch_size
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )

        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.txt_in = nn.Linear(params.context_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for idx in range(params.depth)
            ]
        )

        self.final_layer_img = LastLayer(
            self.hidden_size, 1, self.out_channels, readout_zero_init=False
        )
        self.final_layer_txt = LastLayer(
            self.hidden_size, 1, params.context_dim, readout_zero_init=False
        )

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        b, c, h, w = img.shape

        img = rearrange(
            img,
            "b c (gh ph) (gw pw) -> b (gh gw) (ph pw c)",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")
        img = self.img_in(img)

        if timesteps is None:
            vec = None
        else:
            vec = self.time_in(timestep_embedding(timesteps, 256))

        txt = self.txt_in(txt)
        pe_single = self.pe_embedder(torch.cat((txt_ids,), dim=1))
        pe_double = self.pe_embedder(torch.cat((txt_ids, img_ids), dim=1))

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, pe=(pe_single, pe_double), vec=vec)

        img = self.final_layer_img(img, vec=vec)
        img = rearrange(
            img,
            "b (gh gw) (ph pw c) -> b c (gh ph) (gw pw)",
            ph=self.patch_size,
            pw=self.patch_size,
            gh=h // self.patch_size,
            gw=w // self.patch_size,
        )

        txt = self.final_layer_txt(txt, vec=vec)
        return img, txt, {"final_txt": txt}


def get_weights_to_fix(model):
    with torch.no_grad():
        for name, module in itertools.chain(model.named_modules()):
            if "double_blocks" in name and isinstance(module, torch.nn.Linear):
                yield name, module.weight


class FlowMo(nn.Module):
    def __init__(self, width, config):
        super().__init__()
        code_length = config.model.code_length
        context_dim = config.model.context_dim
        enc_depth = config.model.enc_depth
        dec_depth = config.model.dec_depth

        patch_size = config.model.patch_size
        self.config = config

        self.image_size = config.data.image_size
        self.patch_size = config.model.patch_size
        self.code_length = code_length
        self.dit_mode = "dit_b_4"
        self.context_dim = context_dim
        self.encoder_context_dim = context_dim * (
            1 + (self.config.model.quantization_type == "kl")
        )

        if config.model.quantization_type == "lfq":
            self.quantizer = lookup_free_quantize.LFQ(
                codebook_size=2**self.config.model.codebook_size_for_entropy,
                dim=self.config.model.codebook_size_for_entropy,
                num_codebooks=1,
                token_factorization=False,
            )

        if self.config.model.enc_mup_width is not None:
            enc_width = self.config.model.enc_mup_width
        else:
            enc_width = width

        encoder_params = FluxParams(
            in_channels=3 * patch_size**2,
            context_dim=self.encoder_context_dim,
            patch_size=patch_size,
            depth=enc_depth,
            **DIT_ZOO[self.dit_mode],
        )
        decoder_params = FluxParams(
            in_channels=3 * patch_size**2,
            context_dim=context_dim + 1,
            patch_size=patch_size,
            depth=dec_depth,
            **DIT_ZOO[self.dit_mode],
        )

        # width=4, dit_b_4 is the usual model
        encoder_params.hidden_size = enc_width * (encoder_params.hidden_size // 4)
        decoder_params.hidden_size = width * (decoder_params.hidden_size // 4)
        encoder_params.axes_dim = [
            (d // 4) * enc_width for d in encoder_params.axes_dim
        ]
        decoder_params.axes_dim = [(d // 4) * width for d in decoder_params.axes_dim]

        self.encoder = Flux(encoder_params, name="encoder")
        self.decoder = Flux(decoder_params, name="decoder")

    @torch.compile
    def encode(self, img):
        b, c, h, w = img.shape

        img_idxs, txt_idxs = prepare_idxs(img, self.code_length, self.patch_size)
        txt = torch.zeros(
            (b, self.code_length, self.encoder_context_dim), device=img.device
        )

        _, code, aux = self.encoder(img, img_idxs, txt, txt_idxs, timesteps=None)

        return code, aux

    def _decode(self, img, code, timesteps):
        b, c, h, w = img.shape

        img_idxs, txt_idxs = prepare_idxs(
            img,
            self.code_length,
            self.patch_size,
        )
        pred, _, decode_aux = self.decoder(
            img, img_idxs, code, txt_idxs, timesteps=timesteps
        )
        return pred, decode_aux

    @torch.compile
    def decode(self, *args, **kwargs):
        return self._decode(*args, **kwargs)

    @torch.compile
    def decode_checkpointed(self, *args, **kwargs):
        # Need to compile(checkpoint), not checkpoint(compile)
        assert not kwargs, kwargs
        return torch.utils.checkpoint.checkpoint(
            self._decode,
            *args,
            # WARNING: Do not use_reentrant=True with compile, it will silently
            # produce incorrect gradients!
            use_reentrant=False,
        )

    @torch.compile
    def _quantize(self, code):
        """
        Args:
            code: [b codelength context dim]

        Returns:
            quantized code of the same shape
        """
        b, t, f = code.shape
        indices = None
        if self.config.model.quantization_type == "noop":
            quantized = code
            aux = {"quantizer_loss": torch.tensor(0.0).to(code.device)}
        elif self.config.model.quantization_type == "kl":
            # colocating features of same token before split is maybe slightly
            # better?
            mean, logvar = _get_diagonal_gaussian(
                einops.rearrange(code, "b t f -> b (f t)")
            )
            code = einops.rearrange(
                _sample_diagonal_gaussian(mean, logvar),
                "b (f t) -> b t f",
                f=f // 2,
                t=t,
            )
            aux = {"quantizer_loss": _kl_diagonal_gaussian(mean, logvar)}
        elif self.config.model.quantization_type == "lfq":
            assert f % self.config.model.codebook_size_for_entropy == 0, f
            code = einops.rearrange(
                code,
                "b t (fg fh) -> b fg (t fh)",
                fg=self.config.model.codebook_size_for_entropy,
            )

            (quantized, entropy_aux_loss, indices), breakdown = self.quantizer(
                code, return_loss_breakdown=True
            )
            assert quantized.shape == code.shape
            quantized = einops.rearrange(quantized, "b fg (t fh) -> b t (fg fh)", t=t)

            quantizer_loss = (
                entropy_aux_loss * self.config.model.entropy_loss_weight
                + breakdown.commitment * self.config.model.commit_loss_weight
            )
            code = quantized
            aux = {"quantizer_loss": quantizer_loss}
            aux["vq_codebook_usage"] = torch.tensor(indices.unique().numel() / self.quantizer.codebook_size * 100)
        else:
            raise NotImplementedError
        return code, indices, aux

    def forward(
        self,
        img,
        noised_img,
        timesteps,
        enable_cfg=True,
    ):
        aux = {}

        code, encode_aux = self.encode(img)

        aux["original_code"] = code

        b, t, f = code.shape

        code, _, quant_aux = self._quantize(code)
        aux.update(quant_aux)

        mask = torch.ones_like(code[..., :1])
        code = torch.concatenate([code, mask], axis=-1)
        code_pre_cfg = code

        if self.config.model.enable_cfg and enable_cfg:
            cfg_mask = (torch.rand((b,), device=code.device) > 0.1)[:, None, None]
            code = code * cfg_mask

        v_est, decode_aux = self.decode(noised_img, code, timesteps)
        aux.update(decode_aux)

        if self.config.model.posttrain_sample:
            aux["posttrain_sample"] = self.reconstruct_checkpoint(code_pre_cfg)

        return v_est, aux

    def reconstruct_checkpoint(self, code):
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
        ):
            bs, *_ = code.shape

            z = torch.randn((bs, 3, self.image_size, self.image_size)).cuda()
            ts = torch.rand((bs, self.config.model.posttrain_sample_k + 1)).cumsum(dim=1).cuda()
            ts = ts - ts[:, :1]
            ts = (ts / ts[:, -1:]).flip(dims=(1,))
            dts = ts[:, :-1] - ts[:, 1:]

            for i, (t, dt) in enumerate((zip(ts.T, dts.T))):
                if self.config.model.posttrain_sample_enable_cfg:
                    mask = (torch.rand((bs,), device=code.device) > 0.1)[
                        :, None, None
                    ].to(code.dtype)
                    code_t = code * mask
                else:
                    code_t = code

                vc, _ = self.decode_checkpointed(z, code_t, t)

                z = z - dt[:, None, None, None] * vc
        return z

    @torch.no_grad()
    def reconstruct(self, images, dtype=torch.bfloat16, code=None):
        """
        Args:
            images in [bchw] [-1, 1]

        Returns:
            images in [bchw] [-1, 1]
        """
        model = self
        config = self.config.eval.sampling

        with torch.autocast(
            "cuda",
            dtype=dtype,
        ):
            bs, c, h, w = images.shape
            if code is None:
                x = images.cuda()
                prequantized_code = model.encode(x)[0].cuda()
                code, _, _ = model._quantize(prequantized_code)

            z = torch.randn((bs, 3, h, w)).cuda()

            mask = torch.ones_like(code[..., :1])
            code = torch.concatenate([code * mask, mask], axis=-1)

            cfg_mask = 0.0
            null_code = code * cfg_mask if config.cfg != 1.0 else None

            samples = rf_sample(
                model,
                z,
                code,
                null_code=null_code,
                sample_steps=config.sample_steps,
                cfg=config.cfg,
                schedule=config.schedule,
            )[-1].clip(-1, 1)
        return samples.to(torch.float32)


def rf_loss(config, model, batch, aux_state):
    x = batch["image"]
    b = x.size(0)

    if config.opt.schedule == "lognormal":
        nt = torch.randn((b,)).to(x.device)
        t = torch.sigmoid(nt)
    elif config.opt.schedule == "fat_lognormal":
        nt = torch.randn((b,)).to(x.device)
        t = torch.sigmoid(nt)
        t = torch.where(torch.rand_like(t) <= 0.9, t, torch.rand_like(t))
    elif config.opt.schedule == "uniform":
        t = torch.rand((b,), device=x.device)
    elif config.opt.schedule.startswith("debug"):
        p = float(config.opt.schedule.split("_")[1])
        t = torch.ones((b,), device=x.device) * p
    else:
        raise NotImplementedError

    t = t.view([b, *([1] * len(x.shape[1:]))])
    z1 = torch.randn_like(x)
    zt = (1 - t) * x + t * z1

    zt, t = zt.to(x.dtype), t.to(x.dtype)

    vtheta, aux = model(
        img=x,
        noised_img=zt,
        timesteps=t.reshape((b,)),
    )

    diff = z1 - vtheta - x
    x_pred = zt - vtheta * t

    diffusion_loss = ((diff) ** 2).mean(dim=list(range(1, len(x.shape))))
    diffusion_loss = diffusion_loss.mean()

    aux["loss_dict"] = {}
    aux["loss_dict"]["diffusion_loss"] = diffusion_loss
    for k in aux.keys():
        if k.endswith('_loss'):
            aux["loss_dict"][k] = aux[k]

    if config.opt.lpips_weight != 0.0:
        aux_loss = 0.0
        if config.model.posttrain_sample:
            x_pred = aux["posttrain_sample"]

        lpips_dist = aux_state["lpips_model"](x, x_pred)
        lpips_dist = (config.opt.lpips_weight * lpips_dist).mean() + aux_loss
        aux["loss_dict"]["lpips_loss"] = lpips_dist

    aux["loss_dict"]["total_loss"] = sum(aux["loss_dict"].values())

    # Copy over other metrics
    for key, value in aux.items():
        if key != 'loss_dict' and key not in aux["loss_dict"] and \
            (isinstance(value, torch.Tensor) and value.numel() == 1):
            aux["loss_dict"][key] = value

    return aux["loss_dict"]["total_loss"], aux


def _edm_to_flow_convention(noise_level):
    # z = x + \sigma z'
    return noise_level / (1 + noise_level)


def rf_sample(
    model,
    z,
    code,
    null_code=None,
    sample_steps=25,
    cfg=2.0,
    schedule="linear",
):
    b = z.size(0)
    if schedule == "linear":
        ts = torch.arange(1, sample_steps + 1).flip(0) / sample_steps
        dts = torch.ones_like(ts) * (1.0 / sample_steps)
    elif schedule.startswith("pow"):
        p = float(schedule.split("_")[1])
        ts = torch.arange(0, sample_steps + 1).flip(0) ** (1 / p) / sample_steps ** (
            1 / p
        )
        dts = ts[:-1] - ts[1:]
    else:
        raise NotImplementedError

    if model.config.eval.sampling.cfg_interval is None:
        interval = None
    else:
        cfg_lo, cfg_hi = ast.literal_eval(model.config.eval.sampling.cfg_interval)
        interval = _edm_to_flow_convention(cfg_lo), _edm_to_flow_convention(cfg_hi)

    images = []
    for i, (t, dt) in enumerate((zip(ts, dts))):
        timesteps = torch.tensor([t] * b).to(z.device)
        vc, decode_aux = model.decode(
            img=z, timesteps=timesteps, code=code
        )

        if null_code is not None and (
            interval is None
            or ((t.item() >= interval[0]) and (t.item() <= interval[1]))
        ):
            vu, _ = model.decode(img=z, timesteps=timesteps, code=null_code)
            vc = vu + cfg * (vc - vu)

        z = z - dt * vc
        images.append(z)
    return images
