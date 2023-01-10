"""
Modified from code from https://github.com/lucidrains/muse-maskgit-pytorch/

Basically, maskgit but without text guidance and the corresponding cross attention
"""

import math

import torch
import torch.nn.functional as F
from torch import nn, einsum

import torchvision.transforms as T

from typing import Callable, Optional, List

from einops import rearrange, repeat

from vqvae import VQ_VAE

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# classes


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = torch.zeros(dim, device=torch.device("cuda"))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class GEGLU(nn.Module):
    """https://arxiv.org/abs/2002.05202"""

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return gate * F.gelu(x)


def FeedForward(dim, mult=4):
    """https://arxiv.org/abs/2110.09456"""

    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        LayerNorm(inner_dim),
        nn.Linear(inner_dim, dim, bias=False),
    )


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(2, heads, 1, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        x,
    ):
        h = self.heads

        x = self.norm(x)

        kv_input = x

        q, k, v = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1))

        q = q * self.scale

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        nk, nv = self.null_kv
        nk, nv = map(lambda t: repeat(t, "h 1 d -> b h 1 d", b=x.shape[0]), (nk, nv))

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        attn = F.softmax(sim, dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TransformerBlocks(nn.Module):
    def __init__(self, *, dim, depth, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = LayerNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x

            x = ff(x) + x

        return self.norm(x)


# transformer - it's all we need


class Transformer(nn.Module):
    def __init__(self, *, num_tokens, dim, seq_len, **kwargs):
        super().__init__()
        self.mask_id = num_tokens

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens + 1, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.seq_len = seq_len

        self.transformer_blocks = TransformerBlocks(dim=dim, **kwargs)
        self.norm = LayerNorm(dim)

        self.to_logits = nn.Linear(dim, num_tokens, bias=False)

    def forward(
        self,
        x,
        return_embed=False,
    ):
        device, _, n = x.device, *x.shape
        # assert n <= self.seq_len

        # embed tokens

        x = self.token_emb(x)
        x = x + self.pos_emb(torch.arange(n, device=device))

        x = self.transformer_blocks(x)

        if return_embed:
            return x

        logits = self.to_logits(x)

        return logits


# classifier free guidance functions


def uniform(shape, min=0, max=1, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def prob_mask_like(shape, prob, device=None):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return uniform(shape, device=device) < prob


# sampling helpers


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(2, ind, val)
    return probs


# noise schedules


def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)


# main maskgit classes


class MaskGit(nn.Module):
    def __init__(
        self,
        transformer: Transformer,
        noise_schedule: Callable = cosine_schedule,
    ):
        super().__init__()

        self.transformer = transformer

        self.mask_id = transformer.mask_id
        self.noise_schedule = noise_schedule

    def save(self, path):
        torch.save(self.state_dict(), path)

    def generate(
        self,
        batch_size: int,
        fmap_size,
        temperature=1.0,
        topk_filter_thres=0.9,
        can_remask_prev_masked=False,
        timesteps=18,  # ideal number of steps is 18 in maskgit paper
    ):

        # begin with all image token ids masked

        device = next(self.parameters()).device

        seq_len = fmap_size**2

        shape = (batch_size, seq_len)

        ids = torch.full(shape, self.mask_id, dtype=torch.long, device=device)
        scores = torch.randn(shape, dtype=torch.float32, device=device).softmax(-1)

        starting_temperature = temperature

        for timestep, steps_until_x0 in zip(
            torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))
        ):

            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked = max(int((rand_mask_prob * seq_len).item()), 1)

            masked_indices = scores.topk(num_token_masked, dim=-1).indices

            ids = ids.scatter(1, masked_indices, self.mask_id)

            logits = self.transformer.forward(
                ids,
            )

            filtered_logits = top_k(logits, topk_filter_thres)

            temperature = starting_temperature * (
                steps_until_x0 / timesteps
            )  # temperature is annealed

            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)

            is_mask = ids == self.mask_id

            ids = torch.where(is_mask, pred_ids, ids)

            scores = 1 - logits.gather(2, pred_ids[..., None])
            scores = rearrange(scores, "... 1 -> ...")

            if not can_remask_prev_masked:
                # without doing MLM type 15% random or non-masked predictions
                # non-masked tokens may not get correct logits (scores)
                # but not sure

                scores = scores.masked_fill(~is_mask, -1e5)

        # get ids

        ids = rearrange(ids, "b (i j) -> b i j", i=fmap_size, j=fmap_size)

        return ids

    def forward(
        self,
        ids: torch.Tensor,
        ignore_index=-1,
    ):

        ids = rearrange(ids, "b ... -> b (...)")

        batch, seq_len = ids.shape
        device = ids.device

        # prepare mask

        rand_time = uniform((batch,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (seq_len * rand_mask_probs).round().clamp(min=1)

        batch_randperm = torch.rand((batch, seq_len), device=device).argsort(dim=-1)
        mask = batch_randperm < rearrange(num_token_masked, "b -> b 1")

        labels = torch.where(mask, ids, ignore_index)

        # get loss

        logits = self.transformer(ids)
        logits = rearrange(logits, "b n c -> b c n")

        ce_loss = F.cross_entropy(logits, labels, ignore_index=ignore_index)

        return ce_loss
