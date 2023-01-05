import math
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as path
import sys

import bpdb

dirname = path.dirname(__file__)
sys.path.insert(0, path.join(dirname, "./ascii-dataset/"))
from dataset import AsciiArtDataset
import ascii_util


def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f"Error: {x.max().item()} >= {num_classes}"
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def multinomial_kl(log_prob1, log_prob2):  # compute KL loss on log_prob
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl


def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def extract(a, t, x_shape):
    """ """
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def alpha_schedule(
    # T
    n_time_step,
    # vqvae K param
    N=100,
    att_1=0.99999,
    att_T=0.000009,
    # gamma 1
    ctt_1=0.000009,
    # gamma T
    ctt_T=0.99999,
    device=torch.device("cpu"),
):
    att = torch.arange(0, n_time_step, device=device) / (n_time_step - 1) * (att_T - att_1) + att_1
    att = torch.concat((torch.Tensor([1]).to(device), att))
    at = att[1:] / att[:-1]
    ctt = torch.arange(0, n_time_step, device=device) / (n_time_step - 1) * (ctt_T - ctt_1) + ctt_1
    ctt = torch.concat((torch.Tensor([0]).to(device), ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1 - one_minus_ct
    bt = (1 - at - ct) / N
    att = torch.concat((att[1:], torch.Tensor([1]).to(device)))
    ctt = torch.concat((ctt[1:], torch.Tensor([0]).to(device)))
    btt = (1 - att - ctt) / N
    return at, bt, ct, att, btt, ctt


class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps, dim, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd, diffusion_step):
        super().__init__()
        self.emb = SinusoidalPosEmb(diffusion_step, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)
        self.diff_step = diffusion_step

    def forward(self, x, timestep):
        if timestep[0] >= self.diff_step:
            _emb = self.emb.weight.mean(dim=0, keepdim=True).repeat(len(timestep), 1)
            emb = self.linear(self.silu(_emb)).unsqueeze(1)
        else:
            emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


class DiffusionTransformerBlock(nn.Module):
    def __init__(self, input_size, feature_size, n_diff_step):
        super().__init__()
        self.adaln = AdaLayerNorm(input_size, n_diff_step)
        self.transformer = nn.Transformer(feature_size, batch_first=True)

    def forward(self, _input):
        """
        Predicts x_0 from x_t
        both are one hot
        """
        x_t, t = _input
        bs,k = x_t.shape[0], x_t.shape[1]
        x_t = x_t.reshape(bs, k, -1)
        a_l = self.adaln(x_t, t).movedim(1,2)
        bpdb.set_trace()
        out = self.transformer(a_l, a_l)
        return out


class DiffusionTransformer(pl.LightningModule):
    def __init__(
        self,
        # Side resolution of square input, ex: for a 32 x 32 discrete
        # latent representation, input_size is 32
        input_res,
        # K parameter used by vqvae. The number of codebooks used in
        # the vqvae
        n_k_classes,
        vqvae,
        n_timesteps=100,
        lr=4e-5,
        batch_size=8,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.lr = lr
        self.n_timesteps = n_timesteps
        self.input_res = input_res
        self.input_seq_length = input_res**2
        self.n_classes = n_k_classes
        self.ce_loss = nn.CrossEntropyLoss()
        self.vqvae = vqvae

        at, bt, ct, att, btt, ctt = alpha_schedule(self.n_timesteps, N=n_k_classes - 1, device=torch.device("cuda"))

        self.log_at = torch.log(at)
        self.log_bt = torch.log(bt)
        self.log_ct = torch.log(ct)
        self.log_cumprod_at = torch.log(att)
        self.log_cumprod_bt = torch.log(btt)
        self.log_cumprod_ct = torch.log(ctt)

        self.log_1_min_ct = log_1_min_a(self.log_ct)
        self.log_1_min_cumprod_ct = log_1_min_a(self.log_cumprod_ct)

        self.diffusion_model = nn.Sequential(
            DiffusionTransformerBlock(self.input_seq_length, n_k_classes, self.n_timesteps),
        )

        # Not sure what this does in the loss function
        self.diffusion_acc_list = [0] * self.n_timesteps
        self.diffusion_keep_list = [0] * self.n_timesteps

        self.mask_weight = [1, 1]

    def sample_time(self, b):
        """
        Uniform t sample
        """
        t = torch.randint(0, self.n_timesteps, (b,), device=self.device).long()
        pt = torch.ones_like(t).float() / self.n_timesteps
        return t, pt

    def forward(self, input_sequence, timestep):
        """
        Returns x_0
        """
        # Pass the input sequence through the transformer model
        output_sequence = self.diffusion_model(input_sequence, timestep)

        return output_sequence

    def sample_prior_x_t(self):
        """
        Equation 10
        p(xT ) = transpose([βT , βT , · · · , βT , γT])
        """

    def q_pred(self, log_x_0, t):
        """
        returns log(q(xt|x0))
        Equation 4 of Vector Quantized Diffusion Model for Text-to-Image Synthesis:
        q(xt|x0) = transpose(v) (xt) (Qt * Qt-1 ... * Q1 ) v (x0)
        """
        # log_x_start can be onehot or not
        t = (t + (self.n_timesteps + 1)) % (self.n_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_0.shape)  # at~
        log_cumprod_bt = extract(self.log_cumprod_bt, t, log_x_0.shape)  # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_0.shape)  # ct~
        log_1_min_cumprod_ct = extract(
            self.log_1_min_cumprod_ct, t, log_x_0.shape
        )  # 1-ct~

        log_probs = torch.cat(
            [
                log_add_exp(log_x_0[:, :-1, :] + log_cumprod_at, log_cumprod_bt),
                log_add_exp(log_x_0[:, -1:, :] + log_1_min_cumprod_ct, log_cumprod_ct),
            ],
            dim=1,
        )

        return log_probs

    def log_sample_categorical(
        self, logits
    ):  # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.n_classes)
        return log_sample

    def p_pred(
        self, log_x, t
    ):  # if x0, first p(x0|xt), than sum(q(xt-1|xt,x0)*p(x0|xt))
        log_x_recon = self.predict_x_0(log_x, t)
        log_model_pred = self.q_posterior(log_x_0=log_x_recon, log_x_t=log_x, t=t)
        return log_model_pred, log_x_recon

    @torch.no_grad()
    def p_sample(
        self,
        log_x_t,
        t,
        # prior_rule: 0 for VQ-Diffusion v1, 1 for only high-quality inference, 2 for purity prior
        prior_rule=0,
        prior_weight=0,
        max_sample_per_step=1024,
        to_sample=None,
        sampled=None,
    ):
        """
        Equation 11 of Vector Quantized Diffusion Model for Text-to-Image Synthesis:
        p_θ(x_{t−1}|x_t) = sum from \hat x_0 = 1 to K of q(x_{t−1}|x_t, \hat x_0) p_θ(\hat x_0|x_t)
        """
        model_log_prob, log_x_recon = self.p_pred(log_x_t, t)
        if t[0] > 0 and prior_rule > 0 and to_sample is not None:
            log_x_idx = log_x_t.argmax(1)

            if self.prior_rule == 1:
                score = torch.ones((log_x_t.shape[0], log_x_t.shape[2])).to(
                    log_x_t.device
                )
            elif self.prior_rule == 2:
                score = torch.exp(log_x_recon).max(dim=1).values.clamp(0, 1)
                score /= score.max(dim=1, keepdim=True).values + 1e-10

            if self.prior_rule != 1 and prior_weight > 0:
                # probability adjust parameter, prior_weight: 'r' in Equation.11 of Improved VQ-Diffusion
                prob = ((1 + score * prior_weight).unsqueeze(1) * log_x_recon).softmax(
                    dim=1
                )
                prob = prob.log().clamp(-70, 0)
            else:
                prob = log_x_recon

            out = self.log_sample_categorical(prob)
            out_idx = out.argmax(1)

            out2_idx = log_x_idx.clone()
            _score = score.clone()
            if _score.sum() < 1e-6:
                _score += 1
            _score[log_x_idx != self.n_classes - 1] = 0

            for i in range(log_x_t.shape[0]):
                n_sample = min(to_sample - sampled[i], max_sample_per_step)
                if to_sample - sampled[i] - n_sample == 1:
                    n_sample = to_sample - sampled[i]
                if n_sample <= 0:
                    continue
                sel = torch.multinomial(_score[i], n_sample)
                out2_idx[i][sel] = out_idx[i][sel]
                sampled[i] += (
                    (out2_idx[i] != self.n_classes - 1).sum()
                    - (log_x_idx[i] != self.n_classes - 1).sum()
                ).item()

            out = index_to_log_onehot(out2_idx, self.n_classes)
        else:
            # Gumbel sample
            out = self.log_sample_categorical(model_log_prob)
            sampled = [1024] * log_x_t.shape[0]

        if to_sample is not None:
            return out, sampled
        else:
            return out

    def q_pred_one_timestep(self, log_x_t, t):
        """
        Equation 3 of Vector Quantized Diffusion Model for Text-to-Image Synthesis:
        q(x_t|x_{t-1}) = transpose(v) (x_t) Q_t v (x_{t-1})
        """
        log_at = extract(self.log_at, t, log_x_t.shape)  # at
        log_bt = extract(self.log_bt, t, log_x_t.shape)  # bt
        log_ct = extract(self.log_ct, t, log_x_t.shape)  # ct
        log_1_min_ct = extract(self.log_1_min_ct, t, log_x_t.shape)  # 1-ct

        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:, :-1, :] + log_at, log_bt),
                log_add_exp(log_x_t[:, -1:, :] + log_1_min_ct, log_ct),
            ],
            dim=1,
        )

        return log_probs

    def q_posterior(self, log_x_0, log_x_t, t):
        """
        Equation 5 of Vector Quantized Diffusion Model for Text-to-Image Synthesis:
        q(x_{t-1} | x_t, x_0)
        """
        # notice that log_x_t is onehot
        assert t.min().item() >= 0 and t.max().item() < self.n_timesteps
        batch_size = log_x_0.size()[0]
        onehot_x_t = log_x_t.argmax(1)
        mask = (onehot_x_t == self.n_classes - 1).unsqueeze(1)
        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_x_t)
        log_zero_vector = torch.log(log_one_vector + 1.0e-30).expand(
            -1, -1, self.input_seq_length
        )

        """
        This section of code is confusing. Here is a writeup from the author
        from a github issue:
        https://github.com/microsoft/VQ-Diffusion/issues/20

        In the transition matrix Qt, the i-th row represents probabilities of
        getting different xts given a x0 which is the i-th token; The i-th
        column represents probabilities of getting the current xt (the i-th
                                                                   token) from
        different x0s. When calculating the posterior, our goal is to calculate
        the probability of different x0s transferring to the known xt
        (log_x_t). Therefore, we should calculate the columns corresponding to
        each element in log_x_t. However, the trivial solution may be a little
        complicated. The author used a trick: (1) Obtain Qt by q_pred(…, t).
        (2) Get corresponding rows according to items in log_x_t, i.e.,
        q_pred(log_x_t, t) (3) When [mask] is not considered, because of the
        symmetry of Qt, the rows we got is equivalent to the columns that
        represent probabilities of getting the current xt from different x0s.
        (4) If [mask] is considered, we have to replace the last value in rows
        with ct_cumprod (It is clear referring to Eq.7 in the paper.)
        """
        log_qt = self.q_pred(log_x_t, t)  # q(x_t|x_0)
        log_qt = log_qt[:, :-1, :]
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_0.shape)  # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.n_classes - 1, -1)
        log_qt = (~mask) * log_qt + mask * ct_cumprod_vector

        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)  # q(x_t|x_{t-1})
        log_qt_one_timestep = torch.cat(
            (log_qt_one_timestep[:, :-1, :], log_zero_vector), dim=1
        )
        log_ct = extract(self.log_ct, t, log_x_0.shape)  # ct
        ct_vector = log_ct.expand(-1, self.n_classes - 1, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask) * log_qt_one_timestep + mask * ct_vector

        """
        The goal of reparameterization is to calculate: 
            p(x_{t-1}|x_t) = sum(q(x_{t-1}|x_t,x_0) * p(x_0|x_t) )    Eq.11 in the paper
              where q(x_{t-1}|x_t,x_0) = q(x_t|x_{t-1},x_0) * q(x_{t-1}|x_0) / q(x_t|x_0) (*).

        In (*), because of property of Markov chain,
        q(x_t|x_{t-1},x_0) = q(x_t|x_{t-1}) = log_qt_one_timestep, 
        and p(x_0|x_t) * q(x_{t-1}|x_0) / q(x_t|x_0) = q_pred(q, t-1). 
        Because the sum of p(x_0|x_t) / q(x_t|x_0) is not 1,
        the following lines normalize it and then the second to last line renormalizes it.
        """

        q = log_x_0[:, :-1, :] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = (
            self.q_pred(q, t - 1) + log_qt_one_timestep + q_log_sum_exp
        )
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def predict_x_0(self, log_x_t, t):
        """
        returns log(p(x_0 | x_t))

        Page 5:

        Instead of directly predicting the posterior q(xt−1|xt, x0),
        recent works [1, 23, 26] find that approximating some sur-
        rogate variables, e.g., the noiseless target data q(x0) gives
        better quality. In the discrete setting, we let the network
        predict the noiseless token distribution pθ (  ̃x0|xt, y) at each
        reverse step. We can thus compute the reverse transition
        distribution according to:
        """
        # TODO should this be log_x_t or x_t?
        out = self.diffusion_model((log_x_t, t))

        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.n_classes - 1
        assert out.size()[2:] == x_t.size()[1:]
        log_pred = F.log_softmax(out.double(), dim=1).float()
        batch_size = log_x_t.size()[0]
        if self.zero_vector is None or self.zero_vector.shape[0] != batch_size:
            self.zero_vector = (
                torch.zeros(batch_size, 1, self.input_seq_length).type_as(log_x_t) - 70
            )
        log_pred = torch.cat((log_pred, self.zero_vector), dim=1)
        log_pred = torch.clamp(log_pred, -70, 0)

        return log_pred

    def q_sample(self, log_x_0, t):  # diffusion step, q(x_t|x_0) and sample x_t
        log_EV_qxt_x0 = self.q_pred(log_x_0, t)
        log_sample = self.log_sample_categorical(log_EV_qxt_x0)
        return log_sample

    def training_step(self, batch, batch_idx):
        """
        See Algorithm 1
        """
        ascii_batch, label = batch
        x_0 = self.vqvae.vq_embedding(self.vqvae.encoder(ascii_batch))

        bs = x_0.shape[0]
        t, pt = self.sample_time(bs)
        
        log_x_0 = index_to_log_onehot(x_0, self.n_classes)
        log_x_t = self.q_sample(log_x_0, t)
        x_t = log_x_t.argmax(1)
        log_x_0_recon = self.predict_x_0(log_x_t, t=t)
        log_model_prob = self.q_posterior(log_x_0=log_x_0, log_x_t=log_x_t, t=t)
        bpdb.set_trace()

        x_0_recon = log_x_0_recon.argmax(1)
        x_t_minus_1_recon = log_model_prob.argmax(1)
        x_t_recon = log_x_t.argmax(1)

        for index in range(t.size()[0]):
            this_t = t[index].item()
            same_rate = (x_0_recon[index] == x_0[index]).sum().cpu() / x_0.size()[1]
            self.diffusion_acc_list[this_t] = (
                same_rate.item() * 0.1 + self.diffusion_acc_list[this_t] * 0.9
            )
            same_rate = (
                x_t_minus_1_recon[index] == x_t_recon[index]
            ).sum().cpu() / x_t_recon.size()[1]
            self.diffusion_keep_list[this_t] = (
                same_rate.item() * 0.1 + self.diffusion_keep_list[this_t] * 0.9
            )

        # compute log_true_prob now
        log_true_prob = self.q_posterior(log_x_0=log_x_0, log_x_t=log_x_t, t=t)
        kl = multinomial_kl(log_true_prob, log_model_prob)
        mask_region = (x_t == self.n_classes - 1).float()
        mask_weight = (
            mask_region * self.mask_weight[0]
            + (1.0 - mask_region) * self.mask_weight[1]
        )
        kl = kl * mask_weight
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_0, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        kl_loss = mask * decoder_nll + (1.0 - mask) * kl

        Lt2 = kl_loss.pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        # Upweigh loss term of the kl
        # vb_loss = kl_loss / pt + kl_prior
        loss = kl_loss / pt

        return log_model_prob, loss

    def train_dataloader(self):
        dataset = AsciiArtDataset(res=64, validation_prop=0)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
        )

    def configure_optimizers(self):
        return torch.optim.Adam((*self.diffusion_model.parameters(),), lr=self.lr)
