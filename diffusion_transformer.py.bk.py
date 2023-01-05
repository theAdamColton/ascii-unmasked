# ------------------------------------------
# VQ-Diffusion
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Shuyang Gu
# ------------------------------------------
import math
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

eps = 1e-8


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
    def __init__(self, n_embd, diffusion_step, emb_type="adalayernorm_abs"):
        super().__init__()
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(diffusion_step, n_embd)
        else:
            self.emb = nn.Embedding(diffusion_step, n_embd)
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


def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract(a, t, x_shape):
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f"Error: {x.max().item()} >= {num_classes}"
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def alpha_schedule(
    time_step, N=100, att_1=0.99999, att_T=0.000009, ctt_1=0.000009, ctt_T=0.99999
):
    att = np.arange(0, time_step) / (time_step - 1) * (att_T - att_1) + att_1
    att = np.concatenate(([1], att))
    at = att[1:] / att[:-1]
    ctt = np.arange(0, time_step) / (time_step - 1) * (ctt_T - ctt_1) + ctt_1
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1 - one_minus_ct
    bt = (1 - at - ct) / N
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1 - att - ctt) / N
    return at, bt, ct, att, btt, ctt


def multinomial_kl(log_prob1, log_prob2):  # compute KL loss on log_prob
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        n_classes=1024,
        input_res=32,
        diffusion_step=100,
        auxiliary_loss_weight=0,
        adaptive_auxiliary_loss=False,
        mask_weight=[1, 1],
        learnable_cf=False,
    ):
        super().__init__()

        self.transformer = nn.Transformer(d_model=input_res**2)
        self.adaln = AdaLayerNorm(input_res**2, diffusion_step)

        self.num_timesteps = diffusion_step
        self.n_classes = n_classes
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss
        self.mask_weight = mask_weight

        at, bt, ct, att, btt, ctt = alpha_schedule(self.num_timesteps, N=n_classes - 1)

        at = torch.tensor(at.astype("float64"))
        bt = torch.tensor(bt.astype("float64"))
        ct = torch.tensor(ct.astype("float64"))
        log_at = torch.log(at)
        log_bt = torch.log(bt)
        log_ct = torch.log(ct)
        att = torch.tensor(att.astype("float64"))
        btt = torch.tensor(btt.astype("float64"))
        ctt = torch.tensor(ctt.astype("float64"))
        log_cumprod_at = torch.log(att)
        log_cumprod_bt = torch.log(btt)
        log_cumprod_ct = torch.log(ctt)

        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.0e-5
        assert (
            log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item()
            < 1.0e-5
        )

        self.diffusion_acc_list = [0] * self.num_timesteps
        self.diffusion_keep_list = [0] * self.num_timesteps
        # Convert to float32 and register buffers.
        self.register_buffer("log_at", log_at.float())
        self.register_buffer("log_bt", log_bt.float())
        self.register_buffer("log_ct", log_ct.float())
        self.register_buffer("log_cumprod_at", log_cumprod_at.float())
        self.register_buffer("log_cumprod_bt", log_cumprod_bt.float())
        self.register_buffer("log_cumprod_ct", log_cumprod_ct.float())
        self.register_buffer("log_1_min_ct", log_1_min_ct.float())
        self.register_buffer("log_1_min_cumprod_ct", log_1_min_cumprod_ct.float())

        self.register_buffer("Lt_history", torch.zeros(self.num_timesteps))
        self.register_buffer("Lt_count", torch.zeros(self.num_timesteps))
        self.zero_vector = None

        if learnable_cf:
            self.empty_text_embed = torch.nn.Parameter(
                torch.randn(size=(77, 512), requires_grad=True, dtype=torch.float64)
            )

        self.prior_rule = 0  # inference rule: 0 for VQ-Diffusion v1, 1 for only high-quality inference, 2 for purity prior
        self.prior_ps = 1024  # max number to sample per step
        self.prior_weight = 0  # probability adjust parameter, 'r' in Equation.11 of Improved VQ-Diffusion

        self.update_n_sample()

        self.learnable_cf = learnable_cf

    def update_n_sample(self):
        if self.num_timesteps == 100:
            if self.prior_ps <= 10:
                self.n_sample = [1, 6] + [11, 10, 10] * 32 + [11, 15]
            else:
                self.n_sample = [1, 10] + [11, 10, 10] * 32 + [11, 11]
        elif self.num_timesteps == 50:
            self.n_sample = [10] + [21, 20] * 24 + [30]
        elif self.num_timesteps == 25:
            self.n_sample = [21] + [41] * 23 + [60]
        elif self.num_timesteps == 10:
            self.n_sample = [69] + [102] * 8 + [139]

    def q_pred_one_timestep(self, log_x_t, t):  # q(xt|xt_1)
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

    def q_pred(self, log_x_start, t):  # q(xt|x0)
        """
        Equation 4
        """
        # log_x_start can be onehot or not
        t = (t + (self.num_timesteps + 1)) % (self.num_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)  # at~
        log_cumprod_bt = extract(self.log_cumprod_bt, t, log_x_start.shape)  # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)  # ct~
        log_1_min_cumprod_ct = extract(
            self.log_1_min_cumprod_ct, t, log_x_start.shape
        )  # 1-ct~

        log_probs = torch.cat(
            [
                log_add_exp(log_x_start[:, :-1, :] + log_cumprod_at, log_cumprod_bt),
                log_add_exp(
                    log_x_start[:, -1:, :] + log_1_min_cumprod_ct, log_cumprod_ct
                ),
            ],
            dim=1,
        )

        return log_probs

    def predict_start(self, log_x_t, t):  # p(x0|xt)
        """
        For reference, the bottom of page 5 of Vector Quantized Diffusion Model for Text-to-Image Synthesis
        contains the definition of the proposed model architecture.

        The current timestep t is injected into the network with Adaptive Layer Normalization, (AdaLN)
        """
        x_t = log_onehot_to_index(log_x_t)
        out = self.transformer(self.adaln(x_t, t))

        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.n_classes - 1
        assert out.size()[2:] == x_t.size()[1:]
        log_pred = F.log_softmax(out.double(), dim=1).float()
        batch_size = log_x_t.size()[0]
        if self.zero_vector is None or self.zero_vector.shape[0] != batch_size:
            self.zero_vector = (
                torch.zeros(batch_size, 1, self.content_seq_len).type_as(log_x_t) - 70
            )
        log_pred = torch.cat((log_pred, self.zero_vector), dim=1)
        log_pred = torch.clamp(log_pred, -70, 0)

        return log_pred

    def cf_predict_start(self, log_x_t, t):
        return self.predict_start(log_x_t, t)

    def q_posterior(self, log_x_start, log_x_t, t):
        """
        Equation 5
        """

        # notice that log_x_t is onehot
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        batch_size = log_x_start.size()[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.num_classes - 1).unsqueeze(1)
        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_x_t)
        log_zero_vector = torch.log(log_one_vector + 1.0e-30).expand(
            -1, -1, self.content_seq_len
        )

        log_qt = self.q_pred(log_x_t, t)  # q(xt|x0)
        log_qt = log_qt[:, :-1, :]
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)  # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes - 1, -1)
        log_qt = (~mask) * log_qt + mask * ct_cumprod_vector

        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)  # q(xt|xt_1)
        log_qt_one_timestep = torch.cat(
            (log_qt_one_timestep[:, :-1, :], log_zero_vector), dim=1
        )
        log_ct = extract(self.log_ct, t, log_x_start.shape)  # ct
        ct_vector = log_ct.expand(-1, self.num_classes - 1, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask) * log_qt_one_timestep + mask * ct_vector

        q = log_x_start[:, :-1, :] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = (
            self.q_pred(q, t - 1) + log_qt_one_timestep + q_log_sum_exp
        )
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def p_pred(
        self, log_x, t
    ):  # if x0, first p(x0|xt), than sum(q(xt-1|xt,x0)*p(x0|xt))
        log_x_recon = self.cf_predict_start(log_x, t)
        log_model_pred = self.q_posterior(log_x_start=log_x_recon, log_x_t=log_x, t=t)
        return log_model_pred, log_x_recon

    @torch.no_grad()
    def p_sample(
        self, log_x, t, sampled=None, to_sample=None
    ):  # sample q(x_{t-1) for next step from  x_t, actually is p(x_{t-1}|x_t)
        model_log_prob, log_x_recon = self.p_pred(log_x, t)

        max_sample_per_step = self.prior_ps  # max number to sample per step
        if (
            t[0] > 0 and self.prior_rule > 0 and to_sample is not None
        ):  # prior_rule: 0 for VQ-Diffusion v1, 1 for only high-quality inference, 2 for purity prior
            log_x_idx = log_onehot_to_index(log_x)

            if self.prior_rule == 1:
                score = torch.ones((log_x.shape[0], log_x.shape[2])).to(log_x.device)
            elif self.prior_rule == 2:
                score = torch.exp(log_x_recon).max(dim=1).values.clamp(0, 1)
                score /= score.max(dim=1, keepdim=True).values + 1e-10

            if self.prior_rule != 1 and self.prior_weight > 0:
                # probability adjust parameter, prior_weight: 'r' in Equation.11 of Improved VQ-Diffusion
                prob = (
                    (1 + score * self.prior_weight).unsqueeze(1) * log_x_recon
                ).softmax(dim=1)
                prob = prob.log().clamp(-70, 0)
            else:
                prob = log_x_recon

            out = self.log_sample_categorical(prob)
            out_idx = log_onehot_to_index(out)

            out2_idx = log_x_idx.clone()
            _score = score.clone()
            if _score.sum() < 1e-6:
                _score += 1
            _score[log_x_idx != self.num_classes - 1] = 0

            for i in range(log_x.shape[0]):
                n_sample = min(to_sample - sampled[i], max_sample_per_step)
                if to_sample - sampled[i] - n_sample == 1:
                    n_sample = to_sample - sampled[i]
                if n_sample <= 0:
                    continue
                sel = torch.multinomial(_score[i], n_sample)
                out2_idx[i][sel] = out_idx[i][sel]
                sampled[i] += (
                    (out2_idx[i] != self.num_classes - 1).sum()
                    - (log_x_idx[i] != self.num_classes - 1).sum()
                ).item()

            out = index_to_log_onehot(out2_idx, self.num_classes)
        else:
            # Gumbel sample
            out = self.log_sample_categorical(model_log_prob)
            sampled = [1024] * log_x.shape[0]

        if to_sample is not None:
            return out, sampled
        else:
            return out

    def log_sample_categorical(
        self, logits
    ):  # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):  # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def sample_time(self, b, device):
        """
        Uniform time sample
        """
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        pt = torch.ones_like(t).float() / self.num_timesteps
        return t, pt

    def _train_loss(self, x, is_train=True):  # get the KL loss
        b, device = x.size(0), x.device

        x_start = x
        t, pt = self.sample_time(b, device)

        log_x_start = index_to_log_onehot(x_start, self.num_classes)
        log_xt = self.q_sample(log_x_start=log_x_start, t=t)
        xt = log_onehot_to_index(log_xt)

        ############### go to p_theta function ###############
        log_x0_recon = self.predict_start(log_xt, t=t)  # P_theta(x0|xt)
        log_model_prob = self.q_posterior(
            log_x_start=log_x0_recon, log_x_t=log_xt, t=t
        )  # go through q(xt_1|xt,x0)

        ################## compute acc list ################
        x0_recon = log_onehot_to_index(log_x0_recon)
        x0_real = x_start
        xt_1_recon = log_onehot_to_index(log_model_prob)
        xt_recon = log_onehot_to_index(log_xt)
        for index in range(t.size()[0]):
            this_t = t[index].item()
            same_rate = (
                x0_recon[index] == x0_real[index]
            ).sum().cpu() / x0_real.size()[1]
            self.diffusion_acc_list[this_t] = (
                same_rate.item() * 0.1 + self.diffusion_acc_list[this_t] * 0.9
            )
            same_rate = (
                xt_1_recon[index] == xt_recon[index]
            ).sum().cpu() / xt_recon.size()[1]
            self.diffusion_keep_list[this_t] = (
                same_rate.item() * 0.1 + self.diffusion_keep_list[this_t] * 0.9
            )

        # compute log_true_prob now
        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t)
        kl = multinomial_kl(log_true_prob, log_model_prob)
        mask_region = (xt == self.num_classes - 1).float()
        mask_weight = (
            mask_region * self.mask_weight[0]
            + (1.0 - mask_region) * self.mask_weight[1]
        )
        kl = kl * mask_weight
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
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
        loss1 = kl_loss / pt
        vb_loss = loss1
        if self.auxiliary_loss_weight != 0 and is_train == True:
            kl_aux = multinomial_kl(log_x_start[:, :-1, :], log_x0_recon[:, :-1, :])
            kl_aux = kl_aux * mask_weight
            kl_aux = sum_except_batch(kl_aux)
            kl_aux_loss = mask * decoder_nll + (1.0 - mask) * kl_aux
            if self.adaptive_auxiliary_loss == True:
                addition_loss_weight = (1 - t / self.num_timesteps) + 1.0
            else:
                addition_loss_weight = 1.0

            loss2 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / pt
            vb_loss += loss2

        return log_model_prob, vb_loss

    def forward(
        self,
        input,
        return_loss=False,
        return_logits=True,
        is_train=True,
        **kwargs,
    ):
        if kwargs.get("autocast") == True:
            self.amp = True

        # 1) get embeddding for condition and content     prepare input
        sample_image = input["content_token"].type_as(input["content_token"])
        # cont_emb = self.content_emb(sample_image)

        if is_train == True:
            log_model_prob, loss = self._train_loss(sample_image)
            loss = loss.sum() / (sample_image.size()[0] * sample_image.size()[1])

        # 4) get output, especially loss
        out = {}
        if return_logits:
            out["logits"] = torch.exp(log_model_prob)

        if return_loss:
            out["loss"] = loss
        self.amp = False
        return out

    def sample_fast(
        self,
        condition_token,
        content_token=None,
        filter_ratio=0.5,
        return_logits=False,
        skip_step=1,
    ):

        batch_size = condition_token.shape[0]
        device = self.log_at.device
        start_step = int(self.num_timesteps * filter_ratio)

        assert start_step == 0
        zero_logits = torch.zeros(
            (batch_size, self.num_classes - 1, self.shape), device=device
        )
        one_logits = torch.ones((batch_size, 1, self.shape), device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_z = torch.log(mask_logits)
        start_step = self.num_timesteps
        with torch.no_grad():
            # skip_step = 1
            diffusion_list = [
                index for index in range(start_step - 1, -1, -1 - skip_step)
            ]
            if diffusion_list[-1] != 0:
                diffusion_list.append(0)
            # for diffusion_index in range(start_step-1, -1, -1):
            for diffusion_index in diffusion_list:

                t = torch.full(
                    (batch_size,), diffusion_index, device=device, dtype=torch.long
                )
                log_x_recon = self.cf_predict_start(log_z, t)
                if diffusion_index > skip_step:
                    model_log_prob = self.q_posterior(
                        log_x_start=log_x_recon, log_x_t=log_z, t=t - skip_step
                    )
                else:
                    model_log_prob = self.q_posterior(
                        log_x_start=log_x_recon, log_x_t=log_z, t=t
                    )

                log_z = self.log_sample_categorical(model_log_prob)

        content_token = log_onehot_to_index(log_z)

        output = {"content_token": content_token}
        if return_logits:
            output["logits"] = torch.exp(log_z)
        return output
