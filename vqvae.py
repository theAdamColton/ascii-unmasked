import torch
from torch.utils.data import DataLoader
import math
import torch.nn.functional as F
import pytorch_lightning as pl
from vector_quantize import VQEmbedding

import sys
from os import path

from autoencoder import Encoder, Decoder

from augmentation import RandomRoll

dirname = path.dirname(__file__)
sys.path.insert(0, path.join(dirname, "./ascii-dataset/"))
from dataset import AsciiArtDataset
import ascii_util

sys.path.insert(0, path.join(dirname, "./python-pytorch-font-renderer/"))
from font_renderer import FontRenderer


class VQ_VAE(pl.LightningModule):
    def __init__(
        self,
        lr=5e-5,
        char_weights=None,
        label_smoothing=0.0,
        ce_recon_loss_scale=1.0,
        image_recon_loss_coeff=1.0,
        gumbel_tau_r=5e-5,
        device=torch.device("cuda"),
        kernel_size=3,
        vq_beta=1.0,
        vq_k=512,
        vq_z_dim=128,
        should_random_roll=True,
        random_roll_sigma=4.0,
        random_roll_max_roll=8.0,
        validation_prop=0.01,
        batch_size=16,
    ):
        super().__init__()

        self.encoder = Encoder(kernel_size=kernel_size)
        self.decoder = Decoder(kernel_size=kernel_size)
        self.vq_k = vq_k
        self.lr = lr
        self.ce_recon_loss_scale = ce_recon_loss_scale
        self.image_recon_loss_coeff = image_recon_loss_coeff
        self.gumbel_tau_r = gumbel_tau_r
        self.validation_prop = validation_prop
        self.batch_size = batch_size

        # Codebook embedding
        self.vq_embedding = VQEmbedding(vq_k, vq_z_dim)
        self.vq_beta = vq_beta

        # RandomRoll
        self.should_random_roll = should_random_roll
        if self.should_random_roll:
            self.random_roll = RandomRoll(
                max_shift=random_roll_max_roll, sigma=random_roll_sigma
            )

        # Some reasonable initial parameters
        self.font_renderer = FontRenderer(res=9, device=device, zoom=22)

        self.ce_loss = torch.nn.CrossEntropyLoss(
            weight=char_weights, label_smoothing=label_smoothing
        )

        self.save_hyperparameters(ignore=["font_renderer"])

    def get_z(self, x):
        """
        returns the z discrete space embedding
        """
        z_e_x = self.encoder(x)
        

    def forward(self, x):
        """returns x_hat, z_e_x, z_q_x_st, z_q_z"""
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.vq_embedding.straight_through(z_e_x)
        x_hat = self.decoder(z_q_x_st)
        x_hat_log = F.log_softmax(x_hat, dim=1)
        self.gumbel_tau = max(0.5, math.exp(-self.gumbel_tau_r * self.current_epoch))
        x_hat_gumbel = F.gumbel_softmax(
            x_hat_log, dim=1, tau=self.gumbel_tau, hard=True
        )
        return x_hat_gumbel, z_e_x, z_q_x_st, z_q_x

    def step(self, x, _):
        x_hat_gumbel, z_e_x, z_q_x_st, z_q_x = self.forward(x)
        ce_rec_loss = (
            self.ce_loss(x_hat_gumbel, x.argmax(dim=1)) * self.ce_recon_loss_scale
        )

        base_image = self.font_renderer.render(x)
        recon_image = self.font_renderer.render(x_hat_gumbel)
        im_rec_loss = F.mse_loss(base_image, recon_image) * self.image_recon_loss_coeff

        vq_loss = F.mse_loss(z_q_x, z_e_x.detach())
        commit_loss = F.mse_loss(z_e_x, z_q_x.detach()) * self.vq_beta

        loss = ce_rec_loss + im_rec_loss + vq_loss + commit_loss
        logs = {
            "im": im_rec_loss,
            "ce": ce_rec_loss,
            "vq": vq_loss,
            "cl": commit_loss,
            "l": loss,
            "gt": self.gumbel_tau,
        }

        return loss, logs

    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss, logs = self.step(x, batch_idx)

        self.log_dict(
            {f"t_{k}": v for k, v in logs.items()},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        return loss

    def on_train_epoch_end(self):
        with torch.no_grad():
            x, label = self.train_dataloader().dataset.get_random_training_item()
            x = torch.Tensor(x).to(self.device)

            if self.should_random_roll:
                x = self.random_roll(x)

            self.eval()

            # Reconstructs the item
            x_hat_gumbel, z_e_x, z_q_x_st, z_q_x = self.forward(
                x.to(self.dtype).unsqueeze(0)
            )

            # Renders images
            base_image = self.font_renderer.render(x.unsqueeze(0))
            recon_image = self.font_renderer.render(x_hat_gumbel)

            side_by_side = torch.concat((base_image, recon_image), dim=2).squeeze(0)
            side_by_side = side_by_side.unsqueeze(0)
            # Logs images
            self.logger.experiment.add_image(
                "epoch {}".format(self.current_epoch), side_by_side, 0
            )

            x_str = ascii_util.one_hot_embedded_matrix_to_string(x)
            x_recon_str = ascii_util.one_hot_embedded_matrix_to_string(
                x_hat_gumbel.squeeze(0)
            )
            side_by_side = ascii_util.horizontal_concat(x_str, x_recon_str)
            print(side_by_side)
            print(label)

        self.train()

    def train_dataloader(self):
        dataset = AsciiArtDataset(res=64, validation_prop=self.validation_prop)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
        )

    def validation_dataloader(self):
        validation_dataset = AsciiArtDataset(
            res=64,
            validation_prop=self.validation_prop,
            is_validation_dataset=self.validation_prop > 0.0,
        )
        return DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(
            (
                *self.encoder.parameters(),
                *self.decoder.parameters(),
                *self.vq_embedding.embedding.parameters(),
            ),
            lr=self.lr,
        )
