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
from character_embeddings import generate_embedding_space_distances


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
        vq_z_dim=256,
        should_random_roll=True,
        random_roll_sigma=4.0,
        random_roll_max_roll=10.0,
        validation_prop=0.01,
        batch_size=16,
        max_res=128,
        ce_similarity_loss_coeff=1.0,
    ):
        super().__init__()

        self.encoder = Encoder(kernel_size=kernel_size)
        self.decoder = Decoder(kernel_size=kernel_size)
        self.vq_k = vq_k
        self.hparams.learning_rate = lr
        self.ce_recon_loss_scale = ce_recon_loss_scale
        self.ce_similarity_loss_coeff = ce_similarity_loss_coeff
        self.image_recon_loss_coeff = image_recon_loss_coeff
        self.gumbel_tau_r = gumbel_tau_r
        self.validation_prop = validation_prop
        self.batch_size = batch_size
        self.max_res = max_res

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

        characters_distances = torch.Tensor(generate_embedding_space_distances(n_components=12))
        # Want the characters_similarity_matrix to sum to 1 along it's rows.
        characters_similarity_matrix = (characters_distances / characters_distances.max()).softmax(dim=1)
        self.characters_similarity_matrix = characters_similarity_matrix.to(device)

        self.save_hyperparameters(ignore=["font_renderer"])

    def forward(self, x):
        """returns x_hat, z_e_x, z_q_x_st, z_q_z"""
        x = x.squeeze(0)
        x = x.to(self.dtype)
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.vq_embedding.straight_through(z_e_x)
        x_hat = self.decoder(z_q_x_st, x_res=x.shape[2])
#        x_hat_log = F.log_softmax(x_hat, dim=1)
        self.gumbel_tau = max(0.5, math.exp(-self.gumbel_tau_r * self.current_epoch))
#        x_hat_gumbel = F.gumbel_softmax(
#            x_hat_log, dim=1, tau=self.gumbel_tau, hard=True
#        )
        return x_hat, z_e_x, z_q_x_st, z_q_x

    def get_indeces_from_continuous(self, z_e_x):
        """
        discretizes based on the embedding space
        """
        return self.vq_embedding(z_e_x)

    def decode_from_z_e_x(self, z_e_x, x_res=None):
        """
        decodes from undiscretized latent z_e_x
        returns x_hat, z_q_x_st, z_q_z
        """
        z_q_st, z_q_z = self.vq_embedding.straight_through(z_e_x)
        x_hat = self.decoder(z_q_st, x_res=x_res)
        return x_hat, z_q_st, z_q_z

    def encode(self, x):
        x = x.squeeze(0)
        z_e_x = self.encoder(x)
        indeces = self.vq_embedding.forward(z_e_x)

        return indeces

    def step(self, x, _):
        x_hat, z_e_x, z_q_x_st, z_q_x = self.forward(x)
        ce_rec_loss = (
            self.ce_loss(x_hat, x.argmax(dim=1)) * self.ce_recon_loss_scale
        )

        if self.ce_similarity_loss_coeff > 0.0:
            ce_similarity_loss = (self.characters_similarity_matrix[x.argmax(dim=1)].movedim(-1, 1)*x_hat.softmax(dim=1)).mean() * self.ce_similarity_loss_coeff
        else:
            ce_similarity_loss = 0.0

        if self.image_recon_loss_coeff > 0.0:
            base_image = self.font_renderer.render(x)
            recon_image = self.font_renderer.render(x_hat)
            im_rec_loss = F.mse_loss(base_image, recon_image) * self.image_recon_loss_coeff
        else:
            im_rec_loss = 0.0

        vq_loss = F.mse_loss(z_q_x, z_e_x.detach())
        commit_loss = F.mse_loss(z_e_x, z_q_x.detach()) * self.vq_beta

        loss = ce_rec_loss + im_rec_loss + vq_loss + commit_loss + ce_similarity_loss
        logs = {
            "im": im_rec_loss,
            "ce": ce_rec_loss,
            "sl": ce_similarity_loss,
            "vq": vq_loss,
            "cl": commit_loss,
            "l": loss,
            "gt": self.gumbel_tau,
        }

        return loss, logs

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.squeeze(0)
        if self.should_random_roll:
            x = self.random_roll(x)
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
                x.to(self.dtype).unsqueeze(0).unsqueeze(0)
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
        dataset = AsciiArtDataset(
            res=self.max_res,
            ragged_batch_bin=True,
            ragged_batch_bin_batch_size=self.batch_size,
            validation_prop=self.validation_prop,
        )
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=16,
        )

    def validation_dataloader(self):
        validation_dataset = AsciiArtDataset(
            res=self.max_res,
            ragged_batch_bin=True,
            ragged_batch_bin_batch_size=self.batch_size,
            validation_prop=self.validation_prop,
            is_validation_dataset=self.validation_prop > 0.0,
        )
        return DataLoader(
            validation_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            (
                *self.encoder.parameters(),
                *self.decoder.parameters(),
                *self.vq_embedding.embedding.parameters(),
            ),
            lr=self.hparams.learning_rate,
        )
        #lrs = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6, factor=0.8,cooldown=5, threshold=1e-5)
        #return {"optimizer": optimizer, "lr_scheduler": lrs, "monitor": "t_l"}
        return optimizer

    def get_encoded_fmap_size(self, image_size: int):
        # 64 -> 16
        return image_size // 4

    def decode_from_ids(self, ids: torch.Tensor):
        """
        ids is B by Z by Z
        returns B by 95 by Z*4 by Z*4
        """
        latent_res = ids.shape[1]
        z_q = self.vq_embedding.embedding.weight[ids]
        z_q = torch.movedim(z_q, -1, 1)
        return self.decoder(z_q, x_res=latent_res * 4)
