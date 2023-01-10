import torch
import json
from pytorch_lightning.callbacks import StochasticWeightAveraging, ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from einops import rearrange
import torch.nn.functional as F
from random import randint
import argparse
import pytorch_lightning as pl
import datetime
import os
import sys

import bpdb

dirname = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(dirname, "./ascii-dataset/"))
from dataset import AsciiArtDataset
import ascii_util

from vqvae import VQ_VAE
from transformer import VQGANTransformer
from augmentation import RandomRoll

def get_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        dest="run_name",
        default="vqmaskgit",
    )
    parser.add_argument(
        "--log-name",
        dest="log_name",
        default="run",
    )

    # Training args
    parser.add_argument(
        "--learning-rate", dest="learning_rate", default=5e-5, type=float
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        dest="batch_size",
        default=8,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "-n",
        "--n_epochs",
        dest="n_epochs",
        default=200,
        type=int,
        help="Number of epochs",
    )
    parser.add_argument(
        "--vq-vae-dir",
        dest="vq_vae_dir",
        type=str,
        help="Directory to the vq vae model checkpoint",
        required=True,
    )
    parser.add_argument(
        "--maskgit-dir",
        dest="maskgit_dir",
        type=str,
        help="Directory to the maskgit model checkpoint",
    )

    parser.add_argument(
        "--validation-prop", dest="validation_prop", default=0.0, type=float
    )
    parser.add_argument(
        "--validation-every", dest="validation_every", default=5, type=int
    )
    parser.add_argument(
        "--dont-augment-data",
        dest="dont_augment_data",
        action="store_true",
        default=True,
    )
    parser.add_argument(
            "--find-lr",
            dest="find_lr",
            default=False,
            action="store_true",
            )

    return parser.parse_args()


class MaskGitTrainer(pl.LightningModule):
    def __init__(
        self,
        transformer: VQGANTransformer,
        vae: VQ_VAE,
        batch_size: int,
        should_random_roll=True,
        max_res=105,
        validation_prop=0.01,
        learning_rate=5e-4,
    ):
        super().__init__()

        self.transformer = transformer
        self.vae = vae
        self.vae.eval()
        self.hparams.lr = learning_rate
        self.should_random_roll = should_random_roll
        self.validation_prop = validation_prop
        self.max_res = max_res
        self.automatic_optimization = True
        self.batch_size = batch_size

        if self.should_random_roll:
            self.random_roll = RandomRoll(max_shift=6, sigma=4)

    def step(self, x, _):
        with torch.no_grad():
            indeces = self.vae.encode(x.to(self.vae.dtype))
        logits, targets = self.transformer(indeces)
        logits = rearrange(logits, "b w c -> b c w")

        loss = F.cross_entropy(logits, targets)
        logs = {
            "l": loss,
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        lrs = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-9)
        return {'optimizer':optimizer, 'lr_scheduler': lrs, 'monitor':'t_l'}

    def on_train_epoch_end(self):
        with torch.no_grad():
            self.eval()
            # Generates an image
            id_res = randint(4, 16)
            ids = self.transformer.sample_good(id_res**2, T=18, batch_size=2)
            ids = ids.reshape(ids.shape[0], id_res, id_res)
            ascii_tensors = self.vae.decode_from_ids(ids)
            ascii_tensors_log = F.log_softmax(ascii_tensors)
            ascii_tensors_gumbel = F.gumbel_softmax(ascii_tensors_log, dim=1)

            for i, ascii_tensor in enumerate(ascii_tensors_gumbel):
                ascii_str = ascii_util.one_hot_embedded_matrix_to_string(ascii_tensor)
                print(ascii_str)
                if i+1 < ascii_tensors.shape[0]:
                    print("-**-" * id_res)

        self.train()


if __name__ in {"__main__", "__console__"}:
    args = get_training_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dt_string = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")
    dirpath = "ckpt/{}checkpoint/{}".format(args.run_name, dt_string)
    logger = pl.loggers.TensorBoardLogger(dirpath)
    model_checkpoint = ModelCheckpoint(
        dirpath=dirpath,
        monitor="t_l",
        save_last=True,
        save_top_k=2,
        save_on_train_epoch_end=True,
    )
    os.makedirs(dirpath, exist_ok=True)
    with open(os.path.join(dirpath, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        accelerator="gpu" if device.type == "cuda" else "cpu",
        callbacks=[StochasticWeightAveraging(swa_lrs=0.005), model_checkpoint, LearningRateMonitor()],
        check_val_every_n_epoch=args.validation_every,
        auto_lr_find=True,
        logger=logger,
        log_every_n_steps=10,
        precision=16,
        amp_backend="native",
        accumulate_grad_batches=12,
        gradient_clip_val=0.5,
    )

    vqvae = VQ_VAE.load_from_checkpoint(args.vq_vae_dir)
    vqvae.eval()

    transformer = VQGANTransformer(512, 20**2, vqvae)
    # torchinfo.summary(maskgit, input_size=(7, 16, 16))

    maskgit_trainer = MaskGitTrainer(transformer, vqvae, args.batch_size, learning_rate=args.learning_rate, validation_prop=args.validation_prop)
    maskgit_trainer.to(torch.device("cuda"))

    if args.find_lr:
        lr_finder = trainer.tuner.lr_find(maskgit_trainer)
        new_lr = lr_finder.suggestion()
        print(lr_finder.results)
        print(new_lr)
        maskgit_trainer.hparams.lr = new_lr

    if not args.maskgit_dir:
        trainer.fit(model=maskgit_trainer)
    else:
        trainer.fit(model=maskgit_trainer, ckpt_path=args.maskgit_dir)
