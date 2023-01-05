import torch
import json
from pytorch_lightning.callbacks import StochasticWeightAveraging, ModelCheckpoint
from torch.utils.data import DataLoader
import torchinfo
import argparse
import pytorch_lightning as pl
import datetime
import os
import sys


from vqvae import VQ_VAE
from diffusion_transformer import DiffusionTransformer


def get_training_args():
    parser = argparse.ArgumentParser()
    # Renderer args
    parser.add_argument(
        "--font-res",
        dest="font_res",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--font-zoom",
        dest="font_zoom",
        type=int,
        default=20,
    )

    parser.add_argument("--print-every", "-p", dest="print_every", default=10, type=int)
    parser.add_argument(
        "--run-name",
        dest="run_name",
        default="vqdiffusion",
    )
    parser.add_argument(
        "--log-name",
        dest="log_name",
        default="run",
    )

    # Dataset args
    parser.add_argument(
        "--datapath",
        dest="datapath",
        default=None,
        help="Useful for memory-pinned data directories in /dev/shm/",
    )
    parser.add_argument(
        "--n-workers",
        dest="n_workers",
        type=int,
        default=4,
        help="Number of dataset workers",
    )
    parser.add_argument(
        "--dataset-to-gpu",
        dest="dataset_to_gpu",
        default=False,
        action="store_true",
    )

    # Training args
    parser.add_argument(
        "--learning-rate", dest="learning_rate", default=5e-5, type=float
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        dest="batch_size",
        default=64,
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
        "--vq-diff-dir",
        dest="vq_diff_dir",
        type=str,
        help="Directory to the vq diffusion model checkpoint",
    )

    parser.add_argument(
        "--validation-prop", dest="validation_prop", default=0.0, type=float
    )
    parser.add_argument(
        "--validation-every", dest="validation_every", default=5, type=int
    )

    parser.add_argument(
        "--gumbel-tau-r",
        dest="gumbel_tau_r",
        type=float,
        default=7e-5,
    )
    parser.add_argument(
        "--dont-augment-data",
        dest="dont_augment_data",
        action="store_true",
        default=True,
    )

    return parser.parse_args()


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
        callbacks=[StochasticWeightAveraging(swa_lrs=0.05), model_checkpoint],
        check_val_every_n_epoch=args.validation_every,
        auto_lr_find=True,
        logger=logger,
        log_every_n_steps=10,
        precision=16,
        amp_backend="native",
        gradient_clip_val=1.0,
    )

    vqvae = VQ_VAE.load_from_checkpoint(args.vq_vae_dir)
    vqvae.eval()
    vqdiff = DiffusionTransformer(32, 512, vqvae)

    torchinfo.summary(vqvae.encoder, input_size=(7, 95, 64, 64))
    torchinfo.summary(vqvae.decoder, input_size=(7, 256, 32, 32))

    trainer.fit(model=vqdiff)
