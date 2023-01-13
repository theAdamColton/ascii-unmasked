import torch
import json
from pytorch_lightning.callbacks import (
    StochasticWeightAveraging,
    ModelCheckpoint,
    LearningRateMonitor,
)
from torch.utils.data import DataLoader
import torchinfo
import argparse
import pytorch_lightning as pl
import datetime
import os

from vqvae import VQ_VAE


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

    parser.add_argument(
        "--max-res",
        dest="max_res",
        default=64,
        help="Maximum resolution the dataset will serve",
        type=int,
    )

    # Loss coefficients
    parser.add_argument(
        "--ce-recon-loss-scale", dest="ce_recon_loss_scale", default=0.1, type=float
    )
    parser.add_argument(
        "--ce-similarity-loss-coeff",
        dest="ce_similarity_loss_coeff",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--ce-label-smoothing",
        dest="ce_label_smoothing",
        default=0.0,
        type=float,
        help="Label smoothing [0,1]: https://arxiv.org/pdf/1701.06548.pdf",
    )
    parser.add_argument("--vq-beta", dest="vq_beta", default=1, type=float)
    parser.add_argument("--vq-k", dest="vq_k", default=512, type=int)
    parser.add_argument("--vq-z-dim", dest="vq_z_dim", default=256, type=int)
    parser.add_argument(
        "--image-recon-loss-coeff",
        dest="image_recon_loss_coeff",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--run-name",
        dest="run_name",
        default="vqvae",
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
    parser.add_argument("-l", "--load", dest="load", help="load models from directory")
    parser.add_argument(
        "--validation-prop", dest="validation_prop", default=0.0, type=float
    )
    parser.add_argument(
        "--validation-every", dest="validation_every", default=5, type=int
    )

    parser.add_argument(
        "--space-deemph",
        dest="space_deemph",
        default=1.0,
        type=float,
        help="The space character weight is divided by this number.",
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
    parser.add_argument(
        "--find-lr",
        dest="find_lr",
        default=False,
        action="store_true",
    )
    return parser.parse_args()


if __name__ in {"__main__", "__console__"}:
    args = get_training_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    char_weights = torch.ones(95)
    char_weights[0] = char_weights[0] / args.space_deemph

    if args.load:
        vqvae = VQ_VAE.load_from_checkpoint(
            checkpoint_path=args.load,
            lr=args.learning_rate,
            char_weights=char_weights,
            label_smoothing=args.ce_label_smoothing,
            ce_recon_loss_scale=args.ce_recon_loss_scale,
            image_recon_loss_coeff=args.image_recon_loss_coeff,
            gumbel_tau_r=args.gumbel_tau_r,
            device=device,
            kernel_size=3,
            vq_beta=args.vq_beta,
            vq_z_dim=args.vq_z_dim,
            should_random_roll=not args.dont_augment_data,
            validation_prop=args.validation_prop,
            batch_size=args.batch_size,
            max_res=args.max_res,
            ce_similarity_loss_coeff=args.ce_similarity_loss_coeff,
        )
    else:
        vqvae = VQ_VAE(
            lr=args.learning_rate,
            char_weights=char_weights,
            label_smoothing=args.ce_label_smoothing,
            ce_recon_loss_scale=args.ce_recon_loss_scale,
            image_recon_loss_coeff=args.image_recon_loss_coeff,
            gumbel_tau_r=args.gumbel_tau_r,
            device=device,
            kernel_size=3,
            vq_beta=args.vq_beta,
            vq_z_dim=args.vq_z_dim,
            should_random_roll=not args.dont_augment_data,
            validation_prop=args.validation_prop,
            batch_size=args.batch_size,
            max_res=args.max_res,
            ce_similarity_loss_coeff=args.ce_similarity_loss_coeff,
        )

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
        callbacks=[
            StochasticWeightAveraging(swa_lrs=0.05),
            model_checkpoint,
            LearningRateMonitor(),
        ],
        check_val_every_n_epoch=args.validation_every,
        auto_lr_find=True,
        logger=logger,
        precision=16,
        amp_backend="native",
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
    )

    torchinfo.summary(vqvae.encoder, input_size=(7, 95, 64, 64))

    if args.find_lr:
        lr_finder = trainer.tuner.lr_find(vqvae)
        new_lr = lr_finder.suggestion()
        print(lr_finder.results)
        print(new_lr)
        vqvae.lr = new_lr

    if not args.load:
        trainer.fit(model=vqvae)
    else:
        trainer.fit(
            model=vqvae,
            ckpt_path=args.load,
        )
