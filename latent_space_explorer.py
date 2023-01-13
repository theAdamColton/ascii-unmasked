"""
Traverses the latent space
"""
import random
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
from os import path
import curses
import time
import sys
import numpy as np

dirname = path.dirname(__file__)
sys.path.insert(0, path.join(dirname, "./ascii-dataset/"))
import ascii_util
from dataset import AsciiArtDataset
from dataset import ascii_util

dirname = path.dirname(__file__)
sys.path.insert(0, path.join(dirname, "./python-pytorch-font-renderer/"))
from font_renderer import ContinuousFontRenderer

from vqvae import VQ_VAE
from train_vqmaskgit import MaskGitTrainer


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--frame-rate", dest="frame_rate", type=float, default=60)
    parser.add_argument(
        "--steps",
        dest="steps",
        type=int,
        default=100,
        help="The number of steps between different encoded values",
    )
    parser.add_argument(
        "--hold-length",
        dest="hold_length",
        default=1,
        type=float,
        help="Adjusts the amount of time spent paused during transitions",
    )
    parser.add_argument(
        "--maskgit-r-prop",
        dest="maskgit_r_prop",
        default=0.8,
        type=float,
        help="One minus the expected propotion of latent values that will become masked",
    )
    parser.add_argument(
        "--smooth-factor",
        dest="smooth_factor",
        default=1.0,
        type=float,
        help="Any number in [0,1], represents the smoothing between the different embeddings",
    )
    parser.add_argument(
        "--disable-discrete-mode",
        dest="disable_discrete_mode",
        default=False,
        action="store_true",
        help="Make the vq vae latent space non discretized",
    )
    parser.add_argument(
        "--disable-masking", dest="disable_masking", default=False, action="store_true"
    )
    parser.add_argument("--maskgit-dir", dest="maskgit_dir")
    parser.add_argument(
        "--vqvae-dir",
        dest="vqvae_dir",
        required=True,
    )
    parser.add_argument("--interp-mode", dest="interp_mode", default="zero")
    parser.add_argument(
        "--cuda",
        dest="cuda",
        default=False,
    )
    return parser.parse_args()


@torch.no_grad()
def main(stdscr, args):
    if not curses.can_change_color():
        raise Exception("Cannot change color")

    batch_size = 1

    dataset = AsciiArtDataset(
        res=100, ragged_batch_bin=True, ragged_batch_bin_batch_size=batch_size
    )

    if args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(device)

    dataloader = DataLoader(
        dataset,
        # This batch_size should be similar to what the
        # model was trained on for best results.
        batch_size=batch_size,
    )

    autoenc = VQ_VAE.load_from_checkpoint(
        args.vqvae_dir,
        device=device,
    )

    if not args.disable_masking:
        transformer = MaskGitTrainer.load_from_checkpoint(
            args.maskgit_dir,
            device=device,
            batch_size=1,
            max_res=100,
            validation_prop=0.0,
            vae=autoenc,
        )

    # the k value
    n_z_latents = autoenc.vq_k

    autoenc.eval()
    if args.cuda:
        autoenc.cuda()

    curses.noecho()
    curses.curs_set(False)
    rows, cols = stdscr.getmaxyx()
    window = curses.newwin(rows, cols)

    # Initializes color

    # The color 1 pair is the mask color of the
    # Latent space vis

    curses.init_color(1, 1000, 50, 1)
    curses.init_pair(1, 3, 1)

    # Color pair 2 is the ascii art color
    curses.init_color(2, 1, 1, 1)
    curses.init_pair(2, 2, 255)

    # The color pair i=2-255 is a gradient
    # Threshold for when to make the foreground text light colored
    thresh = 200
    lightness_diff = 30
    for i in range(3, 256):
        greyscale = int(1000 * i / 255)
        curses.init_color(i, greyscale, greyscale, greyscale)
        if i < thresh:
            foreground = min(i + lightness_diff, 255)
        else:
            foreground = max(2, i - lightness_diff)
        curses.init_pair(i, foreground, i)

    window.bkgd(" ", curses.color_pair(2) | curses.A_BOLD)

    next_frame = time.time() + 1 / args.frame_rate

    embedding2, embedding2_input_shape, embedding2_input, embedding2_label = get_random(
        device, dataset, autoenc.encoder
    )

    while True:
        embedding1, embedding1_input_shape, _ = (
            embedding2,
            embedding2_input_shape,
            embedding2_input,
        )
        (
            embedding2,
            embedding2_input_shape,
            embedding2_input,
            embedding2_label,
        ) = get_random(device, dataset, autoenc.encoder)
        embedding2_string = ascii_util.one_hot_embedded_matrix_to_string(
            embedding2_input.squeeze(0)
        )

        for x in np.linspace(0, 1, args.steps):
            while time.time() < next_frame:
                pass
            next_frame = time.time() + 1 / args.frame_rate

            x_scaled = np.log10(x**args.smooth_factor + 1) * 3.322

            interp_embedding = get_interp(
                embedding1, embedding2, x_scaled, interp_mode=args.interp_mode
            )

            # interpolated encoder input shape
            input_shape = int(
                lerp(embedding1_input_shape, embedding2_input_shape, x_scaled)
            )

            if not args.disable_discrete_mode:
                decoded, z_q_st, _ = autoenc.decode_from_z_e_x(
                    interp_embedding, x_res=input_shape
                )
            else:
                decoded = autoenc.decoder(interp_embedding, x_res=input_shape)
            decoded_str = ascii_util.one_hot_embedded_matrix_to_string(decoded[0])

            window.erase()

            y_shift = 10

            # Adds the original embedding2 to the right
            embedding2_y_shift = (min(rows, cols) - embedding2_input_shape) // 2
            embedding2_x_shift = cols - embedding2_input_shape - 10
            put_string(
                embedding2_string,
                rows,
                cols,
                window,
                y_shift=embedding2_y_shift + y_shift,
                x_shift=embedding2_x_shift,
            )

            # Adds the label to below the embedding2_string
            put_string(
                embedding2_label,
                rows,
                cols,
                window,
                y_shift=0,
                x_shift=embedding2_x_shift,
            )

            # If discrete mode shows the latent space
            if not args.disable_discrete_mode:
                indeces = autoenc.get_indeces_from_continuous(z_q_st)[0]
                y = draw_indeces(indeces, window, rows, cols, n_z_latents)
            else:
                y = 0

            # Pad shift places the decoded_str in the middle of the pad, in the
            # middle of where the embed_largest_input_shape would be
            pad_shift = min(rows, cols) - input_shape

            put_string(
                decoded_str,
                rows,
                cols,
                window,
                y_shift=pad_shift // 2 + y_shift,
                x_shift=pad_shift // 2,
                min_row=y + 1,
            )
            window.refresh()

        time.sleep(args.hold_length)

        if not args.disable_masking and not args.disable_discrete_mode:
            next_hold_until = time.time() + args.hold_length * 2
            # takes the indeces, masks them, and then reconstructs them using the transformer model
            logits, targets, a_indices = transformer.transformer(
                indeces.unsqueeze(0).reshape(1, -1),
                r=int(indeces.shape[-1] ** 2 * args.maskgit_r_prop),
            )
            # ignores the sos token
            a_indices = a_indices[0, 1:].reshape(indeces.shape[-1], indeces.shape[-1])
            logits = logits.argmax(-1)
            logits = logits[0, 1:].reshape(indeces.shape[-1], indeces.shape[-1])
            draw_indeces(
                a_indices,
                window,
                rows,
                cols,
                n_z_latents,
                mask_index=transformer.transformer.mask_token_id,
            )
            window.refresh()

            while time.time() < next_hold_until:
                pass
            next_hold_until = time.time() + args.hold_length * 2

            decoded = autoenc.decode_from_ids(logits.unsqueeze(0))
            decoded_str = ascii_util.one_hot_embedded_matrix_to_string(decoded[0])
            put_string(
                decoded_str,
                rows,
                cols,
                window,
                y_shift=pad_shift // 2 + y_shift,
                x_shift=pad_shift // 2,
                min_row=y + 1,
            )
            draw_indeces(
                logits,
                window,
                rows,
                cols,
                n_z_latents,
                mask_index=transformer.transformer.mask_token_id,
            )
            window.refresh()

            while time.time() < next_hold_until:
                pass

        window.clear()


def draw_indeces(indeces, window, rows, cols, n_z_latents, mask_index=-1):
    """
    Returns the last y value drawn to

    n_z_latents is the max value of indeces

    mask_index is the index of the mask
    """
    for y, row in enumerate(indeces):
        x = 0
        for entry in row:
            if bool(entry == mask_index):
                color = 1
            else:
                color = int((entry / n_z_latents) * 256)
                color = color % 254 + 2
            s_ent = "{:<4}".format(str(int(entry)))
            put_string(s_ent, rows, cols, window, y_shift=y, x_shift=x, color=color)
            x += len(s_ent)
    return y


def lerp(z, y, x):
    """
    Lin interp between z and y, with weight x
    """
    return z + x * (y - z)


def __trim(e, trim_amount):
    if trim_amount > 0:
        return e[
            :,
            :,
            trim_amount // 2 : -(trim_amount - trim_amount // 2),
            trim_amount // 2 : -(trim_amount - trim_amount // 2),
        ]
    else:
        return e


def __pad(e, pad_amount):
    return F.pad(
        e,
        (
            pad_amount // 2,
            pad_amount - pad_amount // 2,
            pad_amount // 2,
            pad_amount - pad_amount // 2,
        ),
    )


def get_interp(e1: torch.Tensor, e2: torch.Tensor, x: float, interp_mode="stretch"):
    """
    Interpolates linearly between possibly differently shaped square e1 and e2.
    e1 and e2 are 4d with the last two dimensions being square and first two dimensions,
    being the same between them.

    x is in [0, 1]

    if interp_mode is "stretch", then the embeddings will be interpolated into matching shapes
    if interp_mode is "zero", then the embeddings will be padded with zeros or
        trucated into matching shapes
    """
    e1_res = e1.shape[-1]
    e2_res = e2.shape[-1]
    interp_res = round(lerp(e1_res, e2_res, x))

    if interp_mode == "stretch":
        mode = "nearest"
        e1_interp = F.interpolate(e1, (interp_res, interp_res), mode=mode)
        e2_interp = F.interpolate(e2, (interp_res, interp_res), mode=mode)
    elif interp_mode == "zero":
        if e1_res < interp_res:
            pad_amount = interp_res - e1_res
            e1_interp = __pad(e1, pad_amount)
            trim_amount = e2_res - interp_res
            e2_interp = __trim(e2, trim_amount)

        elif e1_res > interp_res:
            trim_amount = e1_res - interp_res
            e1_interp = __trim(e1, trim_amount)
            pad_amount = interp_res - e2_res
            e2_interp = __pad(e2, pad_amount)
        else:
            e1_interp = e1
            pad_amount = interp_res - e2_res
            e2_interp = __pad(e2, pad_amount)
    else:
        raise Exception("wrong interp mode")

    return lerp(e1_interp, e2_interp, x)


def get_random(device, dataset, encoder):
    """returns random embedding, and the shape of the input, input
    doesn't return any dataset items with the word "sex" in it's label
    """
    bad_labels = ["sex", "naked", "penis", "gun", "swimsuit"]
    good_label = False
    while not good_label:
        img, label = dataset[random.randint(0, len(dataset) - 1)]
        img = img[0]
        label = label[0]

        found_bad = False
        for bad_label in bad_labels:
            if bad_label in label.lower():
                found_bad = True
                break
        good_label = not found_bad

    img = torch.FloatTensor(img).to(device)
    img = img.unsqueeze(0)
    with torch.no_grad():
        embedding = encoder(img)
    return embedding, img.shape[-1], img, label


def put_string(string, rows, cols, window, y_shift=0, x_shift=0, color=2, min_row=0):
    x_shift = max(x_shift, 0)
    for y, line in enumerate(string.splitlines()):
        if y + y_shift >= rows:
            break
        if y + y_shift < min_row:
            continue
        window.addstr(
            y + y_shift, x_shift, line[: cols - x_shift - 1], curses.color_pair(color)
        )


if __name__ in {"__main__", "__console__"}:
    args = get_args()
    curses.wrapper(main, args)
