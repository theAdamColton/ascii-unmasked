"""
Traverses the latent space
Make sure that your terminal is large enough, or curses will throw an error
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
import bpdb

dirname = path.dirname(__file__)
sys.path.insert(0, path.join(dirname, "./ascii-dataset/"))
import ascii_util
from dataset import AsciiArtDataset

dirname = path.dirname(__file__)
sys.path.insert(0, path.join(dirname, "./python-pytorch-font-renderer/"))
from font_renderer import ContinuousFontRenderer

from vqvae import VQ_VAE


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--frame-rate", dest="frame_rate", type=float, default=60)
    parser.add_argument("--steps", dest="steps", type=int, default=200)
    parser.add_argument("--hold-length", dest="hold_length", default=2, type=float)
    parser.add_argument(
        "--smooth-factor",
        dest="smooth_factor",
        default=1.0,
        type=float,
        help="Any number in [0,1], represents the smoothing between the different embeddings",
    )
    parser.add_argument(
        "--model-dir",
        dest="model_dir",
        default=path.join(
            path.dirname(__file__),
            "models/autoenc_vanilla_deep_cnn_one_hot_64_with_noise",
        ),
    )
    parser.add_argument(
        "--cuda",
        dest="cuda",
        default=False,
    )
    return parser.parse_args()


@torch.no_grad()
def main(stdscr, args):
    dataset = AsciiArtDataset(res=64, ragged_batch_bin=True, ragged_batch_bin_batch_size=1)

    cuda = args.cuda
    if cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(device)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
    )

    autoenc = VQ_VAE.load_from_checkpoint(
        args.model_dir,
        device=device,
    )

    autoenc.eval()
    if cuda:
        autoenc.cuda()

    curses.noecho()
    curses.curs_set(False)
    rows, cols = stdscr.getmaxyx()
    pad = curses.newpad(rows, cols)

    assert rows >= 64 and cols >= 64, "Terminal size needs to be at least 64x64"

    next_frame = time.time() + 1 / args.frame_rate
    embedding2, embedding2_input_shape = get_random(device, dataset, autoenc.encoder)
    while True:
        embedding1, embedding1_input_shape = embedding2, embedding2_input_shape
        embedding2, embedding2_input_shape = get_random(device, dataset, autoenc.encoder)

        if embedding2.shape[-1] > embedding1.shape[-1]:
            embed_smallest = embedding1
            embed_largest = embedding2
            embed_smallest_shape = embedding1.shape[-1]
            embed_largest_shape = embedding2.shape[-1]
            embed_largest_input_shape = embedding2_input_shape
        else:
            embed_smallest = embedding2
            embed_largest = embedding1
            embed_smallest_shape = embedding2.shape[-1]
            embed_largest_shape = embedding1.shape[-1]
            embed_largest_input_shape = embedding1_input_shape

        embed_shape_diff = abs(embedding1.shape[-1] - embedding2.shape[-1])
        input_shape_diff = abs(embedding1_input_shape - embedding2_input_shape)


        """
        embedding1_shape and embedding2_shape might be different.
        How we handle that:
            * The dimensions of the interpolated embedding will linearly
            interpolate between the integer steps between the integer sizes of
            the two embeddings.

            * The rendered intermediate output will be padded with space
            characters to be in the center.
        """

        for x in np.linspace(0, 1, args.steps):

            while time.time() < next_frame:
                pass

            next_frame = time.time() + 1 / args.frame_rate

            x_scaled = np.log10(x**args.smooth_factor + 1) * 3.322
            # interpolated embedding shape
            embed_shape = int(embed_smallest_shape + round(embed_shape_diff * x_scaled))
            padding = embed_shape - embed_smallest_shape
            trimming = embed_largest_shape - embed_shape
            embed_smallest_padded = F.pad(embed_smallest, (padding//2, padding - padding//2, padding//2, padding - padding//2))
            if trimming != 0:
                embed_largest_trimmed = embed_largest[:, :, trimming//2:-(trimming - trimming//2), trimming//2:-(trimming - trimming//2)]
            else:
                embed_largest_trimmed = embed_largest

            if not embed_largest_trimmed.shape == embed_smallest_padded.shape:
                bpdb.set_trace()

            if embedding1.shape[-1] < embedding2.shape[-1]:
                interp_embedding = x_scaled * embed_largest_trimmed + (1 - x) * embed_smallest_padded
            else:
                interp_embedding = x_scaled * embed_smallest_padded + (1 - x) * embed_largest_trimmed

            # interpolated encoder input shape
            if embedding1.shape[-1] < embedding2.shape[-1]:
                input_shape = int(input_shape_diff * x_scaled + embedding1_input_shape)
            else:
                input_shape = int(input_shape_diff * x_scaled + embedding2_input_shape)

            decoded = autoenc.decoder(interp_embedding, x_res=input_shape)
            decoded_str = ascii_util.one_hot_embedded_matrix_to_string(decoded[0])
            #bpdb.set_trace()

            # Pad shift places the decoded_str in the middle of the pad, in the
            # middle of where the embed_largest_input_shape would be
            pad_shift = embed_largest_input_shape - input_shape
            pad.addstr(pad_shift//2, pad_shift - pad_shift//2, decoded_str)
            pad.refresh(0, 0, 0, 0, 64, 64)

        time.sleep(args.hold_length)



def get_random(device, dataset, encoder):
    """returns random embedding, and the shape of the input"""
    img, _ = dataset[random.randint(0, len(dataset) - 1)]
    img = img[0]
    img = torch.FloatTensor(img).to(device)
    img = img.unsqueeze(0)
    with torch.no_grad():
        embedding = encoder(img)
    return embedding, img.shape[-1]


if __name__ in {"__main__", "__console__"}:
    args = get_args()
    curses.wrapper(main, args)
