"""
Performs PCA on character images.
This is for the purpose of obtaining character embeddings based on pixel
appearance of the characters, which can be useful in experimenting with the
Markov transition matrix in the forward diffusion process.

Adapted from https://medium.com/@sebastiannorena/pca-principal-components-analysis-applied-to-images-of-faces-d2fc2c083371
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy
from PIL import Image
import os

import string_utils

imres = 64  # 64x64 hardcoded size of the character images
font_im_dir = os.path.join(
    os.path.abspath(
        os.path.dirname(__file__) + "/python-pytorch-font-renderer/font_images/"
    )
)
imfiles = os.listdir(font_im_dir)


def demonstration():
    n_comps = 8
    embeddings, projected = generate_character_embeddings(n_comps)

    if input("Show images?").startswith("y"):
        print(f"Images generated with primary {n_comps} components")
        for i, proj in enumerate(projected):
            im = proj.reshape(imres, imres)
            plt.imshow(im, cmap="gray")
            imnum = int(string_utils.remove_suffix(imfiles[i], ".png"))
            plt.title("im {} char '{}'".format(imnum, chr(imnum + 32)))
            plt.show()

    embeddings_2d, _ = generate_character_embeddings(2)

    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    plt.title("2D character embeddings")

    for i in range(embeddings_2d.shape[0]):
        plt.text(
            x=embeddings_2d[i, 0] + 0.3,
            y=embeddings_2d[i, 1] + 0.3,
            s=chr(i + 32),
            fontdict={"size": 26},
        )
    plt.show()

    if input("save embeddings?").startswith("y"):
        with open("./character_embeddings.npy", "wb") as f:
            np.save(f, embeddings_2d)


def generate_character_embeddings(n_components=12):
    """
    Returns (embeddings, reprojected)
    embeddings is (95 x n_components) for ascii characters starting from 32,
    and going to 126.
    """
    images = np.zeros([len(imfiles), 64, 64])

    for i, imfile in enumerate(imfiles):
        with Image.open(os.path.join(font_im_dir, imfile)) as im:
            im_arr = np.asarray(im)
            images[i] = im_arr

    images_flattened = images.reshape(images.shape[0], imres**2)
    p = PCA(n_components=n_components)
    images_pca = p.fit(images_flattened)

    transformed_comps = images_pca.transform(images_flattened)
    projected = images_pca.inverse_transform(transformed_comps)
    return (transformed_comps, projected)


def generate_embedding_space_distances(n_components=12):
    """
    Generates the PCA embeddings for the characters, and computes pairwise
    euclidian distances between each character embedding. Returns a 95x95
    matrix where M[i,j] is the distance in the embedding space of character i
    to character j.
    """
    embeddings, _ = generate_character_embeddings(n_components=n_components)
    
    distances = scipy.spatial.distance.cdist(embeddings, embeddings, metric='euclidean')
    return distances

if __name__ in {"__main__", "__console__"}:
    if input("Create PCA plot?").startswith("y"):
        demonstration()
    else:
        ncomp = int(input("Number of components?"))
        transformed_comps, _ = generate_character_embeddings(n_components=ncomp)
        filename = input("Filename?")
        with open(filename, "wb") as f:
            np.save(f, transformed_comps)
        print("Saved.")
