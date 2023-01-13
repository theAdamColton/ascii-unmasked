# Ascii Unmasked

This repository contains code for training a VQ-VAE and subsequent transformer on ascii art data. Art data used for this model was taken from multiple sources, mainly https://asciiart.website/, https://textart.io/, and https://www.asciiart.eu/. This project was only possible thanks to the labor of the talented ascii artists that contributed their work to the internet. 

### Mask-and-replace diffusion strategy

The architecture for this model was closely adapeted from [MaskGIT](https://openaccess.thecvf.com/content/CVPR2022/papers/Chang_MaskGIT_Masked_Generative_Image_Transformer_CVPR_2022_paper.pdf) and [Neural Discrete Representation Learning](https://arxiv.org/pdf/1711.00937v2.pdf). Code was adapted from [MaskGIT-pytorch](https://github.com/dome272/MaskGIT-pytorch) and [pytorch-vqvae](https://github.com/ritheshkumar95/pytorch-vqvae/)

### How to train a model

This has only been tested on a CUDA gpu using AMP precision settings. 

Follow instructions in the submodule ascii-dataset to gather the dataset and clean the files. Assuming the ascii files are in `./ascii-dataset/data_aggregation/data/**/*.txt`, you may train the autoencoder using `train_vqvae.py` the default arguments should be reasonable. Checkpoints will automatically be saved to `./ckpt/`. Note that some parameters such as the VQ Z dimension are hardcoded in the autoencoder architecture and cannot be changed from the command line without also adjusting the model in `autoencoder.py`.

Once the autoencoder is trained, you may train the masked transformer model using `train_vqmask.py`. You will have to specify the path to the autoencoder ckpt file. 
