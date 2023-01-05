### Mask-and-replace diffusion strategy

From the paper https://arxiv.org/pdf/2111.14822.pdf Vector Quantized Diffusion Model for Text-to-Image Synthesis

"corrupt the tokens by stochastically masking some of them so that the corrupted locations can be explicitly known by the reverse network."

Each token will have K+1 states, because of the addition of an extra mask token. 

The current timestep t is injected into the network with Adaptive Layer Normalization (AdaLN) operator, i.e., AdaLN(h, t) = atLayerNorm(h) + bt, where h is the intermediate activations, at and bt are obtained from a linear projection of the timestep embedding.
