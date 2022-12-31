
# How I choose to define my Markov transition matrix

Conceptually, in the forward process I want character vectors to be more likely
to be corrupted into a character vector that is similar in appearance to the
current character. For example, `-` should be more likely to be corrupted into
`_` than the character `@`.

### Structured Denoising Diffusion Models in Discrete State-Spaces https://arxiv.org/pdf/2107.03006.pdf

Token embedding distance (Appendix A.2.4). Textual data does not have ordinal structure, but
there may still be interesting semantic relationships. For instance, in a character level vocabulary
vowels may be more similar to each other than they are to consonants. As a demonstration of the
generality of the D3PM framework, we explore using similarity in an embedding space to guide the
forward process, and construct a doubly-stochastic transition matrix that transitions more frequently
between tokens that have similar embeddings while maintaining a uniform stationary distribution.
