import latent_space_explorer
import torch

a = torch.randn(1, 1, 3, 3) * 100
b = torch.randn(1, 1, 27, 27)

for x in torch.linspace(0.0, 1.0, 10):
    print(x)
    latent_space_explorer.get_interp(b, a, float(x), interp_mode="zero")

for x in torch.linspace(0.0, 1.0, 10):
    print(x)
    latent_space_explorer.get_interp(a, b, float(x), interp_mode="zero")
