import torch


class RandomRoll(torch.nn.Module):
    """
    Rolls each input image in the input batch by a number from the distribution N(0, max_shift/2), or N(0, sigma) if sigma is specified.

    shifts will be <= max_shift
    """

    def __init__(self, max_shift: int, sigma: float = None):
        """
        max_shift: the maximum shift that can be applied
        mu: optional std of the distribution the shift will be sampled from
        """
        super().__init__()

        if not sigma:
            self.sigma = max_shift / 2
        else:
            self.sigma = sigma
        self.max_shift = max_shift

    def forward(self, x):
        batch_size = x.shape[0]
        shifts = torch.normal(0, self.sigma, size=(batch_size, 2))
        shifts = shifts.to(torch.int)
        shifts = shifts.clamp(-self.max_shift, self.max_shift)
        # Tensor.roll doesn't support batch rolling, so the following loop is unfortunately needed
        for i in range(batch_size):
            x[i] = x[i].roll(shifts=tuple(shifts[i].tolist()), dims=(1, 2))

        return x
