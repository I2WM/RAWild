import torch
import torch.nn as nn


class RGBuvHistogram(nn.Module):
    """Differentiable RGB-uv histogram used as RAWild AdaLN condition input."""

    def __init__(self, num_bins=64, sigma=0.1, val_range=(-3.0, 3.0),
                 downsample_factor=16):
        super().__init__()
        self.num_bins = num_bins
        self.sigma = sigma
        self.val_range = val_range
        self.downsample_factor = downsample_factor
        if downsample_factor > 1:
            self.pool = nn.AvgPool2d(
                kernel_size=downsample_factor, stride=downsample_factor)
        else:
            self.pool = nn.Identity()

        bin_centers = torch.linspace(val_range[0], val_range[1], num_bins)
        self.register_buffer('bin_centers', bin_centers)

    def forward(self, x):
        batch = x.shape[0]
        eps = 1e-6

        x_pooled = self.pool(x)
        x_flat = x_pooled.view(batch, 3, -1)

        r = x_flat[:, 0:1, :] + eps
        g = x_flat[:, 1:2, :] + eps
        b = x_flat[:, 2:3, :] + eps

        u = torch.log(r / g)
        v = torch.log(b / g)

        bin_centers = self.bin_centers.view(1, self.num_bins, 1)
        u_weights = torch.exp(-0.5 * ((u - bin_centers) / self.sigma) ** 2)
        v_weights = torch.exp(-0.5 * ((v - bin_centers) / self.sigma) ** 2)

        hist = torch.bmm(u_weights, v_weights.transpose(1, 2))
        hist = hist / hist.sum(dim=[1, 2], keepdim=True).clamp(min=eps)
        return hist.unsqueeze(1)
