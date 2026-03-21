import torch
import torch.nn as nn


class INR(nn.Module):
    """
    A small MLP that predicts pixel values from (x,y) coordinates.
    Weights are supplied externally by the hypernetwork (VAE decoder).
    """

    def __init__(self, coord_dim=2, hidden_dim=20, n_hidden=2, out_dim=1):
        super().__init__()
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.out_dim = out_dim

        # Compute the total number of weights this INR needs
        dims = [coord_dim] + [hidden_dim] * n_hidden + [out_dim]
        self.weight_shapes = []
        self.bias_shapes = []
        total = 0
        for i in range(len(dims) - 1):
            self.weight_shapes.append((dims[i + 1], dims[i]))
            self.bias_shapes.append((dims[i + 1],))
            total += dims[i + 1] * dims[i] + dims[i + 1]
        self.num_weights = total

    def forward(self, coords, flat_weights):
        """
        coords:       (batch, n_pixels, 2)
        flat_weights: (batch, num_weights)
        returns:      (batch, n_pixels, 1)  — logits
        """
        batch = coords.shape[0]
        idx = 0
        x = coords  # (batch, n_pixels, coord_dim)

        for i, (ws, _bs) in enumerate(zip(self.weight_shapes, self.bias_shapes)):  # noqa: B905
            out_f, in_f = ws
            W = flat_weights[:, idx : idx + out_f * in_f].view(batch, out_f, in_f)  # noqa: N806
            idx += out_f * in_f
            b = flat_weights[:, idx : idx + out_f].view(batch, 1, out_f)
            idx += out_f

            # (batch, n_pixels, out_f)
            x = torch.bmm(x, W.transpose(1, 2)) + b

            # ReLU on hidden layers, sigmoid on last
            x = torch.relu(x) if i < len(self.weight_shapes) - 1 else torch.sigmoid(x)

        return x  # (batch, n_pixels, 1)
