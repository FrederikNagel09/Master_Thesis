import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Batched linear layer helper (used by SIREN)
# ---------------------------------------------------------------------------


def batched_linear_mm(x, wb):
    """
    x  : (B, N, D1)
    wb : (B, D1+1, D2) OR (1, D1+1, D2) — last row is the bias
    """
    # 1. Ensure x has the correct batch dimension
    B = x.shape[0]  # noqa: N806

    # 2. Create bias vector
    one = torch.ones(B, x.shape[1], 1, device=x.device)
    x_cat = torch.cat([x, one], dim=-1)  # (B, N, D1+1)

    # 3. Ensure wb is (B, D1+1, D2)
    # If wb is (D1+1, D2), add batch dim: (1, D1+1, D2)
    if wb.dim() == 2:
        wb = wb.unsqueeze(0)

    # 4. Perform batch matrix multiplication
    # (B, N, D1+1) @ (B, D1+1, D2) -> (B, N, D2)
    return torch.matmul(x_cat, wb)


# ---------------------------------------------------------------------------
# SIREN  (unchanged from original)
# ---------------------------------------------------------------------------


class SIREN(nn.Module):
    def __init__(
        self,
        depth,
        in_dim,
        out_dim,
        hidden_dim,
        out_bias=0,
        omega=30.0,
        out_activation="tanh",
    ):
        super().__init__()
        self.omega = omega
        self.depth = depth
        self.out_activation = out_activation  # "tanh" or "sigmoid"
        self.param_shapes = dict()  # noqa: C408

        last_dim = in_dim
        for i in range(depth):
            cur_dim = hidden_dim if i < depth - 1 else out_dim
            self.param_shapes[f"wb{i}"] = (last_dim + 1, cur_dim)
            last_dim = cur_dim

        self.params = None
        self.out_bias = out_bias

    def siren_activation(self, x):
        return torch.sin(self.omega * x)

    def init_wb(self, shape, name):
        if name == "wb0":
            num_input = shape[0] - 1
            bound = 1 / num_input
            weight = torch.empty(shape[1], shape[0] - 1)
            nn.init.uniform_(weight, -bound, bound)
            bias = torch.zeros(shape[1], 1)
            return torch.cat([weight, bias], dim=1).t().detach()
        else:
            num_input = shape[0] - 1
            bound = np.sqrt(6 / num_input) / self.omega
            weight = torch.empty(shape[1], shape[0] - 1)
            nn.init.uniform_(weight, -bound, bound)
            bias = torch.zeros(shape[1], 1)
            return torch.cat([weight, bias], dim=1).t().detach()

    def set_params(self, params):
        self.params = params

    def forward(self, x):
        B, query_shape = x.shape[0], x.shape[1:-1]  # noqa: N806
        x = x.view(B, -1, x.shape[-1])
        for i in range(self.depth):
            x = batched_linear_mm(x, self.params[f"wb{i}"])
            x = (
                self.siren_activation(x)
                if i < self.depth - 1
                else torch.sigmoid(x)
                if self.out_activation == "sigmoid"
                else torch.tanh(x)
            )
        x = x.view(B, *query_shape, -1)
        return x

    def get_last_layer(self):
        return self.params[f"wb{self.depth - 1}"]
