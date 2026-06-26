"""HyperDiffusion-style tokenizers for flat parameter vectors."""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class HyperDiffTokenSpec:
    name: str
    shape: tuple
    start: int
    end: int
    token_start: int
    token_end: int
    token_type: int
    chunk: int
    num_chunks: int
    raw_dim: int


def _numel(shape):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def _projection(in_dim, out_dim):
    if in_dim == out_dim:
        return nn.Identity()
    return nn.Linear(in_dim, out_dim)


def _token_type(name):
    if name.endswith("bias") or name.endswith("bias_modulation"):
        return 1
    if name.endswith("weight") or name.endswith("weight_modulation"):
        return 0
    return 2


def build_hyperdiff_token_specs(param_shapes, tokens_per_tensor=1, chunk_size=None):
    """Create token specs by chunking each parameter tensor independently."""
    specs = []
    offset = 0
    for name, shape in param_shapes.items():
        shape = tuple(shape)
        numel = _numel(shape)
        if chunk_size is not None:
            num_chunks = max(1, (numel + int(chunk_size) - 1) // int(chunk_size))
        else:
            num_chunks = max(1, min(int(tokens_per_tensor), numel))
        chunk_width = (numel + num_chunks - 1) // num_chunks

        for chunk_idx in range(num_chunks):
            token_start = chunk_idx * chunk_width
            token_end = min(numel, token_start + chunk_width)
            if token_start >= token_end:
                continue
            specs.append(
                HyperDiffTokenSpec(
                    name=name,
                    shape=shape,
                    start=offset,
                    end=offset + numel,
                    token_start=token_start,
                    token_end=token_end,
                    token_type=_token_type(name),
                    chunk=chunk_idx,
                    num_chunks=num_chunks,
                    raw_dim=token_end - token_start,
                )
            )
        offset += numel
    return specs, offset


def _read_token(params, spec):
    chunk = params[:, spec.start : spec.end]
    return chunk[:, spec.token_start : spec.token_end]


def _write_token(output, values, spec):
    output[:, spec.start + spec.token_start : spec.start + spec.token_end] = values


class HyperDiffTokenizer(nn.Module):
    """Project layer/tensor chunks into transformer tokens."""

    def __init__(
        self,
        param_shapes,
        hidden_dim,
        tokens_per_tensor=1,
        chunk_size=None,
        max_token_types=3,
    ):
        super().__init__()
        self.specs, self.num_params = build_hyperdiff_token_specs(
            param_shapes,
            tokens_per_tensor=tokens_per_tensor,
            chunk_size=chunk_size,
        )
        self.hidden_dim = hidden_dim
        self.projections = nn.ModuleList(
            _projection(spec.raw_dim, hidden_dim) for spec in self.specs
        )
        self.position_embedding = nn.Parameter(torch.zeros(len(self.specs), hidden_dim))
        self.type_embedding = nn.Embedding(max_token_types, hidden_dim)

    @property
    def num_tokens(self):
        return len(self.specs)

    def forward(self, params):
        tokens = []
        type_ids = []
        for spec, projection in zip(self.specs, self.projections):  # noqa: B905
            tokens.append(projection(_read_token(params, spec)))
            type_ids.append(spec.token_type)
        token_tensor = torch.stack(tokens, dim=1)
        type_ids = torch.tensor(type_ids, device=params.device, dtype=torch.long)
        return (
            token_tensor
            + self.position_embedding.unsqueeze(0)
            + self.type_embedding(type_ids).unsqueeze(0)
        )


class HyperDiffDetokenizer(nn.Module):
    """Map HyperDiffusion-style chunk tokens back to flat parameters."""

    def __init__(self, token_specs, num_params, hidden_dim):
        super().__init__()
        self.specs = list(token_specs)
        self.num_params = num_params
        self.projections = nn.ModuleList(
            _projection(hidden_dim, spec.raw_dim) for spec in self.specs
        )

    def forward(self, tokens):
        output = tokens.new_zeros(tokens.shape[0], self.num_params)
        for token_idx, (spec, projection) in enumerate(
            zip(self.specs, self.projections)
        ):  # noqa: B905
            values = projection(tokens[:, token_idx])
            _write_token(output, values, spec)
        return output
