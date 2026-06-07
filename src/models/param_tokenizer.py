"""Tokenizers for flat INR parameter vectors."""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ParamTokenSpec:
    name: str
    shape: tuple
    start: int
    end: int
    token_type: int
    column: int | None
    raw_dim: int


def _numel(shape):
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


def build_param_token_specs(param_shapes):
    """Create token specs from parameter shapes using weight columns as tokens."""
    specs = []
    offset = 0
    for name, shape in param_shapes.items():
        shape = tuple(shape)
        numel = _numel(shape)
        if len(shape) == 2 and (name.endswith("weight") or name.endswith("weight_modulation") or name.endswith("bias_modulation")):
            out_dim, in_dim = shape
            for column in range(in_dim):
                specs.append(
                    ParamTokenSpec(
                        name=name,
                        shape=shape,
                        start=offset,
                        end=offset + numel,
                        token_type=1 if name.endswith("bias_modulation") else 0,
                        column=column,
                        raw_dim=out_dim,
                    )
                )
        else:
            specs.append(
                ParamTokenSpec(
                    name=name,
                    shape=shape,
                    start=offset,
                    end=offset + numel,
                    token_type=1 if name.endswith("bias") else 2,
                    column=None,
                    raw_dim=numel,
                )
            )
        offset += numel
    return specs, offset


def _read_token(params, spec):
    chunk = params[:, spec.start : spec.end]
    if spec.column is None:
        return chunk
    tensor = chunk.reshape(params.shape[0], *spec.shape)
    return tensor[:, :, spec.column]


def _write_token(output, values, spec):
    if spec.column is None:
        output[:, spec.start : spec.end] = values
        return
    tensor = output[:, spec.start : spec.end].reshape(output.shape[0], *spec.shape)
    tensor[:, :, spec.column] = values


def _projection(in_dim, out_dim):
    if in_dim == out_dim:
        return nn.Identity()
    return nn.Linear(in_dim, out_dim)


class ParamTokenizer(nn.Module):
    """Project flat parameter vectors into weight-column tokens."""

    def __init__(self, param_shapes, hidden_dim, max_token_types=3):
        super().__init__()
        self.specs, self.num_params = build_param_token_specs(param_shapes)
        self.hidden_dim = hidden_dim
        self.projections = nn.ModuleList(_projection(spec.raw_dim, hidden_dim) for spec in self.specs)
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
        return token_tensor + self.position_embedding.unsqueeze(0) + self.type_embedding(type_ids).unsqueeze(0)


class ParamDetokenizer(nn.Module):
    """Map parameter tokens back to the original flat parameter layout."""

    def __init__(self, token_specs, num_params, hidden_dim):
        super().__init__()
        self.specs = list(token_specs)
        self.num_params = num_params
        self.projections = nn.ModuleList(_projection(hidden_dim, spec.raw_dim) for spec in self.specs)

    def forward(self, tokens):
        output = tokens.new_zeros(tokens.shape[0], self.num_params)
        for token_idx, (spec, projection) in enumerate(zip(self.specs, self.projections)):  # noqa: B905
            values = projection(tokens[:, token_idx])
            _write_token(output, values, spec)
        return output
