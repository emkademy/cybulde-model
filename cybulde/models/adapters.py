from operator import attrgetter
from typing import Literal, Optional

import torch
import torch.nn.functional as F

from torch import Tensor, nn
from transformers.modeling_outputs import BaseModelOutputWithPooling


class Adapter(nn.Module):
    pass


class Normalization(nn.Module):
    def __init__(self, p: float = 2.0) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, p=self.p, dim=1)


class FCLayer(Adapter):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        activation_fn: Optional[nn.Module] = None,
        dropout: float = 0.0,
        batch_norm: bool = False,
        order: str = "LABDN",
    ) -> None:
        super().__init__()

        order = order.upper()

        layers: dict[str, tuple[str, nn.Module]] = {"L": ("linear", nn.Linear(in_features, out_features, bias=bias))}

        if activation_fn is not None:
            layers["A"] = ("activation_fn", activation_fn)

        if batch_norm:
            layers["B"] = (
                "batch_norm",
                nn.BatchNorm1d(out_features if order.index("L") < order.index("B") else in_features),
            )

        if dropout > 0.0:
            layers["D"] = ("dropout", nn.Dropout(dropout))

        if "N" in order:
            layers["N"] = ("normalization", Normalization())

        self.layers = nn.Sequential()
        for layer_code in order:
            if layer_code in layers:
                name, layer = layers[layer_code]
                self.layers.add_module(name, layer)

    def forward(self, x: Tensor) -> Tensor:
        output: Tensor = self.layers(x)
        return output


class MLPLayer(Adapter):
    def __init__(
        self,
        output_feature_sizes: list[int],
        biases: Optional[list[bool]] = None,
        activation_fns: Optional[list[Optional[str]]] = None,
        dropout_drop_probs: Optional[list[float]] = None,
        batch_norms: Optional[list[bool]] = None,
        order: str = "LABDN",
        standardize_input: bool = True,
    ) -> None:
        super().__init__()

        self.output_feature_sizes = output_feature_sizes
        self.output_embedding_size = output_feature_sizes[-1]

        nrof_layers = len(self.output_feature_sizes) - 1
        biases = [False] * nrof_layers if biases is None else biases
        activation_functions: list[Optional[str]] = [None] * nrof_layers if activation_fns is None else activation_fns
        dropout_drop_probabilities = [0.0] * nrof_layers if dropout_drop_probs is None else dropout_drop_probs
        batch_normalizations = [False] * nrof_layers if batch_norms is None else batch_norms

        assert (
            nrof_layers
            == len(activation_functions)
            == len(dropout_drop_probabilities)
            == len(batch_normalizations)
            == len(biases)
        )

        self.adapter = nn.Sequential()

        if standardize_input:
            self.adapter.add_module(
                "standardize_input", nn.LayerNorm(output_feature_sizes[0], elementwise_affine=False)
            )

        for i in range(nrof_layers):
            activation_function = activation_functions[i]
            self.adapter.add_module(
                f"fc_layer_{i}",
                FCLayer(
                    in_features=output_feature_sizes[i],
                    out_features=output_feature_sizes[i + 1],
                    bias=biases[i],
                    activation_fn=getattr(nn, activation_function)() if activation_function is not None else None,
                    dropout=dropout_drop_probabilities[i],
                    batch_norm=batch_normalizations[i],
                    order=order,
                ),
            )

    def forward(self, backbone_output: Tensor) -> Tensor:
        output: Tensor = self.adapter(backbone_output)
        return output


class MLPWithPooling(Adapter):
    def __init__(
        self,
        output_feature_sizes: list[int],
        biases: Optional[list[bool]] = None,
        activation_fns: Optional[list[Optional[str]]] = None,
        dropout_drop_probs: Optional[list[float]] = None,
        batch_norms: Optional[list[bool]] = None,
        order: str = "LABDN",
        standardize_input: bool = True,
        pooling_method: Optional[str] = None,
        output_attribute_to_use: Optional[Literal["pooler_output", "last_hidden_state"]] = None,
    ) -> None:
        super().__init__()

        self.output_feature_sizes = output_feature_sizes
        self.output_embedding_size = output_feature_sizes[-1]

        nrof_layers = len(output_feature_sizes) - 1
        if nrof_layers > 0:
            self.projection = MLPLayer(
                output_feature_sizes=output_feature_sizes,
                biases=biases,
                activation_fns=activation_fns,
                dropout_drop_probs=dropout_drop_probs,
                batch_norms=batch_norms,
                order=order,
                standardize_input=standardize_input,
            )
        else:
            self.projection = nn.Identity()  # type: ignore

        if pooling_method == "mean_pooler":
            self.pooler = mean_pool_tokens
        elif pooling_method == "cls_pooler":
            self.pooler = cls_pool_tokens
        else:
            self.pooler = nn.Identity()

        if output_attribute_to_use is not None:
            self.get_output_tensor = attrgetter(output_attribute_to_use)
        else:
            self.get_output_tensor = nn.Identity()  # type: ignore

    def forward(self, backbone_output: BaseModelOutputWithPooling) -> Tensor:
        output = self.get_output_tensor(backbone_output)
        output = self.pooler(output)
        output = self.projection(output)
        assert isinstance(output, Tensor)
        return output


def mean_pool_tokens(tensor: Tensor) -> Tensor:
    dims = len(tensor.shape)
    if dims != 3:
        raise ValueError(f"Tokens pooling expects exactly 3 dimensional tensor, got: {dims}")
    return torch.mean(tensor, dim=1)


def cls_pool_tokens(tensor: Tensor) -> Tensor:
    dims = len(tensor.shape)
    if dims != 3:
        raise ValueError(f"Tokens pooling expects exactly 3 dimensional tensor, got: {dims}")
    return tensor[:, 0, :]
