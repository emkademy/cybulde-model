from torch import Tensor, nn


class Head(nn.Module):
    pass


class SoftmaxHead(Head):
    def __init__(self, in_features: int, out_features: int, dim: int = 1) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Linear(in_features, out_features), nn.Softmax(dim=dim))

    def forward(self, x: Tensor) -> Tensor:
        output: Tensor = self.head(x)
        return output


class SigmoidHead(Head):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        self.head = nn.Sequential(nn.Linear(in_features, out_features), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        output: Tensor = self.head(x)
        return output
