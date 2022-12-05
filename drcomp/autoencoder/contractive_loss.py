import torch
from torch import Tensor, nn


class ContractiveLoss(nn.Module):
    def __init__(
        self,
        encoder,
        lambd: float,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.lambd = lambd

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return contractive_loss(input, target, self.lambd, self.ae, self.reduction)


def contractive_loss(input, target, lambd, ae, reduction: str):
    squared_error = (input - target) ** 2
    enc_weights = [ae.encoder[i].weight for i in reversed(range(1, len(ae.encoder), 2))]
    penalty = lambd * torch.norm(torch.chain_matmul(*enc_weights))
    contr_loss = torch.mean(squared_error + penalty, 0)
    if reduction == "mean":
        return torch.mean(contr_loss)
    elif reduction == "sum":
        return torch.sum(contr_loss)
    else:
        raise ValueError(
            f"value for 'reduction' must be 'mean' or 'sum', got {reduction}"
        )
