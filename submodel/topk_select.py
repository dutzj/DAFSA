import torch
from torch import Tensor

from torch_geometric.nn.inits import uniform
from torch_geometric.nn.pool.select import Select
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import cumsum, scatter, softmax
from typing import Callable, Optional, Union, Any


def topk(
    score: Tensor,
    ratio: Optional[Union[float, int]],
    batch: Tensor,
    min_score: Optional[float] = None,
    tol: float = 1e-7,
) -> Tensor:
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter(score, batch, reduce='max')[batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (score > scores_min).nonzero().view(-1)
        return perm

    if ratio is not None:
        num_nodes = scatter(batch.new_ones(score.size(0)), batch, reduce='sum')

        if ratio >= 1:
            k = num_nodes.new_full((num_nodes.size(0), ), int(ratio))
        else:
            k = (float(ratio) * num_nodes.to(score.dtype)).ceil().to(torch.long)

        score, x_perm = torch.sort(score.view(-1), descending=True)
        batch = batch[x_perm]
        batch, batch_perm = torch.sort(batch, descending=False, stable=True)

        arange = torch.arange(score.size(0), dtype=torch.long, device=score.device)
        ptr = cumsum(num_nodes)
        batched_arange = arange - ptr[batch]
        mask = batched_arange < k[batch]

        return x_perm[batch_perm[mask]]

    raise ValueError("At least one of the 'ratio' and 'min_score' parameters "
                     "must be specified")


class SelectTopK(Select):

    def __init__(
        self,
        in_channels: int,
        ratio: Union[int, float] = 0.5,
        need_softmax: bool = False,
        min_score: Optional[float] = None,
        activate: Union[str, Callable] = 'tanh',
    ):
        super().__init__()

        if ratio is None and min_score is None:
            raise ValueError(f"At least one of the 'ratio' and 'min_score' "
                             f"parameters must be specified in "
                             f"'{self.__class__.__name__}'")

        self.in_channels = in_channels
        self.ratio = ratio
        self.need_softmax = need_softmax
        self.min_score = min_score
        if min_score is not None:
            assert activate is not None
            self.activate = activation_resolver(activate)
        self.weight = torch.nn.Parameter(torch.empty(1, in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)

    def forward(
        self,
        score: Tensor,
        batch: Optional[Tensor] = None,
    ) -> tuple[Any, Tensor, Tensor]:

        if self.min_score is not None:
            score = self.activate(score / self.weight.norm(p=2, dim=-1))
        else:
            if self.need_softmax:
                score = softmax(score, batch)
            else:
                score = score
        topk_index = topk(score, self.ratio, batch, self.min_score)
        topk_index = torch.sort(topk_index, descending=False)[0]

        return topk_index, score[topk_index], batch[topk_index]

    def __repr__(self) -> str:
        if self.min_score is None:
            arg = f'ratio={self.ratio}'
        else:
            arg = f'min_score={self.min_score}'
        return f'{self.__class__.__name__}({self.in_channels}, {arg})'


