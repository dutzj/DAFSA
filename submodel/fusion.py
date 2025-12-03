import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax

from typing import Optional


class IntraInterFusionModal(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.intra_1 = nn.Linear(in_features=input_dim, out_features=output_dim)
        self.intra_2 = nn.Linear(in_features=input_dim, out_features=output_dim)
        self.inter_1 = nn.Linear(in_features=input_dim, out_features=output_dim)
        self.inter_2 = nn.Linear(in_features=input_dim, out_features=output_dim)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, intra, inter):
        intra = self.tanh(self.intra_1(intra))
        inter = self.tanh(self.inter_1(inter))
        gate = self.sigmoid(self.intra_2(intra) + self.inter_2(inter))
        fusion = gate * intra + (1 - gate) * inter
        return fusion


class FrameUpdate(pyg_nn.MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attention: bool = True,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        if attention:
            out_channels = out_channels // heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attention = attention
        self.concat = concat
        if self.attention:
            self.heads = heads
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.lin = Linear(in_channels, heads * out_channels, bias=False, weight_initializer='glorot')
            self.att = Parameter(torch.empty(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.lin = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')

        if bias and concat:
            self.bias = Parameter(torch.empty(self.heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()

        if self.attention:
            glorot(self.att)
        zeros(self.bias)

    @disable_dynamic_shapes(required_args=['num_edges'])
    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_attr: Optional[Tensor] = None,
                num_edges: Optional[int] = None):

        num_nodes = x.size(0)

        if num_edges is None:
            num_edges = 0
            if hyperedge_index.numel() > 0:
                num_edges = int(hyperedge_index[1].max()) + 1

        x = self.lin(x)

        alpha = None

        if self.attention:
            assert hyperedge_attr is not None
            x = x.view(-1, self.heads, self.out_channels)
            hyperedge_attr = self.lin(hyperedge_attr)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads, self.out_channels)
            x_i = x[hyperedge_index[0]]
            x_j = hyperedge_attr[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            if self.attention:
                alpha = softmax(alpha, hyperedge_index[1], num_nodes=num_edges)

        out = self.propagate(hyperedge_index, x=x, alpha=alpha, size=(num_nodes, num_edges))
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * x_j
        else:
            out = x_j
        return out


class SemanticFusionModule(nn.Module):
    def __init__(self, model_dim, attention=True, num_heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True):
        super().__init__()
        self.bn_intra = pyg_nn.GraphNorm(model_dim)
        self.bn_inter = pyg_nn.GraphNorm(model_dim)

        self.frame_intra = FrameUpdate(
            in_channels=model_dim, out_channels=model_dim, attention=attention,
            heads=num_heads, negative_slope=negative_slope,dropout=dropout, bias=bias
        )
        self.frame_inter = FrameUpdate(
            in_channels=model_dim, out_channels=model_dim, attention=attention,
            heads=num_heads, negative_slope=negative_slope, dropout=dropout, bias=bias
        )

    def forward(self, intra, inter):
        f_intra = self.frame_intra(x=self.bn_intra(intra.x), hyperedge_index=intra.edge_index,
                                   hyperedge_attr=intra.edge_attr, num_edges=intra.num_edges.sum())
        f_inter = self.frame_inter(x=self.bn_inter(inter.x), hyperedge_index=inter.edge_index,
                                   hyperedge_attr=inter.edge_attr, num_edges=inter.num_edges.sum())
        fusion = torch.cat((
            torch.mean(f_intra, dim=0),
            torch.mean(f_inter, dim=0),
            torch.mean(intra.x, dim=0),
            torch.mean(inter.x, dim=0)
        ), dim=-1)
        return fusion
