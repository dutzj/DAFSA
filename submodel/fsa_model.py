import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter, softmax

from typing import Optional


class FrameSemanticAnalysis(MessagePassing):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        lexical_unit_attention: bool = True,
        frame_attention: bool = True,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        if concat:
            out_channels = out_channels // heads

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lexical_unit_attention = lexical_unit_attention
        self.frame_attention = frame_attention

        if self.lexical_unit_attention or self.frame_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.lin = Linear(in_channels, heads * out_channels, bias=False, weight_initializer='glorot')
            self.att = Parameter(torch.empty(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
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
        # if self.use_attention:
        if self.lexical_unit_attention or self.frame_attention:
            glorot(self.att)
        zeros(self.bias)

    @disable_dynamic_shapes(required_args=['num_edges'])
    def forward(self, x: Tensor, edge_index: Tensor,
                edge_weight: Optional[Tensor] = None,
                edge_attr: Optional[Tensor] = None,
                num_edges: Optional[int] = None):
        r"""
        Args:
            x: Node feature matrix
            edge_index: The hyperedge indices, mapping from nodes to edges.
            edge_weight: Hyperedge weights
            edge_attr: Hyperedge feature matrix
            num_edges (int, optional) : The number of edges
        """
        num_nodes = x.size(0)

        if num_edges is None:
            num_edges = 0
            if edge_index.numel() > 0:
                num_edges = int(edge_index[1].max()) + 1

        if edge_weight is None:
            edge_weight = x.new_ones(num_edges)

        x = self.lin(x)

        alpha_node_to_edge = None
        alpha_edge_to_node = None

        if self.lexical_unit_attention or self.frame_attention:
            assert edge_attr is not None
            x = x.view(-1, self.heads, self.out_channels)
            edge_attr = self.lin(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x_i = x[edge_index[0]]
            x_j = edge_attr[edge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            if self.lexical_unit_attention:
                alpha_node_to_edge = softmax(alpha, edge_index[1], num_nodes=num_edges)
            if self.frame_attention:
                alpha_edge_to_node = softmax(alpha, edge_index[0], num_nodes=num_nodes)

        D = scatter(edge_weight[edge_index[1]], edge_index[0],
                    dim=0, dim_size=num_nodes, reduce='sum')
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter(x.new_ones(edge_index.size(1)), edge_index[1],
                    dim=0, dim_size=num_edges, reduce='sum')
        B = 1.0 / B
        B[B == float("inf")] = 0

        out = self.propagate(edge_index, x=x, norm=B, alpha=alpha_node_to_edge, need_edge_arr=True,
                             size=(num_nodes, num_edges))
        norm_edge_attr, edge_attr = out[:, :, : self.out_channels], out[:, :, self.out_channels:]

        out = self.propagate(edge_index.flip([0]), x=norm_edge_attr, norm=D, alpha=alpha_edge_to_node,
                             need_edge_arr=False, size=(num_edges, num_nodes))

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
            edge_attr = edge_attr.contiguous().view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
            edge_attr = edge_attr.mean(dim=1)
        if self.bias is not None:
            out = out + self.bias
        return out, edge_attr, alpha_node_to_edge.mean(dim=1), alpha_edge_to_node.mean(dim=1)

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor, need_edge_arr: bool) -> Tensor:

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, self.heads, self.out_channels)
        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out
            x_j = alpha.view(-1, self.heads, 1) * x_j.view(-1, self.heads, self.out_channels)

        if need_edge_arr:
            out = torch.cat((out, x_j.view(-1, self.heads, self.out_channels)), dim=-1)

        return out
