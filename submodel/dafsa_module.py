import torch.nn as nn

from submodel.fsa_model import FrameSemanticAnalysis
from submodel.topk_select import SelectTopK
from submodel.fusion import IntraInterFusionModal


class FrameSemanticAnalysisTopK(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8, concat=True, negative_slope=0.2, node_attention=True,
                 edge_attention=True, dropout=0.5, bias=True, k_percent=0.5, norm='graph'):
        super().__init__()

        self.bn = nn.BatchNorm1d(output_dim)
        self.k_percent = k_percent

        self.gat = FrameSemanticAnalysis(
            in_channels=input_dim,
            out_channels=output_dim,
            lexical_unit_attention=node_attention,
            frame_attention=edge_attention,
            heads=num_heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            bias=bias
        )
        self.topk = SelectTopK(in_channels=output_dim)

    def forward(self, fs, multi=False):
        x, edge_index, edge_attr, num_edges = (
            fs.x, fs.edge_index, fs.edge_attr, fs.num_edges.sum())
        x, edge_attr, alpha_n2e, alpha_e2n = self.gat(
            x=self.bn(x),
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_edges=num_edges
        )

        if 0 < self.k_percent < 1:
            topk_index, _, batch_ei = self.topk(score=alpha_n2e, batch=fs.batch_ei)
            fs.x = x
            fs.edge_index = fs.edge_index[:, topk_index]
            fs.edge_attr = fs.edge_attr
            fs.batch_ei = fs.batch_ei[topk_index]

        return fs


class SemanticFrameProcessingUnit(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8, concat=True,
                 negative_slope=0.2, node_attention=True, edge_attention=True,
                 dropout=0.5, bias=True, k_percent=0.8, residual=True):
        super().__init__()

        self.residual = residual
        self.fusion = IntraInterFusionModal(input_dim=input_dim, output_dim=output_dim)

        self.model_intra = FrameSemanticAnalysisTopK(
            input_dim=input_dim,
            output_dim=output_dim,
            node_attention=node_attention,
            edge_attention=edge_attention,
            num_heads=num_heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            bias=bias,
            k_percent=k_percent)
        self.model_inter = FrameSemanticAnalysisTopK(
            input_dim=input_dim,
            output_dim=output_dim,
            node_attention=node_attention,
            edge_attention=edge_attention,
            num_heads=num_heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            bias=bias,
            k_percent=k_percent)

    def forward(self, fs_intra, fs_inter):
        fs_intra = self.model_intra(fs_intra, multi=False)
        fs_inter = self.model_inter(fs_inter, multi=True)

        fusion = self.fusion(fs_intra.x, fs_inter.x)

        if self.residual:
            fs_intra.x = fusion + fs_intra.x
            fs_inter.x = fusion + fs_inter.x
        else:

            fs_intra.x = fusion
            fs_inter.x = fusion
        return fs_intra, fs_inter


class DynamicAwareFrameSemanticAnalysisNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, num_layer, num_heads=8, concat=True, negative_slope=0.2,
                 node_attention=True, edge_attention=True, dropout=0.5, bias=True, activate='relu', k_percent=0.8, residual=True):
        super().__init__()

        self.act_mode = activate
        self.residual = residual
        if self.act_mode is not None:
            assert activate in ['relu', 'gelu', 'leaky_relu', 'elu']
            if activate == 'relu':
                self.activate = nn.ReLU()
            elif activate == 'gelu':
                self.activate = nn.GELU()
            elif activate == 'leaky_relu':
                self.activate = nn.LeakyReLU()
            else:
                self.activate = nn.ELU()

        self.hyper_gat = nn.ModuleList([])
        for i in range(num_layer):
            if i == 0:
                k = k_percent
            else:
                k = 1
            self.hyper_gat.append(
                SemanticFrameProcessingUnit(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    num_heads=num_heads,
                    concat=concat,
                    negative_slope=negative_slope,
                    node_attention=node_attention,
                    edge_attention=edge_attention,
                    dropout=dropout,
                    bias=bias,
                    k_percent=k
                )
            )

    def forward(self, fs_intra, fs_inter):
        for module in self.hyper_gat:

            fs_intra, fs_inter = module(fs_intra, fs_inter)

            if self.act_mode is not None:
                fs_intra.x = self.activate(fs_intra.x)
                fs_inter.x = self.activate(fs_inter.x)
        return fs_intra, fs_inter

