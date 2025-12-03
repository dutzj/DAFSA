import torch
import torch.nn as nn

from torch_geometric.data import Batch

from transformers import BertModel, BertTokenizer

from submodel.unimodal_encoder import UnimodalEncoder
from submodel.utils import batch_to_batch
from submodel.dafsa_module import DynamicAwareFrameSemanticAnalysisNetwork
from submodel.fsi_model import frame_structure_initialization_module
from submodel.fusion import SemanticFusionModule


class BertTextEncoder(nn.Module):
    def __init__(self, model_dir=None):
        super().__init__()

        tokenizer_class = BertTokenizer
        model_class = BertModel
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_model_name_or_path=model_dir)
        self.model = model_class.from_pretrained(pretrained_model_name_or_path=model_dir)

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, text):
        input_ids = text[0, :].long().unsqueeze(dim=0)
        input_mask = text[1, :].float().unsqueeze(dim=0)
        segment_ids = text[2, :].long().unsqueeze(dim=0)

        with torch.no_grad():
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]
        return last_hidden_states.squeeze(dim=0)


class PredictModel(nn.Module):
    def __init__(self, input_dim, output_dim=1, dropout=0.2, activate='relu'):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=input_dim // 4)
        self.fc2 = nn.Linear(in_features=input_dim // 4, out_features=output_dim)
        self.dropout = nn.Dropout(dropout)

        assert activate in ['relu', 'gelu', 'leaky_relu', 'elu', 'tanh']
        if activate == 'relu':
            self.activate = nn.ReLU()
        elif activate == 'gelu':
            self.activate = nn.GELU()
        elif activate == 'leaky_relu':
            self.activate = nn.LeakyReLU()
        elif activate == 'elu':
            self.activate = nn.ELU()
        elif activate == 'tanh':
            self.activate = nn.Tanh()
        else:
            ValueError('gnn_activate must be relu, gelu, leaky_relu, elu')

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.activate(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DAFSAModel(nn.Module):
    def __init__(self, dim_t, dim_a, dim_v, dim_m, num_layer, num_heads=8, concat=True,
                 negative_slope=0.2, node_attention=True, edge_attention=True, bias=True,
                 gnn_activate='relu', pred_activate='relu',
                 lstm_dropout=0.5, gnn_dropout=0.5, pred_dropout=0.3, k_percent=0.8, residual=True):
        super().__init__()

        self.bert = BertTextEncoder(
            model_dir=r'E:\My_Learning\Code\PaperCode\paper1\SA_HyGraph\bert_cache\bert-base-chinese'
        )

        self.semantic_fusion = SemanticFusionModule(dim_m, attention=False, num_heads=1, concat=False)



        self.unimodal_encoder = UnimodalEncoder(
            dim_t=dim_t,
            dim_a=dim_a,
            dim_v=dim_v,
            dim_m=dim_m,
            lstm_dropout=lstm_dropout,
        )

        self.dafsa = DynamicAwareFrameSemanticAnalysisNetwork(
            input_dim=dim_m,
            output_dim=dim_m,
            num_layer=num_layer,
            num_heads=num_heads,
            concat=concat,
            negative_slope=negative_slope,
            node_attention=node_attention,
            edge_attention=edge_attention,
            dropout=gnn_dropout,
            bias=bias,
            activate=gnn_activate,
            k_percent=k_percent,
            residual=residual
        )

        self.predict_model = PredictModel(input_dim=dim_m * 4, output_dim=1,
                                          dropout=pred_dropout, activate=pred_activate)

    def forward(self, u_t, u_a, u_v):
        # t = [self.bert(ti) for ti in t]
        u_t, u_a, u_v = self.unimodal_encoder(u_t, u_a, u_v)

        fs_intra, fs_inter = frame_structure_initialization_module(
            batch_data=(u_t, u_a, u_v),
            remove_zero=True
        )

        fs_intra, fs_inter = self.dafsa(fs_intra, fs_inter)

        fs_intra, fs_inter = self.multi_batch2batch(fs_intra, fs_inter)
        batch_fusion = []
        for intra, inter in zip(
                Batch.to_data_list(fs_intra),
                Batch.to_data_list(fs_inter)
        ):
            fusion = self.semantic_fusion(intra, inter)
            batch_fusion.append(fusion)

        out = self.predict_model( torch.stack(batch_fusion, dim=0))
        return out

    def multi_batch2batch(self, fs_intra, fs_inter):
        fs_intra = batch_to_batch(fs_intra)
        fs_inter = batch_to_batch(fs_inter)
        return fs_intra, fs_inter


if __name__ == '__main__':

    from submodel.fsi_model import generate_test_data

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model = DAFSAModel(dim_t=768, dim_a=74, dim_v=35, dim_m=512, num_layer=3,
                  num_heads=4, k_percent=0.6, concat=True, residual=False).to(device)

    test_t, text_a, text_v = generate_test_data(2, 768, 74, 35, device=torch.device('cuda'))

    print(model(test_t, text_a, text_v))