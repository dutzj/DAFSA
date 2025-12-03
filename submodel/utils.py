import torch
from collections import defaultdict
from torch_geometric.utils import scatter
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse, to_dense_adj


def multimodal_hyperedge_index_adjust(multimodal_hypergraph):
    multimodal_hypergraph.edge_index[1] = (
            multimodal_hypergraph.edge_index[1] - multimodal_hypergraph.edge_tune)
    return multimodal_hypergraph


def multimodal_hyperedge_index_restore(multimodal_hypergraph):
    multimodal_hypergraph.edge_index[1] = (
            multimodal_hypergraph.edge_index[1] + multimodal_hypergraph.edge_tune)
    return multimodal_hypergraph


def unique_edge(edge_index, edge_attr):
    device = edge_index.device
    index_dict = defaultdict(list)
    for hye in torch.unique(edge_index[1]):
        index_hye = torch.nonzero(edge_index[1] == hye)
        hyn = edge_index[0][index_hye.squeeze()].cpu().numpy().reshape(-1)
        key = ','.join(map(str, sorted(hyn, reverse=False)))
        index_dict[key].append(hye)

    pooled_edge_index = []
    pooled_edge_attr = []
    for i, key in enumerate(index_dict.keys()):
        node_idx = torch.tensor(list(map(int, key.split(','))), device=device)
        edge_idx = torch.stack([node_idx, torch.ones(node_idx.size(-1), device=device) * i], dim=0)
        pooled_edge_index.append(edge_idx)
        pooled_edge_attr.append(edge_attr[index_dict[key], :].mean(dim=0))

    return torch.cat(pooled_edge_index, dim=-1).to(torch.long), torch.stack(pooled_edge_attr, dim=0).to(torch.float32)


def unique_edge_(edge_index, edge_attr):
    device = edge_index.device
    dense_adj = to_dense_adj(edge_index).squeeze()

    if (dense_adj.sum(dim=0) == 0).sum() > 0:
        mask = dense_adj.sum(dim=0) != 0
        dense_adj = dense_adj[:, mask]
        edge_attr = edge_attr[mask]

    unique_edge, inverse_indices = torch.unique(dense_adj, dim=1, return_inverse=True)
    unique_inverse_indices = torch.unique(inverse_indices, sorted=True)
    unique_dense_adj = torch.zeros_like(dense_adj, device=device)
    unique_dense_adj[:, unique_inverse_indices] = unique_edge
    unique_coo_adj, _ = dense_to_sparse(unique_dense_adj)

    unique_edge_attr = scatter(edge_attr, inverse_indices, dim=0, reduce='mean')
    return unique_coo_adj, unique_edge_attr


def batch_to_batch(batch_hypergraph, need_unique=True):
    new_batch_hyg = []
    node_num, edge_num = 0, 0
    for bi in torch.unique(batch_hypergraph.batch):
        x = batch_hypergraph.x[batch_hypergraph.batch == bi]
        edge_index = batch_hypergraph.edge_index[:, batch_hypergraph.batch_ei == bi]
        edge_index = edge_index - torch.tensor([[node_num], [edge_num]], dtype=torch.long, device=x.device)
        node_num += x.size(0)
        edge_num += batch_hypergraph.num_edges[bi]
        edge_attr = batch_hypergraph.edge_attr[batch_hypergraph.batch_e == bi, :]
        if need_unique:
            try:
                edge_index, edge_attr = unique_edge_(edge_index, edge_attr)
            except:
                edge_index, edge_attr = edge_index, edge_attr
        new_batch_hyg.append(Data(x=x, edge_index=edge_index, edge_fea=edge_attr, num_edges=edge_attr.size(0)))
    return Batch.from_data_list(new_batch_hyg)