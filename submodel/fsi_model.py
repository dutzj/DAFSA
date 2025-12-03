import torch
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch


def frame_structure_initialization_module(batch_data, remove_zero=True):
    batch_t, batch_a, batch_v = batch_data
    device = batch_t[0].device
    fs_intra, fs_inter = [], []


    for i, (t, a, v) in enumerate(zip(batch_t, batch_a, batch_v)):

        if remove_zero:
            t = t[t.sum(dim=-1) != 0]
            a = a[a.sum(dim=-1) != 0]
            v = v[v.sum(dim=-1) != 0]

        num_lu = t.size(0) + a.size(0) + v.size(0)
        edge_index, node_index = torch.meshgrid(
            torch.arange(start=0, end=num_lu), torch.arange(start=0, end=num_lu),
            indexing="ij")
        edge_index, node_index = edge_index.flatten(), node_index.flatten()
        n2e_index = torch.stack([node_index, edge_index], dim=0).to(device)

        lu_type = torch.cat(
            (
                torch.ones(size=(t.size(0),), dtype=torch.long) * 0,
                torch.ones(size=(a.size(0),), dtype=torch.long) * 1,
                torch.ones(size=(v.size(0),), dtype=torch.long) * 2
            ), dim=-1)
        edge_index_mask, node_index_mask = torch.meshgrid(lu_type, lu_type, indexing="ij")
        edge_index_mask, node_index_mask = edge_index_mask.flatten(), node_index_mask.flatten()
        n2e_mask = torch.stack([node_index_mask, edge_index_mask], dim=0).to(device)

        l2f_index_intra = n2e_index[:, (n2e_mask[0] - n2e_mask[1] == 0)]
        l2f_index_inter = n2e_index[:, (n2e_mask[0] - n2e_mask[1] != 0)]

        fs_intra.append(
            Data(x=torch.cat((t, a, v), dim=0),
                 edge_index=l2f_index_intra,
                 edge_attr=torch.cat((t, a, v), dim=0),
                 num_edges=l2f_index_intra[-1].unique().size(0),
                 batch_ei=torch.zeros_like(l2f_index_intra[-1], dtype=torch.long),
                 batch_e=torch.zeros(l2f_index_intra[-1].unique().size(0), dtype=torch.long, device=device),
                 node_type=lu_type.to(device).long())
        )

        fs_inter.append(
            Data(x=torch.cat((t, a, v), dim=0),
                 edge_index=l2f_index_inter,
                 edge_attr=torch.cat((t, a, v), dim=0),
                 num_edges=l2f_index_inter[-1].unique().size(0),
                 batch_ei=torch.zeros_like(l2f_index_inter[-1], dtype=torch.long),
                 batch_e=torch.zeros(l2f_index_inter[-1].unique().size(0), dtype=torch.long, device=device),
                 node_type=lu_type.to(device).long())
        )

    fs_intra = Batch.from_data_list(fs_intra)
    fs_inter = Batch.from_data_list(fs_inter)
    return fs_intra, fs_inter


def generate_test_data(batch_size, dim_t, dim_a, dim_v, device):
    batch_t, batch_a, batch_v = [], [], []
    for i in range(batch_size):
        batch_t.append(
            torch.cat(
                (
                    torch.rand((4, dim_t), dtype=torch.float, device=device),
                    torch.zeros((2, dim_t), dtype=torch.float, device=device)),
                dim=0
                ),
            )
        batch_a.append(
            torch.cat(
                (
                    torch.rand((3, dim_a), dtype=torch.float, device=device),
                    torch.zeros((3, dim_a), dtype=torch.float, device=device)),
                dim=0
            ),
        )
        batch_v.append(
            torch.cat(
                (
                    torch.rand((2, dim_v), dtype=torch.float, device=device),
                    torch.zeros((4, dim_v), dtype=torch.float, device=device)),
                dim=0
            ),
        )
    return batch_t, batch_a, batch_v


if __name__ == '__main__':

    torch.set_printoptions(threshold=10000, linewidth=2000)
    test_data = generate_test_data(2, 256, 256, 256, device=torch.device('cuda'))

    fs_ra, fs_er = frame_structure_initialization_module(test_data, remove_zero=True)
    print(fs_ra.edge_index.shape)
    print(fs_er.edge_index.shape)