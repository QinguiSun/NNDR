import torch
from torch_sparse import SparseTensor, spspmm
from torch_geometric.utils import degree


def nnpr(edge_index, iteration, device='cuda', sample=False, scale=1):
    # edge_index.to(device)
    nodes = torch.max(edge_index).item() + 1
    value_edge_index = torch.ones((edge_index.shape[1])).to(device)
    edge_type = torch.zeros((edge_index.shape[1])).to(device)
    edge_weights = torch.ones((edge_index.shape[1])).to(device)
    edge_index_distant = edge_index.clone().to(device)
    edge_index = edge_index.clone().to(device)
    value_edge_index_distant = value_edge_index.clone().to(device)
    for _ in range(iteration - 1):
        edge_index_distant, value_edge_index_distant = spspmm(
            edge_index_distant,
            value_edge_index_distant,
            edge_index,
            value_edge_index,
            nodes, nodes, nodes, coalesced=True
        )


    if sample:
        degrees: list = degree(edge_index[0]).tolist()

        # 1. 计算切分点
        ei = edge_index_distant[0]
        ei_ = torch.cat([ei[0:1], ei[:-1]])

        cutpoints = torch.nonzero(ei - ei_).squeeze().tolist()
        if isinstance(cutpoints, list):
            cutpoints = [0] + cutpoints + [ei.shape[0]]
        else:
            cutpoints = [0] + [cutpoints] + [ei.shape[0]]


        # 2. 切分spt
        adj_raw = [(edge_index_distant[:, start:end], value_edge_index_distant[start:end]) 
                    for start, end in zip(cutpoints[:-1], cutpoints[1:])]

        # 3. 对spt进行筛选
        adj_selected_raw = []
        weights_raw = []
        for index, (data, value) in enumerate(adj_raw):
            num = int(degrees[index] * scale)
            probabilities = torch.exp(1 - value) + 1e-9
            if num < data.shape[1]:
                if not num:
                    num = 1
                index = torch.multinomial(probabilities, num)
                adj_selected_raw.append(data[:, torch.sort(index).values])
                weights_raw.append(probabilities[torch.sort(index).values])
            else:
                adj_selected_raw.append(data)
                weights_raw.append(probabilities)
        # 4. 将筛选出来的spt拼接成最终的spt
        adj_selected = torch.cat(adj_selected_raw, dim=-1)
        edge_type_d = torch.ones((adj_selected.shape[1])).to(device)
        edge_weights_d = torch.cat(weights_raw, dim=-1)

        edge_type = torch.cat((edge_type, edge_type_d), dim=0)
        edge_index = torch.cat((edge_index, adj_selected), dim=1)
        edge_weights = torch.cat((edge_weights, edge_weights_d), dim=0)
    else:
        edge_index = torch.cat((edge_index, edge_index_distant), dim=-1)
        edge_type_d = torch.ones_like(value_edge_index_distant)
        edge_type = torch.cat((edge_type, edge_type_d), dim=0)
        edge_weights_d = torch.tensor([1.], device=device) / value_edge_index_distant
        edge_weights = torch.cat((edge_weights, edge_weights_d))


    return edge_index, edge_type, edge_weights


def innpr(edge_index, depth: int, iteration: int):
    for i in range(iteration):
        edge_index, edge_type, edge_weights = nnpr(edge_index=edge_index, iteration=depth, sample=True, scale=1)
    return edge_index, edge_type, edge_weights