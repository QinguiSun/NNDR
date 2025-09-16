"""NNPR (K-hop Neighborhood-based PPR).

This module implements a graph rewiring strategy that adds edges between nodes
and their k-hop neighbors, weighted by a value related to the Personalized
PageRank (PPR).
"""
import torch
from torch_sparse import SparseTensor, spspmm
from torch_geometric.utils import degree


def nnpr(edge_index, iteration, device='cuda', sample=False, scale=1):
    """Augments the graph with k-hop connections.

    This function computes the k-hop neighbors for each node and adds them
    to the graph. The new edges can be sampled based on node degrees.

    Args:
        edge_index (torch.Tensor): The original edge index.
        iteration (int): The number of hops (k) to consider.
        device (str, optional): The device to run computations on.
                                Defaults to 'cuda'.
        sample (bool, optional): Whether to sample the new edges.
                                 Defaults to False.
        scale (int, optional): A scaling factor for sampling, used if
                               `sample` is True. Defaults to 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing
            the new edge index, edge types, and edge weights.
    """
    nodes = torch.max(edge_index).item() + 1
    value_edge_index = torch.ones((edge_index.shape[1])).to(device)
    edge_type = torch.zeros((edge_index.shape[1])).to(device)
    edge_weights = torch.ones((edge_index.shape[1])).to(device)
    edge_index_distant = edge_index.clone().to(device)
    original_edge_index = edge_index.clone().to(device)
    value_edge_index_distant = value_edge_index.clone().to(device)

    # Compute k-hop adjacency matrix
    for _ in range(iteration - 1):
        edge_index_distant, value_edge_index_distant = spspmm(
            edge_index_distant,
            value_edge_index_distant,
            original_edge_index,
            value_edge_index,
            nodes, nodes, nodes, coalesced=True
        )

    if sample:
        degrees: list = degree(original_edge_index[0]).tolist()

        # 1. Find cut points to separate adjacency lists for each node
        ei = edge_index_distant[0]
        ei_ = torch.cat([ei[0:1], ei[:-1]])

        cutpoints = torch.nonzero(ei - ei_).squeeze().tolist()
        if isinstance(cutpoints, list):
            cutpoints = [0] + cutpoints + [ei.shape[0]]
        else:
            cutpoints = [0] + [cutpoints] + [ei.shape[0]]

        # 2. Split the sparse adjacency tensor into a list of tensors
        adj_raw = [(edge_index_distant[:, start:end], value_edge_index_distant[start:end])
                   for start, end in zip(cutpoints[:-1], cutpoints[1:])]

        # 3. Sample from the k-hop neighbors for each node
        adj_selected_raw = []
        weights_raw = []
        for index, (data, value) in enumerate(adj_raw):
            num = int(degrees[index] * scale)
            probabilities = torch.exp(1 - value) + 1e-9
            if num < data.shape[1]:
                if not num:
                    num = 1
                idx = torch.multinomial(probabilities, num)
                adj_selected_raw.append(data[:, torch.sort(idx).values])
                weights_raw.append(probabilities[torch.sort(idx).values])
            else:
                adj_selected_raw.append(data)
                weights_raw.append(probabilities)

        # 4. Concatenate the sampled edges back into a single tensor
        adj_selected = torch.cat(adj_selected_raw, dim=-1)
        edge_type_d = torch.ones((adj_selected.shape[1])).to(device)
        edge_weights_d = torch.cat(weights_raw, dim=-1)

        edge_type = torch.cat((edge_type, edge_type_d), dim=0)
        edge_index = torch.cat((original_edge_index, adj_selected), dim=1)
        edge_weights = torch.cat((edge_weights, edge_weights_d), dim=0)
    else:
        edge_index = torch.cat((original_edge_index, edge_index_distant), dim=-1)
        edge_type_d = torch.ones_like(value_edge_index_distant)
        edge_type = torch.cat((edge_type, edge_type_d), dim=0)
        edge_weights_d = torch.tensor([1.], device=device) / value_edge_index_distant
        edge_weights = torch.cat((edge_weights, edge_weights_d))

    return edge_index, edge_type, edge_weights


def innpr(edge_index, depth: int, iteration: int):
    """Iteratively applies the NNPR rewiring.

    Args:
        edge_index (torch.Tensor): The original edge index.
        depth (int): The number of hops (k) to consider in each iteration.
        iteration (int): The number of times to apply the NNPR rewiring.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing
            the final edge index, edge types, and edge weights.
    """
    for i in range(iteration):
        edge_index, edge_type, edge_weights = nnpr(
            edge_index=edge_index, iteration=depth, sample=True, scale=1
        )
    return edge_index, edge_type, edge_weights