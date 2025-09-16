"""DIGL pre-processing, from https://github.com/gasteigerjo/gdc.git

This module implements the graph rewiring strategy based on Personalized
PageRank (PPR) as described in the paper "Diffusion Improves Graph Learning".
The implementation is adapted from the original source code released by the
authors.
"""
import numpy as np
import torch
from torch_geometric.data import Data


def get_adj_matrix(dataset: Data) -> np.ndarray:
    """Computes the adjacency matrix from a PyG Data object.

    Args:
        dataset (Data): A PyTorch Geometric Data object containing the graph.

    Returns:
        np.ndarray: The adjacency matrix of the graph.
    """
    num_nodes = dataset.x.shape[0]
    adj_matrix = np.zeros(shape=(num_nodes, num_nodes))
    for i, j in zip(dataset.edge_index[0], dataset.edge_index[1]):
        adj_matrix[i, j] = 1.
    return adj_matrix


def get_ppr_matrix(
        adj_matrix: np.ndarray,
        alpha: float = 0.1) -> np.ndarray:
    """Computes the Personalized PageRank (PPR) matrix.

    Args:
        adj_matrix (np.ndarray): The adjacency matrix of the graph.
        alpha (float, optional): The teleport probability. Defaults to 0.1.

    Returns:
        np.ndarray: The Personalized PageRank matrix.
    """
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1/np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * H)


def get_top_k_matrix(A: np.ndarray, k: int = 128) -> np.ndarray:
    """Keeps only the top-k entries for each column of a matrix.

    Args:
        A (np.ndarray): The input matrix.
        k (int, optional): The number of top entries to keep. Defaults to 128.

    Returns:
        np.ndarray: The sparsified matrix with only top-k entries per column.
    """
    num_nodes = A.shape[0]
    row_idx = np.arange(num_nodes)
    A[A.argsort(axis=0)[:num_nodes - k], row_idx] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1  # avoid dividing by zero
    return A/norm


def get_clipped_matrix(A: np.ndarray, eps: float = 0.01) -> np.ndarray:
    """Clips matrix entries below a certain threshold.

    Args:
        A (np.ndarray): The input matrix.
        eps (float, optional): The threshold. Defaults to 0.01.

    Returns:
        np.ndarray: The clipped and column-normalized matrix.
    """
    num_nodes = A.shape[0]
    A[A < eps] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1  # avoid dividing by zero
    return A/norm


def rewire(base: Data, alpha: float, k: int = None, eps: float = None):
    """Rewires the graph based on the PPR matrix.

    This function computes the PPR matrix and then sparsifies it by either
    keeping the top-k entries or clipping values below a threshold.

    Args:
        base (Data): The input graph in PyG Data format.
        alpha (float): The teleport probability for PPR.
        k (int, optional): The number of top entries to keep for sparsification.
                           Defaults to None.
        eps (float, optional): The threshold for clipping for sparsification.
                               Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the new edge
                                           index and the edge types.
    """
    # generate adjacency matrix from sparse representation
    adj_matrix = get_adj_matrix(base)
    # obtain exact PPR matrix
    ppr_matrix = get_ppr_matrix(adj_matrix, alpha=alpha)

    if k is not None:
            # print(f'Selecting top {k} edges per node.')
            ppr_matrix = get_top_k_matrix(ppr_matrix, k=k)
    elif eps is not None:
            # print(f'Selecting edges with weight greater than {eps}.')
            ppr_matrix = get_clipped_matrix(ppr_matrix, eps=eps)
    else:
        raise ValueError("Either k or eps must be specified for sparsification.")

    # create PyG Data object
    edges_i = []
    edges_j = []
    edge_attr = []
    edge_type = []
    for i, row in enumerate(ppr_matrix):
        for j in np.where(row > 0)[0]:
            edges_i.append(i)
            edges_j.append(j)
            if adj_matrix[i, j] == 1:
                edge_type.append(0)
            else:
                edge_type.append(1)
            edge_attr.append(ppr_matrix[i, j])
    edge_index = [edges_i, edges_j]
    edge_type = torch.LongTensor(edge_type)

    data = Data(
        x=base.x,
        edge_index=torch.LongTensor(edge_index),
        y=base.y
    )
    return data.edge_index, edge_type