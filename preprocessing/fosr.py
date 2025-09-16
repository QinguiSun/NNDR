"""FOSR (Fiedler-based Spectral Rewiring).

This module implements the FOSR graph rewiring algorithm, which aims to
improve the spectral properties of the graph (specifically, the Fiedler
eigenvalue) by adding edges. The functions are JIT-compiled with numba for
performance.
"""
from torch_geometric.utils import to_networkx
from numba import jit, int64
import networkx as nx
import numpy as np
from math import inf


@jit(nopython=True)
def choose_edge_to_add(x, edge_index, degrees):
	"""Chooses an edge (u, v) to add which minimizes y[u]*y[v].

	This is a helper function for the FOSR algorithm.
	"""
	n = x.size
	m = edge_index.shape[1]
	y = x / ((degrees + 1) ** 0.5)
	products = np.outer(y, y)
	for i in range(m):
		u = edge_index[0, i]
		v = edge_index[1, i]
		products[u, v] = inf
	for i in range(n):
		products[i, i] = inf
	smallest_product = np.argmin(products)
	return (smallest_product % n, smallest_product // n)


@jit(nopython=True)
def compute_degrees(edge_index, num_nodes=None):
	"""Computes the degrees of all nodes in the graph."""
	if num_nodes is None:
		num_nodes = np.max(edge_index) + 1
	degrees = np.zeros(num_nodes)
	m = edge_index.shape[1]
	for i in range(m):
		degrees[edge_index[0, i]] += 1
	return degrees


@jit(nopython=True)
def add_edge(edge_index, u, v):
	"""Adds an undirected edge (u, v) to the edge index."""
	new_edge = np.array([[u, v], [v, u]])
	return np.concatenate((edge_index, new_edge), axis=1)


@jit(nopython=True)
def adj_matrix_multiply(edge_index, x):
	"""Computes Ax, where A is the adjacency matrix of the graph."""
	n = x.size
	y = np.zeros(n)
	m = edge_index.shape[1]
	for i in range(m):
		u = edge_index[0, i]
		v = edge_index[1, i]
		y[u] += x[v]
	return y


@jit(nopython=True)
def compute_spectral_gap(edge_index, x):
	"""Computes the spectral gap of the graph."""
	m = edge_index.shape[1]
	n = np.max(edge_index) + 1
	degrees = compute_degrees(edge_index, num_nodes=n)
	y = adj_matrix_multiply(edge_index, x / (degrees ** 0.5)) / (degrees ** 0.5)
	for i in range(n):
		if x[i] > 1e-9:
			return 1 - y[i]/x[i]
	return 0.


@jit(nopython=True)
def _edge_rewire(edge_index, edge_type, x=None, num_iterations=50, initial_power_iters=50):
	"""Core implementation of the FOSR algorithm."""
	m = edge_index.shape[1]
	n = np.max(edge_index) + 1
	if x is None:
		x = 2 * np.random.random(n) - 1
	degrees = compute_degrees(edge_index, num_nodes=n)
	for i in range(initial_power_iters):
		x = x - x.dot(degrees ** 0.5) * (degrees ** 0.5)/sum(degrees)
		y = x + adj_matrix_multiply(edge_index, x / (degrees ** 0.5)) / (degrees ** 0.5)
		x = y / np.linalg.norm(y)
	for I in range(num_iterations):
		i, j = choose_edge_to_add(x, edge_index, degrees=degrees)
		edge_index = add_edge(edge_index, i, j)
		degrees[i] += 1
		degrees[j] += 1
		edge_type = np.append(edge_type, 1)
		edge_type = np.append(edge_type, 1)
		x = x - x.dot(degrees ** 0.5) * (degrees ** 0.5)/sum(degrees)
		y = x + adj_matrix_multiply(edge_index, x / (degrees ** 0.5)) / (degrees ** 0.5)
		x = y / np.linalg.norm(y)
	return edge_index, edge_type, x


def edge_rewire(edge_index, x=None, edge_type=None, num_iterations=50, initial_power_iters=5):
	"""Performs FOSR edge rewiring.

    This function adds edges to the graph to improve its spectral gap, based
    on the Fiedler vector.

    Args:
        edge_index (np.ndarray): The original edge index of the graph.
        x (np.ndarray, optional): The initial Fiedler vector approximation.
                                  If None, it's randomly initialized.
                                  Defaults to None.
        edge_type (np.ndarray, optional): The original edge types. If None,
                                          they are initialized to zeros.
                                          Defaults to None.
        num_iterations (int, optional): The number of edges to add.
                                        Defaults to 50.
        initial_power_iters (int, optional): The number of power iterations
                                             to approximate the Fiedler vector.
                                             Defaults to 5.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the new
            edge index, the new edge types, and the final Fiedler vector
            approximation.
    """
	m = edge_index.shape[1]
	n = np.max(edge_index) + 1
	if x is None:
		x = 2 * np.random.random(n) - 1
	if edge_type is None:
		edge_type = np.zeros(m, dtype=np.int64)
	return _edge_rewire(edge_index, edge_type=edge_type, x=x, num_iterations=num_iterations, initial_power_iters=initial_power_iters)