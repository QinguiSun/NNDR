"""Graph Neural Network models and layers.

This module defines the GNN architectures used in the experiments, including
several custom GNN layers based on PyTorch Geometric. It features standard
layers like GCN and GIN, as well as relational and ordered-neuron variants.
"""
from typing import Callable, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import ModuleList, Dropout, ReLU
from torch_geometric.nn import (GCNConv, RGCNConv, SAGEConv, GATConv,
                              GatedGraphConv, GINConv, FiLMConv, global_mean_pool)
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

# IMPORTANT: The original `measure_smoothing` module containing the
# `dirichlet_normalized` function was not found in the repository.
# The following is a placeholder function that returns 0.0.
# For full functionality, this should be replaced with the original
# implementation if it can be found.
def dirichlet_normalized(x, edge_index):
    """Placeholder for the original dirichlet_normalized function.

    This function is intended to compute the normalized Dirichlet energy of the
    node embeddings, which is a measure of their smoothness over the graph.

    Args:
        x (np.ndarray): Node features or embeddings.
        edge_index (np.ndarray): The graph's edge index.

    Returns:
        float: The computed Dirichlet energy. Currently returns 0.0.
    """
    return 0.0


class WGINConv(GINConv):
    """Weighted Graph Isomorphism Network (GIN) layer.

    This layer is a modification of the standard GINConv that supports
    edge weights in its propagation step.

    Args:
        nn (Callable): A neural network that maps node features to a new
            feature space.
        eps (float, optional): The epsilon value for the GIN aggregation.
                               Defaults to 0.
        train_eps (bool, optional): If True, epsilon is a trainable parameter.
                                    Defaults to False.
    """
    def __init__(self, nn: Callable[..., Any], eps: float = 0, train_eps: bool = False, **kwargs):
        super().__init__(nn, eps, train_eps, **kwargs)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, edge_weight: Adj = None) -> Tensor:
        """Forward pass of the WGINConv layer."""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, size=size, edge_weight=edge_weight)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)


class OGCNConv(torch.nn.Module):
    """Ordered-Neuron GCN (O-GCN) layer.

    This layer implements the Ordered-Neuron mechanism on top of a GCN layer,
    designed to handle different message types (e.g., from original vs. rewired
    edges) in a structured way.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        num_relations (int, optional): Number of edge types. Defaults to 2.
        first (bool, optional): Whether this is the first layer in the GNN.
                                Defaults to False.
        last (bool, optional): Whether this is the last layer in the GNN.
                               Defaults to False.
    """
    def __init__(self, in_features, out_features, num_relations=2, first=False, last=False) -> None:
        super(OGCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        convs = []
        for i in range(self.num_relations):
            convs.append(GCNConv(in_features, out_features))
        self.convs = ModuleList(convs)
        self.self_loop_conv = nn.Linear(in_features, out_features)
        self.first = first
        self.last = last
        if first:
            self.gat_x = nn.Linear(out_features, out_features // 2)
            self.gat_l = nn.Linear(out_features, out_features // 2)
            self.gat_r = nn.Linear(out_features, out_features // 2)
        else:
            self.gat = nn.Linear(3 * out_features, out_features)

    def forward(self, x: Tensor, edge_index: Adj, edge_type: Tensor, edge_weight: OptTensor = None) -> Tensor:
        """Forward pass of the OGCNConv layer."""
        x_new_x = self.self_loop_conv(x)
        if edge_weight is not None:
            x_new_n = self.convs[0](x, edge_index[:, edge_type == 0], edge_weight=edge_weight[edge_type == 0])
            x_new_d = self.convs[1](x, edge_index[:, edge_type == 1], edge_weight=edge_weight[edge_type == 1])
        else:
            x_new_n = self.convs[0](x, edge_index[:, edge_type == 0])
            x_new_d = self.convs[1](x, edge_index[:, edge_type == 1])

        if self.first:
            x_x = self.gat_x(x_new_x)
            x_l = self.gat_l(x_new_n)
            x_r = self.gat_r(x_new_d)
            out = torch.cat((x_x + x_l, x_r), dim=1)
        elif self.last:
            return x_new_x
        else:
            gat_raw = F.softmax(self.gat(torch.cat((x_new_x, x_new_n, x_new_d), dim=1)), dim=-1)
            gat = torch.cumsum(gat_raw, dim=-1)
            out = x_new_d.flip(dims=[1]) * gat + x_new_x + x_new_n
        return out


class OGINConv(torch.nn.Module):
    """Ordered-Neuron GIN (O-GIN) layer.

    This layer implements the Ordered-Neuron mechanism on top of a GIN layer.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        num_relations (int, optional): Number of edge types. Defaults to 2.
        first (bool, optional): Whether this is the first layer in the GNN.
                                Defaults to False.
        last (bool, optional): Whether this is the last layer in the GNN.
                               Defaults to False.
    """
    def __init__(self, in_features, out_features, num_relations=2, first=False, last=False) -> None:
        super(OGINConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        convs = []
        for i in range(self.num_relations):
            convs.append(WGINConv(nn.Sequential(nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features), nn.ReLU(), nn.Linear(out_features, out_features))))
        self.convs = ModuleList(convs)
        self.self_loop_conv = nn.Linear(in_features, out_features)
        self.first = first
        self.last = last
        if self.first:
            self.gat_x = nn.Linear(out_features, out_features // 2)
            self.gat_l = nn.Linear(out_features, out_features // 2)
            self.gat_r = nn.Linear(out_features, out_features // 2)
        if not self.first and not self.last:
            self.gat = nn.Linear(3 * out_features, out_features)

    def forward(self, x: Tensor, edge_index: Adj, edge_type: Tensor, edge_weight: OptTensor = None) -> Tensor:
        """Forward pass of the OGINConv layer."""
        x_new_x = self.self_loop_conv(x)
        if edge_weight is not None:
            x_new_n = self.convs[0](x, edge_index[:, edge_type == 0], edge_weight=edge_weight[edge_type == 0])
            x_new_d = self.convs[1](x, edge_index[:, edge_type == 1], edge_weight=edge_weight[edge_type == 1])
        else:
            x_new_n = self.convs[0](x, edge_index[:, edge_type == 0])
            x_new_d = self.convs[1](x, edge_index[:, edge_type == 1])

        if self.first:
            x_x = self.gat_x(x_new_x)
            x_l = self.gat_l(x_new_n)
            x_r = self.gat_r(x_new_d)
            out = torch.cat((x_x + x_l, x_r), dim=1)
        elif self.last:
            out = x_new_x
        else:
            gat_raw = F.softmax(self.gat(torch.cat((x_new_x, x_new_n, x_new_d), dim=1)), dim=-1)
            gat = torch.cumsum(gat_raw, dim=-1)
            out = (x_new_x + x_new_n) * (1 - gat) + (x_new_x + x_new_n + x_new_d.flip(dims=[1])) * gat
        return out


class RGATConv(torch.nn.Module):
    """Relational Graph Attention (R-GAT) layer.

    This layer applies a separate GAT convolution for each edge type and
    aggregates the results.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        num_relations (int): Number of edge types.
    """
    def __init__(self, in_features, out_features, num_relations):
        super(RGATConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.self_loop_conv = torch.nn.Linear(in_features, out_features)
        convs = []
        for i in range(self.num_relations):
            convs.append(GATConv(in_features, out_features))
        self.convs = ModuleList(convs)

    def forward(self, x: Tensor, edge_index: Adj, edge_type: Tensor) -> Tensor:
        """Forward pass of the RGATConv layer."""
        x_new = self.self_loop_conv(x)
        for i, conv in enumerate(self.convs):
            rel_edge_index = edge_index[:, edge_type == i]
            x_new += conv(x, rel_edge_index)
        return x_new


class RGINConv(torch.nn.Module):
    """Relational Graph Isomorphism Network (R-GIN) layer.

    This layer applies a separate GIN convolution for each edge type and
    aggregates the results.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        num_relations (int): Number of edge types.
    """
    def __init__(self, in_features, out_features, num_relations):
        super(RGINConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.self_loop_conv = torch.nn.Linear(in_features, out_features)
        convs = []
        for i in range(self.num_relations):
            convs.append(GINConv(nn.Sequential(nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features), nn.ReLU(), nn.Linear(out_features, out_features))))
        self.convs = ModuleList(convs)

    def forward(self, x: Tensor, edge_index: Adj, edge_type: Tensor) -> Tensor:
        """Forward pass of the RGINConv layer."""
        x_new = self.self_loop_conv(x)
        for i, conv in enumerate(self.convs):
            rel_edge_index = edge_index[:, edge_type == i]
            x_new += conv(x, rel_edge_index)
        return x_new


class GNN(torch.nn.Module):
    """A generic Graph Neural Network model.

    This class provides a flexible GNN architecture that can be configured
    with different layer types and hyperparameters.

    Args:
        args (AttrDict): A dictionary-like object containing model
                         hyperparameters.
    """
    def __init__(self, args):
        super(GNN, self).__init__()
        self.args = args
        self.num_relations = args.num_relations
        self.layer_type = args.layer_type
        num_features = [args.input_dim] + list(args.hidden_layers) + [args.output_dim]
        self.num_layers = len(num_features) - 1
        layers = []
        for i, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            layers.append(self.get_layer(in_features, out_features, first=(i == 0), last=(i == len(num_features) - 2)))
        self.layers = ModuleList(layers)
        self.dropout = Dropout(p=args.dropout)
        self.act_fn = ReLU()

        if self.args.last_layer_fa:
            # add transformation associated with complete graph if last layer is fully adjacent
            if self.layer_type == "R-GCN" or self.layer_type == "GCN":
                self.last_layer_transform = torch.nn.Linear(self.args.hidden_dim, self.args.output_dim)
            elif self.layer_type == "R-GIN" or self.layer_type == "GIN":
                self.last_layer_transform = nn.Sequential(nn.Linear(self.args.hidden_dim, self.args.hidden_dim), nn.BatchNorm1d(self.args.hidden_dim), nn.ReLU(), nn.Linear(self.args.hidden_dim, self.args.output_dim))
            else:
                raise NotImplementedError

    def get_layer(self, in_features: int, out_features: int, first: bool = False, last: bool = False) -> torch.nn.Module:
        """Factory method to create a GNN layer based on the layer_type.

        Args:
            in_features (int): Number of input features for the layer.
            out_features (int): Number of output features for the layer.
            first (bool, optional): Whether this is the first layer.
                                    Defaults to False.
            last (bool, optional): Whether this is the last layer.
                                   Defaults to False.

        Raises:
            NotImplementedError: If the specified layer_type is not supported.

        Returns:
            torch.nn.Module: The created GNN layer.
        """
        if self.layer_type == "GCN":
            return GCNConv(in_features, out_features)
        elif self.layer_type == "R-GCN":
            return RGCNConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "R-GAT":
            return RGATConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "R-GIN":
            return RGINConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "GIN":
            return GINConv(nn.Sequential(nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features), nn.ReLU(), nn.Linear(out_features, out_features)))
        elif self.layer_type == "SAGE":
            return SAGEConv(in_features, out_features)
        elif self.layer_type == "FiLM":
            return FiLMConv(in_features, out_features)
        elif self.layer_type in ["O-GCN", "OW-GCN"]:
            return OGCNConv(in_features, out_features, first=first, last=last)
        elif self.layer_type in ["O-GIN", "OW-GIN"]:
            return OGINConv(in_features, out_features, first=first, last=last)
        raise NotImplementedError(f"Layer type {self.layer_type} not supported.")

    def forward(self, graph, measure_dirichlet=False):
        """Forward pass of the GNN model.

        Args:
            graph (Data): The input graph in PyG Data format.
            measure_dirichlet (bool, optional): If True, computes and returns
                the Dirichlet energy of the final embeddings instead of the
                graph-level predictions. Defaults to False.

        Returns:
            torch.Tensor: The graph-level predictions or the Dirichlet energy.
        """
        x, edge_index, ptr, batch = graph.x, graph.edge_index, graph.ptr, graph.batch
        x = x.float()
        for i, layer in enumerate(self.layers):
            if self.layer_type in ["R-GCN", "R-GAT", "R-GIN", "FiLM", "O-GCN", "O-GIN"]:
                x_new = layer(x, edge_index, edge_type=graph.edge_type)
            elif self.layer_type in ["OW-GCN", "OW-GIN"]:
                x_new = layer(x, edge_index, edge_type=graph.edge_type, edge_weight=graph.edge_weight)
            else:
                x_new = layer(x, edge_index)
            if i != self.num_layers - 1:
                x_new = self.act_fn(x_new)
                x_new = self.dropout(x_new)
            if i == self.num_layers - 1 and self.args.last_layer_fa:
                # handle final layer when making last layer FA
                combined_values = global_mean_pool(x, batch)
                combined_values = self.last_layer_transform(combined_values)
                if self.layer_type in ["R-GCN", "R-GIN"]:
                    x_new += combined_values[batch]
                else:
                    x_new = combined_values[batch]
            x = x_new
        if measure_dirichlet:
            # check dirichlet energy instead of computing final values
            energy = dirichlet_normalized(x.cpu().numpy(), graph.edge_index.cpu().numpy())
            return energy
        x = global_mean_pool(x, batch)
        return x