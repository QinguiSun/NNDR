"""Experiment runner for graph classification tasks.

This module provides the `Experiment` class, which encapsulates the entire
workflow for training and evaluating a GNN model on a graph classification
dataset. It handles data loading, model initialization, training loops,
evaluation, and model saving.
"""
import torch
import numpy as np
from attrdict import AttrDict
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from math import inf

from models.graph_model import GNN, dirichlet_normalized

default_args = AttrDict(
    {"learning_rate": 1e-3,
     "max_epochs": 1000000,
     "display": True,
     "device": None,
     "eval_every": 1,
     "stopping_criterion": "validation",
     "stopping_threshold": 1.01,
     "patience": 100,
     "train_fraction": 0.8,
     "validation_fraction": 0.1,
     "test_fraction": 0.1,
     "dropout": 0.0,
     "weight_decay": 1e-5,
     "input_dim": None,
     "hidden_dim": 32,
     "output_dim": 1,
     "hidden_layers": None,
     "num_layers": 1,
     "batch_size": 64,
     "layer_type": "R-GCN",
     "num_relations": 2,
     "last_layer_fa": False,
     "save": False
     }
)


class Experiment:
    """A class to manage a single graph classification experiment.

    This class sets up the dataset, model, and training configuration, and
    provides methods to run the training loop and evaluate the model.

    Args:
        args (AttrDict, optional): Hyperparameters for the experiment.
                                   Defaults to None.
        dataset (Dataset, optional): The full dataset. Defaults to None.
        train_dataset (Dataset, optional): The training dataset.
                                           Defaults to None.
        validation_dataset (Dataset, optional): The validation dataset.
                                                Defaults to None.
        test_dataset (Dataset, optional): The test dataset. Defaults to None.
    """
    def __init__(self, args=None, dataset=None, train_dataset=None, validation_dataset=None, test_dataset=None):
        self.args = default_args + args
        self.dataset = dataset
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.loss_fn = torch.nn.CrossEntropyLoss()

        if self.args.device is None:
            self.args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.args.hidden_layers is None:
            self.args.hidden_layers = [self.args.hidden_dim] * self.args.num_layers
        if self.args.input_dim is None:
            self.args.input_dim = self.dataset[0].x.shape[1]
        for graph in self.dataset:
            if "edge_type" not in graph.keys():
                num_edges = graph.edge_index.shape[1]
                graph.edge_type = torch.zeros(num_edges, dtype=int)
        if self.args.num_relations is None:
            if self.args.rewiring == "None":
                self.args.num_relations = 1
            else:
                self.args.num_relations = 2
        self.model = GNN(self.args).to(self.args.device)
        # randomly assign a train/validation/test split, or train/validation split if test already assigned
        if self.test_dataset is None:
            dataset_size = len(self.dataset)
            train_size = int(self.args.train_fraction * dataset_size)
            validation_size = int(self.args.validation_fraction * dataset_size)
            test_size = dataset_size - train_size - validation_size
            self.train_dataset, self.validation_dataset, self.test_dataset = random_split(
                self.dataset, [train_size, validation_size, test_size])
        elif self.validation_dataset is None and self.train_dataset is not None:
            train_size = int(self.args.train_fraction * len(self.train_dataset))
            validation_size = len(self.train_dataset) - train_size
            self.train_dataset, self.validation_dataset = random_split(
                self.train_dataset, [train_size, validation_size])

    def run(self):
        """Runs the training and evaluation loop.

        Returns:
            Tuple[float, float, float, float]: A tuple containing the final
                training accuracy, validation accuracy, test accuracy, and
                Dirichlet energy.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer)

        if self.args.display:
            print("Starting training")
        best_train_acc = 0.0
        best_validation_acc = 0.0
        best_test_acc = 0.0
        train_goal = 0.0
        validation_goal = 0.0
        epochs_no_improve = 0

        train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
        validation_loader = DataLoader(self.validation_dataset, batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=True)
        complete_loader = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True)

        for epoch in range(1, 1 + self.args.max_epochs):
            self.model.train()
            total_loss = 0
            optimizer.zero_grad()

            for graph in train_loader:
                graph = graph.to(self.args.device)
                y = graph.y.to(self.args.device)

                out = self.model(graph)
                loss = self.loss_fn(input=out, target=y)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            new_best_str = ''
            scheduler.step(total_loss)
            if epoch % self.args.eval_every == 0:
                train_acc = self.eval(loader=train_loader)
                validation_acc = self.eval(loader=validation_loader)
                test_acc = self.eval(loader=test_loader)
                if self.args.stopping_criterion == "train":
                    if train_acc > train_goal:
                        best_train_acc = train_acc
                        best_validation_acc = validation_acc
                        best_test_acc = test_acc
                        epochs_no_improve = 0
                        train_goal = train_acc * self.args.stopping_threshold
                        new_best_str = ' (new best train)'
                    elif train_acc > best_train_acc:
                        best_train_acc = train_acc
                        best_validation_acc = validation_acc
                        best_test_acc = test_acc
                    else:
                        epochs_no_improve += 1
                elif self.args.stopping_criterion == 'validation':
                    if validation_acc > validation_goal:
                        best_train_acc = train_acc
                        best_validation_acc = validation_acc
                        best_test_acc = test_acc
                        epochs_no_improve = 0
                        validation_goal = validation_acc * self.args.stopping_threshold
                        new_best_str = ' (new best validation)'
                    elif validation_acc >= best_validation_acc:
                        # Only update if the validation accuracy is better or equal
                        best_train_acc = train_acc
                        best_validation_acc = validation_acc
                        best_test_acc = test_acc
                    else:
                        epochs_no_improve += 1

                if self.args.display:
                    print(f'Epoch {epoch}, Train acc: {train_acc:.3f}, Validation acc: {validation_acc:.3f}{new_best_str}, Test acc: {test_acc:.3f}')
                if epochs_no_improve > self.args.patience:
                    if self.args.display:
                        print(f'{self.args.patience} epochs without improvement, stopping training')
                        print(f'Best train acc: {best_train_acc:.3f}, Best validation acc: {best_validation_acc:.3f}, Best test acc: {best_test_acc:.3f}')
                    energy = self.check_dirichlet(loader=complete_loader)
                    if self.args.save:
                        if self.args.display:
                            print('Saving model...')
                        self.save_model(best_test_acc)
                    return best_train_acc, best_validation_acc, best_test_acc, energy
        if self.args.display:
            print('Reached max epoch count, stopping training')
            print(f'Best train acc: {best_train_acc:.3f}, Best validation acc: {best_validation_acc:.3f}, Best test acc: {best_test_acc:.3f}')
        energy = self.check_dirichlet(loader=complete_loader)
        return best_train_acc, best_validation_acc, best_test_acc, energy

    def eval(self, loader):
        """Evaluates the model on a given data loader.

        Args:
            loader (DataLoader): The data loader for evaluation.

        Returns:
            float: The accuracy of the model on the dataset.
        """
        self.model.eval()
        sample_size = len(loader.dataset)
        total_correct = 0
        with torch.no_grad():
            for graph in loader:
                graph = graph.to(self.args.device)
                y = graph.y.to(self.args.device)
                out = self.model(graph)
                _, pred = out.max(dim=1)
                total_correct += pred.eq(y).sum().item()

        return total_correct / sample_size

    def check_dirichlet(self, loader):
        """Computes the average Dirichlet energy of the node embeddings.

        Args:
            loader (DataLoader): The data loader to compute energy for.

        Returns:
            float: The average Dirichlet energy.
        """
        self.model.eval()
        total_energy = 0
        num_graphs = 0
        with torch.no_grad():
            for graph in loader:
                graph = graph.to(self.args.device)
                total_energy += self.model(graph, measure_dirichlet=True)
                num_graphs += 1
        return total_energy / num_graphs if num_graphs > 0 else 0

    def save_model(self, acc, path='../checkpoint/'):
        """Saves the trained model to a file.

        Args:
            acc (float): The test accuracy of the model, used in the filename.
            path (str, optional): The directory to save the model in.
                                  Defaults to '../checkpoint/'.
        """
        name = f"{self.args.layer_type}_{self.args.num_layers}layer_{self.args.hidden_dim}hd_{acc:.3f}acc.pt"
        torch.save(self.model, path+name)
        if self.args.display:
            print(f"Model saved to {path+name}")

