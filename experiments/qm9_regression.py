import math
from typing import List, Tuple

import torch
from attrdict import AttrDict
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.graph_model import GNN


default_args = AttrDict({
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "max_epochs": 300,
    "eval_every": 1,
    "patience": 50,
    "display": True,
    "device": None,
    "batch_size": 64,
    "dropout": 0.0,
    "hidden_dim": 128,
    "hidden_layers": None,
    "num_layers": 6,
    "layer_type": "GIN",
    "input_dim": None,
    "output_dim": 1,
    "num_relations": 1,
    "target_mean": 0.0,
    "target_std": 1.0,
    "num_workers": 0,
})


class QM9RegressionExperiment:
    """Train a GNN to regress a molecular property on QM9."""

    def __init__(
        self,
        args: AttrDict,
        train_dataset,
        validation_dataset,
        test_dataset,
    ) -> None:
        self.args = default_args + args
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

        if self.args.device is None:
            self.args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.args.hidden_layers is None:
            self.args.hidden_layers = [self.args.hidden_dim] * self.args.num_layers

        if self.args.input_dim is None:
            sample = self.train_dataset[0]
            self.args.input_dim = sample.x.size(-1)

        if self.args.num_relations is None:
            self.args.num_relations = 1

        self.model = GNN(self.args).to(self.args.device)
        self.loss_fn = torch.nn.MSELoss()

    def _train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        running_loss = 0.0

        for batch in loader:
            batch = batch.to(self.args.device)
            optimizer.zero_grad()

            pred = self.model(batch)
            target = batch.y.to(self.args.device)
            if target.dim() == 1:
                target = target.unsqueeze(-1)

            loss = self.loss_fn(pred, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch.num_graphs

        return running_loss / len(loader.dataset)

    def _compute_mae(self, loader: DataLoader) -> float:
        self.model.eval()
        error = 0.0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.args.device)
                pred = self.model(batch)
                target = batch.y.to(self.args.device)
                if target.dim() == 1:
                    target = target.unsqueeze(-1)

                pred = pred * self.args.target_std + self.args.target_mean
                target = target * self.args.target_std + self.args.target_mean

                error += (pred - target).abs().sum().item()

        return error / len(loader.dataset)

    def run(self) -> Tuple[float, float, float, List[dict]]:
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
        )
        val_loader = DataLoader(
            self.validation_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=10)

        best_val_mae = math.inf
        best_test_mae = math.inf
        best_train_mae = math.inf
        epochs_no_improve = 0
        history: List[dict] = []

        for epoch in range(1, self.args.max_epochs + 1):
            train_loss = self._train_epoch(train_loader, optimizer)

            if epoch % self.args.eval_every == 0:
                train_mae = self._compute_mae(train_loader)
                val_mae = self._compute_mae(val_loader)
                test_mae = self._compute_mae(test_loader)

                scheduler.step(val_mae)

                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_test_mae = test_mae
                    best_train_mae = train_mae
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if self.args.display:
                    lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"Epoch {epoch:03d} | LR {lr:.2e} | Loss {train_loss:.6f} | "
                        f"Train MAE {train_mae:.6f} | Val MAE {val_mae:.6f} | Test MAE {test_mae:.6f}"
                    )

                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_mae": train_mae,
                        "val_mae": val_mae,
                        "test_mae": test_mae,
                    }
                )

                if epochs_no_improve >= self.args.patience:
                    if self.args.display:
                        print(
                            f"Stopping early after {self.args.patience} evaluations without improvement."
                        )
                    break

        return best_train_mae, best_val_mae, best_test_mae, history
