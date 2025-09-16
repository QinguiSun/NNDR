"""Train graph neural networks to predict QM9 molecular properties."""

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from attrdict import AttrDict
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T

from experiments.qm9_regression import QM9RegressionExperiment
from hyperparams import get_args_from_input


class SetTarget(T.BaseTransform):
    """Keep only the label associated with a particular QM9 target."""

    def __init__(self, index: int) -> None:
        self.index = index

    def __call__(self, data):
        data.y = data.y[self.index : self.index + 1]
        return data


class AddEdgeType(T.BaseTransform):
    """Ensure an ``edge_type`` tensor exists for relation-aware layers."""

    def __call__(self, data):
        num_edges = data.edge_index.size(1)
        edge_type = getattr(data, "edge_type", None)
        if edge_type is None or edge_type.numel() != num_edges:
            data.edge_type = data.edge_index.new_zeros(num_edges, dtype=torch.long)
        else:
            data.edge_type = edge_type.view(-1).long()
        return data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_splits(
    dataset: QM9, train_size: int, val_size: int, test_size: int | None
) -> Tuple[QM9, QM9, QM9]:
    total = len(dataset)
    if train_size <= 0 or train_size > total:
        raise ValueError(
            f"train_size must be between 1 and {total}, received {train_size}."
        )

    train_size = min(train_size, total)
    remaining = total - train_size
    if remaining <= 0:
        raise ValueError(
            "Not enough molecules left for validation/test splits. Reduce train_size."
        )

    val_size = min(val_size, remaining)
    if val_size <= 0:
        raise ValueError("val_size must be positive after accounting for train_size.")

    remaining -= val_size
    if remaining <= 0:
        raise ValueError(
            "Not enough molecules left for the test split. Adjust the split sizes."
        )

    if test_size is None:
        test_size = remaining
    else:
        test_size = min(test_size, remaining)

    if test_size <= 0:
        raise ValueError("test_size must be positive after adjusting for other splits.")

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size : train_size + val_size]
    test_dataset = dataset[train_size + val_size : train_size + val_size + test_size]

    return train_dataset, val_dataset, test_dataset


def main() -> None:
    default_args = AttrDict(
        {
            "dataset_root": "./data/QM9",
            "prop": 0,
            "qm9_radius": 2.0,
            "qm9_max_neighbors": 32,
            "train_size": 110_000,
            "val_size": 10_000,
            "test_size": None,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "batch_size": 64,
            "max_epochs": 200,
            "patience": 20,
            "eval_every": 1,
            "dropout": 0.0,
            "hidden_dim": 128,
            "num_layers": 6,
            "layer_type": "GIN",
            "num_relations": 1,
            "output_dim": 1,
            "num_trials": 1,
            "num_workers": 0,
            "seed": 0,
            "display": True,
        }
    )

    args = default_args
    args += get_args_from_input()

    set_seed(args.seed)

    dataset_root = Path(args.dataset_root).expanduser()
    dataset_root.mkdir(parents=True, exist_ok=True)

    target_index = args.prop

    transform = T.Compose(
        [
            T.RadiusGraph(
                r=args.qm9_radius,
                loop=False,
                max_num_neighbors=args.qm9_max_neighbors,
            ),
            T.ToUndirected(),
            T.Distance(norm=False, cat=False),
            SetTarget(target_index),
            AddEdgeType(),
        ]
    )

    dataset = QM9(str(dataset_root), transform=transform)

    if dataset.data.y.dim() == 1:
        dataset.data.y = dataset.data.y.unsqueeze(-1)

    target_mean = dataset.data.y.mean(dim=0, keepdim=True)
    target_std = dataset.data.y.std(dim=0, keepdim=True)
    if torch.any(target_std == 0):
        raise ValueError("Target standard deviation is zero; cannot normalise labels.")

    dataset.data.y = (dataset.data.y - target_mean) / target_std
    target_mean_scalar = target_mean.view(-1).item()
    target_std_scalar = target_std.view(-1).item()

    train_dataset, val_dataset, test_dataset = compute_splits(
        dataset, args.train_size, args.val_size, args.test_size
    )

    args.target_mean = target_mean_scalar
    args.target_std = target_std_scalar
    args.input_dim = train_dataset[0].x.size(-1)

    print(
        "Loaded QM9 with {} molecules (train={}, val={}, test={}).".format(
            len(dataset), len(train_dataset), len(val_dataset), len(test_dataset)
        )
    )
    print(
        f"Predicting target index {target_index} with mean={target_mean_scalar:.6f}, std={target_std_scalar:.6f}."
    )

    num_trials = max(1, args.num_trials)
    train_maes: List[float] = []
    val_maes: List[float] = []
    test_maes: List[float] = []

    for trial in range(num_trials):
        if args.display:
            print(f"\n=== Trial {trial + 1}/{num_trials} ===")

        set_seed(args.seed + trial)

        experiment = QM9RegressionExperiment(
            args=args,
            train_dataset=train_dataset,
            validation_dataset=val_dataset,
            test_dataset=test_dataset,
        )
        train_mae, val_mae, test_mae, _ = experiment.run()

        train_maes.append(train_mae)
        val_maes.append(val_mae)
        test_maes.append(test_mae)

        print(
            f"Trial {trial + 1}: Train MAE={train_mae:.6f}, Val MAE={val_mae:.6f}, Test MAE={test_mae:.6f}"
        )

    def summarise(values: List[float]) -> Tuple[float, float]:
        array = np.asarray(values, dtype=np.float64)
        mean = float(array.mean())
        std = float(array.std(ddof=0))
        return mean, std

    train_mean, train_std = summarise(train_maes)
    val_mean, val_std = summarise(val_maes)
    test_mean, test_std = summarise(test_maes)

    print(
        "\nFinished {} trial(s).".format(num_trials)
        + f"\nTrain MAE: {train_mean:.6f} ± {train_std:.6f}"
        + f"\nVal MAE:   {val_mean:.6f} ± {val_std:.6f}"
        + f"\nTest MAE:  {test_mean:.6f} ± {test_std:.6f}"
    )


if __name__ == "__main__":
    main()
