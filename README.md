# NNDR
Code for paper 「Addressing Over-Squashing in GNNs with Graph Rewiring and Ordered Neurons」

## QM9 regression

To train a model that predicts the QM9 dipole moment (target index 0), run:

```
python run_qm9_regression.py --max_epochs 200 --batch_size 64
```

By default the script builds radius-graph neighbourhoods (2 Å cutoff, 32 neighbours),
normalises the target values, and evaluates mean absolute error in the original units.
Command-line flags allow adjusting hyperparameters such as the dataset split sizes
(`--train_size`, `--val_size`, `--test_size`), radius graph configuration
(`--qm9_radius`, `--qm9_max_neighbors`), number of training epochs and trials, or
the random `--seed`.
