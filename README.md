# Addressing Over-Squashing in GNNs with Graph Rewiring and Ordered Neurons

This repository contains the official source code for the paper "Addressing Over-Squashing in GNNs with Graph Rewiring and Ordered Neurons".

It provides implementations of several graph rewiring techniques and a novel GNN architecture using Ordered Neurons to address the over-squashing problem in Graph Neural Networks.

## Repository Structure

The repository is organized as follows:

- `run_graph_classification.py`: The main script to launch experiments.
- `experiments/`: Contains the `Experiment` class that manages the training and evaluation workflow.
- `models/`: Contains the GNN models and layer implementations, including our proposed O-GCN and O-GIN layers.
- `preprocessing/`: Contains implementations of various graph rewiring algorithms (DIGL, FOSR, SDRF, NNPR).
- `hyperparams.py`: Defines the command-line arguments for configuring experiments.
- `requirements.txt`: A list of all the necessary Python packages to run the code.
- `results/`: Directory where the output CSV and log files are saved (will be created on first run).
- `data/`: Directory where datasets are downloaded (will be created on first run).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo-url/NNDR.git
    cd NNDR
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment (e.g., conda or venv).
    ```bash
    pip install -r requirements.txt
    ```
    Depending on your system and CUDA version, you might need to install PyTorch and PyTorch Geometric separately by following the official instructions on their websites.

## Running Experiments

The main script to run experiments is `run_graph_classification.py`. You can specify the dataset, rewiring method, GNN model, and other hyperparameters via command-line arguments.

### Example Usage

To run an experiment with the `fosr` rewiring method on the `MUTAG` dataset, you can use the following command:

```bash
python run_graph_classification.py --dataset mutag --rewiring fosr --layer_type GCN
```

To run with our proposed Ordered-Neuron GCN (O-GCN) without any rewiring:

```bash
python run_graph_classification.py --dataset mutag --rewiring None --layer_type O-GCN
```

### Available Arguments

-   `--dataset`: Name of the dataset to use (e.g., `mutag`, `enzymes`, `proteins`, `collab`, `imdb`, `reddit`).
-   `--rewiring`: Type of rewiring to be performed (e.g., `fosr`, `sdrf`, `digl`, `nnpr`, `None`).
-   `--layer_type`: Type of GNN layer to use (e.g., `GCN`, `GIN`, `O-GCN`, `O-GIN`, `R-GCN`).
-   `--num_iterations`: Number of iterations for the rewiring algorithm.
-   `--num_layers`: Number of GNN layers.
-   `--hidden_dim`: Hidden dimension of the GNN layers.
-   `--learning_rate`: Learning rate for the optimizer.
-   `--num_trials`: Number of times to run the experiment with different random seeds.
-   `--logfile`: Name of the CSV file to save the results.

For a full list of arguments, please refer to `hyperparams.py`.

## Citation

If you find this code useful for your research, please consider citing our paper:

```
@article{your_citation_key,
  title={Addressing Over-Squashing in GNNs with Graph Rewiring and Ordered Neurons},
  author={Your Name(s)},
  journal={Your Journal/Conference},
  year={Your Year}
}
```
