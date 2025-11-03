# Short-Range Oversquashing

Research project investigating oversquashing phenomena in Message Passing Neural Networks (MPNNs), specifically focusing on short-range information bottlenecks in graph neural networks.

## Description

This project addresses the oversquashing problem in Graph Neural Networks (GNNs), where information bottlenecks prevent effective message passing between nodes. The research implements and evaluates various mechanisms to mitigate oversquashing issues, including:

- Virtual node architectures
- Multiple architectural variants (GIN, GAT, GCN, GGNN, GraphSAGE, FSW)
- Set-based models (SetTransformer, Sumformer)
- Analysis of the two-radius problem in star graphs

Built with PyTorch, PyTorch Geometric, and PyTorch Lightning, this project aims to enhance GNN performance on tasks requiring long-range dependencies without information loss.

---

## Features

- **Multiple GNN Architectures**: Support for GIN, GAT, GCN, GGNN, GraphSAGE, Transformer, FSW
- **Set-Based Models**: SetTransformer and Sumformer implementations
- **Virtual Nodes**: Single and multiple virtual node configurations
- **Synthetic Benchmarks**: Two-radius and one-radius problem generators
- **Energy Metrics**: Gradient norm and Dirichlet energy computation
- **Comprehensive Logging**: CSV logging with PyTorch Lightning

---

## Installation

### Prerequisites
- Python 3.11+
- CUDA-capable GPU (recommended)
- Conda or pip package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YakovM93/short-range-oversquashing.git
   cd short-range-oversquashing
   ```

2. **Create a virtual environment**:
   ```bash
   conda create --name oversquash python=3.11 -c conda-forge
   conda activate oversquash
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   For LRGB datasets (optional):
   ```bash
   pip install -r lrgb_requirements.txt
   ```

---

## Usage

### Basic Training

Train a GIN model on the two-radius star graph problem:

```bash
python train.py --model_type GIN --task_type two --star_variant connected --start 2 --end 10
```

### Command-Line Arguments

- `--model_type`: Model architecture (`GIN`, `GAT`, `GCN`, `GGNN`, `SAGE`, `SW`, `SetTransformer`, `Sumformer`, `MLP`)
- `--task_type`: Task type (`two` for two-radius problem, `one` for one-radius problem)
- `--star_variant`: Star graph variant (`connected` for two-radius with central node(s), `disconnected` for no central nodes)
- `--start`: Starting value for parameter n (minimum number of nodes)
- `--end`: Ending value for parameter n (exclusive)
- `--use_virtual_nodes`: Enable virtual nodes
- `--num_virtual_nodes`: Number of virtual nodes (default: 1)
- `--K`: Number of central nodes for two-radius problem (default: 1)

### Examples

**Train with virtual nodes**:
```bash
python train.py --model_type GIN --task_type two --use_virtual_nodes --num_virtual_nodes 3
```

**Train SetTransformer**:
```bash
python train.py --model_type SetTransformer --task_type two --num_heads 4 --dropout 0.1
```

**Train on disconnected graphs**:
```bash
python train.py --model_type GIN --star_variant disconnected --start 5 --end 15
```

---

## Configuration

Model configurations are defined in `configs/task_config.yaml`. Key parameters:

- **Learning rate**: Model-specific learning rates
- **Batch size**: Training batch size
- **Hidden dimensions**: Model hidden layer dimensions
- **Number of layers**: GNN depth
- **Dropout**: Dropout rate for regularization

---

## Experiments

The project supports experiments on:

1. **Two-Radius Problem**: Nodes are separated by distance 2 through central nodes
2. **One-Radius Problem**: Direct connections between node groups
3. **Disconnected Graphs**: Testing information flow without direct paths

### Energy Metrics

The code computes two energy metrics:
- **Gradient Norm**: Measures gradient flow from source to target nodes
- **Dirichlet Energy**: Measures smoothness of node representations (MAD normalized by embedding norms)

---

## Research Context

This project is part of research conducted at the Technion – Israel Institute of Technology, investigating the short-range oversquashing phenomenon in MPNNs. The work explores how information bottlenecks occur even over short distances in graph structures and proposes solutions through architectural modifications.

---



## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contact

**Yaakov Mishayev**  
Technion – Israel Institute of Technology  
Email: yakov-m@campus.technion.ac.il

**Yonatan Sverdlov**  
Technion – Israel Institute of Technology  
Email: yonatans@campus.technion.ac.il

For questions, feedback, or collaboration opportunities, feel free to reach out!

---
