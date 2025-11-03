import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict

class MLPModel(nn.Module):
    """
    Simple MLP for two-radius star graph.
    Concatenates all nodes and predicts B node labels.
    """
    
    def __init__(self, args: EasyDict):
        super(MLPModel, self).__init__()
        
        self.n = args.n
        self.K = getattr(args, 'K', 1)
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        self.hidden_dim = getattr(args, 'mlp_hidden_dim', 1024)
        self.depth = args.depth
        self.dropout = getattr(args, 'dropout', 0.3)
        
        # Total nodes: A nodes + K central + B nodes
        self.total_nodes = 2 * self.n + self.K
        self.flat_dim = self.total_nodes * self.in_dim
        
        # Build simple MLP
        layers = []
        in_features = self.flat_dim
        
        for i in range(self.depth - 1):
            layers.append(nn.Linear(in_features, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            in_features = self.hidden_dim
        
        # CRITICAL FIX: Output n predictions (one for each B node)
        # Instead of outputting out_dim, output n * out_dim
        layers.append(nn.Linear(in_features, self.n * self.out_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Add embedding extraction layers for compatibility with compute_node_embedding
        self.embedding_mlp = nn.Sequential(*layers[:-1])  # All layers except the final one
    
    def compute_node_embedding(self, batch):
        """
        Compute node embeddings for compatibility with energy computation.
        Returns embeddings of shape [num_nodes, hidden_dim].
        """
        x = batch.x
        batch_idx = batch.batch
        batch_size = batch_idx.max().item() + 1
        
        # Output tensor for embeddings
        embeddings = torch.zeros(x.shape[0], self.hidden_dim, device=x.device)
        
        # Process each graph
        for b in range(batch_size):
            # Get nodes for this graph
            mask = batch_idx == b
            node_indices = mask.nonzero(as_tuple=True)[0]
            nodes = x[mask]
            
            # Flatten all nodes
            flat = nodes.reshape(-1)
            
            # Get embeddings (before final classification layer)
            emb = self.embedding_mlp(flat)  # Shape: [hidden_dim]
            
            # For now, assign the same embedding to all nodes in the graph
            # or you could reshape it differently depending on your needs
            # Since this is for B node MAD computation, we mainly care about B nodes
            b_start = self.n + self.K
            b_indices = node_indices[b_start:b_start + self.n]
            
            # Simple approach: give each B node the same embedding
            # (you might want to make this more sophisticated)
            embeddings[b_indices] = emb.unsqueeze(0).expand(self.n, -1)
        
        return embeddings
    
    def forward(self, batch):
        x = batch.x
        batch_idx = batch.batch
        batch_size = batch_idx.max().item() + 1
        
        # Output tensor
        outputs = torch.zeros(x.shape[0], self.out_dim, device=x.device)
        
        # Process each graph
        for b in range(batch_size):
            # Get nodes for this graph
            mask = batch_idx == b
            node_indices = mask.nonzero(as_tuple=True)[0]
            nodes = x[mask]
            
            # Flatten all nodes
            flat = nodes.reshape(-1)
            
            # Get predictions for ALL B nodes at once
            pred = self.mlp(flat)  # Shape: [n * out_dim]
            
            # Reshape to get individual prediction for each B node
            pred = pred.reshape(self.n, self.out_dim)  # Shape: [n, out_dim]
            
            # Assign each prediction to its corresponding B node
            b_start = self.n + self.K
            b_indices = node_indices[b_start:b_start + self.n]
            
            # Each B node gets its own unique prediction
            outputs[b_indices] = pred  # pred shape: (n, out_dim)
        
        return outputs