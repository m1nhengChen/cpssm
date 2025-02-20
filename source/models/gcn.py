import torch
from torch import nn
import torch_geometric
from BrainGB.models import GAT, GCN, MLP
from omegaconf import DictConfig

        
class MSGCN(torch.nn.Module):
    def __init__(self, config: DictConfig):
        super(MSGCN, self).__init__()
        self.pooling = config.model.pooling
        self.num_nodes = config.model.num_nodes

        # Initialize GCN model
        gcn_model = GCN(8, self.num_nodes)

        # Pass args to BrainNN (config must have pooling)
        self.model = BrainNN(config.model, gcn_model,MLP(2 * self.num_nodes))

    def forward(self, time_series,data):
        # Forward pass through BrainNN
        return self.model(data)

import torch
from collections import defaultdict
import numpy as np 
from itertools import permutations
from torch_geometric.utils import to_dense_adj
from torch.nn import functional as F


class BrainNN(torch.nn.Module):
    def __init__(self, args, gnn, discriminator=lambda x, y: x @ y.t()):
        super(BrainNN, self).__init__()
        self.gnn = gnn
        self.pooling = args.pooling
        self.discriminator = discriminator

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        g = self.gnn(x, edge_index, edge_attr, batch)
        log_logits = F.log_softmax(g, dim=-1)
        
        return log_logits