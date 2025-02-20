import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from .ptdec import DEC
from typing import List
from .components import InterpretableTransformerEncoder
from omegaconf import DictConfig
from ..base import BaseModel
from mamba_ssm import Mamba
from torch.cuda.amp import autocast
from .moe_module import *








class SSEPoolingEncoder(nn.Module):
    """
    Transformer encoder with Pooling mechanism.
    Input size: (batch_size, input_node_num, input_feature_size)
    Output size: (batch_size, output_node_num, input_feature_size)
    """

    def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num, mask,pooling=True, orthogonal=True, freeze_center=False, project_assignment=True, cp_masking=True ):
        super().__init__()
        self.transformer = MambaLayer(input_feature_size,hidden_size=hidden_size,cp_masking=cp_masking,cp_mask=mask)

        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size *
                          input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size,
                          input_feature_size * input_node_num),
            )
            self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size, encoder=self.encoder,
                           orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)

    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, x):
        x = self.transformer(x)
        if self.pooling:
            x, assignment = self.dec(x)
            return x, assignment
        return x, None

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)

def generate_adjacency_matrix(dim, core_rate):
    """
    Generate an adjacency matrix of size dim x dim.

    Parameters:
    - dim (int): Size of the matrix (number of nodes).
    - core_rate (float): Proportion of core nodes (value between 0 and 1).

    Returns:
    - torch.Tensor: Adjacency matrix of size dim x dim.
    """
    assert 0 <= core_rate <= 1, "core_rate must be between 0 and 1."

    # Determine the number of core nodes
    num_core_nodes = int(core_rate * dim)

    # Initialize adjacency matrix
    adjacency_matrix = torch.zeros((dim, dim), device='cuda')

    # Core nodes indices
    core_indices = list(range(num_core_nodes))

    # Set connections for core nodes (fully connected among themselves)
    for i in core_indices:
        for j in core_indices:
            adjacency_matrix[i, j] = 1.0

    # Set connections between core nodes and periphery nodes
    for i in range(num_core_nodes, dim):
        for j in core_indices:
            adjacency_matrix[i, j] = 1.0
            adjacency_matrix[j, i] = 1.0
     
    adjacency_matrix.requires_grad_(False)
    # adjacency_matrix = nn.Parameter(adjacency_matrix)
    return adjacency_matrix

class CorePeripherySSM(BaseModel):

    def __init__(self, config: DictConfig):

        super().__init__()

        self.attention_list = nn.ModuleList()
        forward_dim = config.dataset.node_sz

        self.pos_encoding = config.model.pos_encoding
        if self.pos_encoding == 'identity':
            self.node_identity = nn.Parameter(torch.zeros(
                config.dataset.node_sz, config.model.pos_embed_dim), requires_grad=True)
            forward_dim = config.dataset.node_sz + config.model.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)

        sizes = config.model.sizes
        sizes[0] = config.dataset.node_sz
        in_sizes = [config.dataset.node_sz] + sizes[:-1]
        do_pooling = config.model.pooling
        self.do_pooling = do_pooling
        if config.dataset.name=='adni':
            self.dim=148
        elif config.dataset.name=='abide':
            self.dim=200
        else:
            print("error!")
            assert "undefined dataset."
        self.cp_mask=generate_adjacency_matrix(dim=self.dim,core_rate=config.model.core_rate)
        for index, size in enumerate(sizes):
            self.attention_list.append(
                SSEPoolingEncoder(input_feature_size=forward_dim,
                                    input_node_num=in_sizes[index],
                                    hidden_size=1024,
                                    output_node_num=size,
                                    pooling=do_pooling[index],
                                    orthogonal=config.model.orthogonal,
                                    freeze_center=config.model.freeze_center,
                                    project_assignment=config.model.project_assignment,
                                    cp_masking=config.model.cp_masking,
                                    mask=self.cp_mask))

        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(8 * sizes[-1], 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

    def forward(self,
                time_seires: torch.tensor,
                node_feature: torch.tensor):

        bz, _, _, = node_feature.shape

        if self.pos_encoding == 'identity':
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        assignments = []

        for atten in self.attention_list:
            node_feature, assignment = atten(node_feature)
            assignments.append(assignment)

        node_feature = self.dim_reduction(node_feature)

        node_feature = node_feature.reshape((bz, -1))

        return self.fc(node_feature)

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.dec.get_cluster_centers()

    def loss(self, assignments):
        """
        Compute KL loss for the given assignments. Note that not all encoders contain a pooling layer.
        Inputs: assignments: [batch size, number of clusters]
        Output: KL loss
        """
        decs = list(
            filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=nn.ReLU(), dropout=0.5):
        super(MLP, self).__init__()
        layers = []
        self.drop = nn.Dropout(dropout)
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation)
            if dropout>0:
                layers.append(self.drop)
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation)
                if dropout>0:
                    layers.append(self.drop)
            layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MambaLayer(nn.Module):
    def __init__(self, dim, hidden_size,cp_mask,cp_masking=True,d_state = 16, d_conv = 4, expand = 2,num_expert=8,top_k=2):
        super().__init__()
        self.hidden=hidden_size
        self.dim = dim
        self.cp_masking=cp_masking
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        # self.mamba=InterpretableTransformerEncoder(d_model=dim, nhead=4,
        #                                                     dim_feedforward=hidden_size,
        #                                                     batch_first=True)
        
        # self.mamba2 = Mamba(
        #         d_model=dim, # Model dimension d_model
        #         d_state=d_state,  # SSM state expansion factor
        #         d_conv=d_conv,    # Local convolution width
        #         expand=expand,    # Block expansion factor
        # )
        
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3,stride=1,padding=1,bias=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3,stride=1,padding=1,bias=True)
        self.cp_mask=cp_mask
        if self.cp_masking:
            self.mlp = FMoETransformerMLP(num_expert=num_expert, n_router=1, d_model=self.dim, d_hidden=self.hidden, activation=nn.GELU(), top_k=top_k)
            # self.mlp = MLP(input_dim=self.dim, hidden_dim=self.hidden, output_dim=self.dim, num_layers=2, activation=nn.GELU(), dropout=0.0)
        else:
            self.mlp = MLP(input_dim=self.dim, hidden_dim=self.hidden, output_dim=self.dim, num_layers=2, activation=nn.GELU(), dropout=0.0)
        self.act = nn.LayerNorm(self.dim*self.dim)
        # self.fc1=nn.Linear(self.dim, self.hidden, bias=True)
        # self.act2=nn.GELU()
        # self.fc2=nn.Linear(self.hidden, self.dim, bias=True)
    
    @torch.amp.autocast(device_type='cuda', enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C= x.shape[:2]
        
        
        # x1_flat = reversed_x1.reshape(B, n_tokens)
        # x1_flat = x1_flat + pe[:n_tokens, :]
        

        x1_norm = self.norm(x)
        x1_mamba = self.mamba(x1_norm)
        # x1_norm=self.act(self.proj(x1_norm.flatten(1))).reshape(B,C,C)
        out1 = x1_mamba+x1_norm
        out2 =self.act(out1.flatten(1)).reshape(B,C,C)
        out2= self.mlp(out2)
        if self.cp_masking:
            out2=out2*self.cp_mask
        out = out1+ out2

        return out