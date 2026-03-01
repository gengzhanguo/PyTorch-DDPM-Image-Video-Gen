import torch
import torch.nn as nn
import torch.nn.functional as F
import math




# -------------------- Time Embedding --------------------
class TimeEmbedding(nn.Module):
    """
    Maps the Time Embedding to the Required output Dimension.
    """
    def __init__(self, 
                 n_out:int, # Output Dimension
                 t_emb_dim:int = 128 # Time Embedding Dimension
                ):
        super(TimeEmbedding, self).__init__()
        
        # Time Embedding Block
        self.te_block = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(t_emb_dim, n_out)
        )
        
    def forward(self, x):
        return self.te_block(x)