#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 20:44:58 2021

@author: abdel
from https://github.com/ds4dm/ecole/blob/master/examples/branching-imitation.ipynb with some modifications
"""

import torch
import torch_geometric
from torch_geometric.nn import SAGEConv

class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """
    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, idx):
        data = torch.load(self.sample_files[idx])
        return data


class GNNPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.emb_size = emb_size = 16 #uniform node feature embedding dim
        self.k = 16 #kmax pooling
        self.n_convs = 4 #number of convolutions to perform parralelly
        drop_rate = 0.35
        
        # static data
        cons_nfeats = 1 
        edge_nfeats = 1
        var_nfeats = 6
        

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )


        
        #double check
        self.convs = torch.nn.ModuleList( [ SAGEConv((cons_nfeats, var_nfeats ), emb_size) for i in range(self.n_convs) ])
        
        #self.convs = torch.nn.ModuleList( [ SAGEConv((emb_size, emb_size ), emb_size) for i in range(self.n_convs) ])
        
        self.pool = torch_geometric.nn.global_sort_pool
        
        self.final_mlp = torch.nn.Sequential( 
                                    torch.nn.Linear(self.k*emb_size*self.n_convs, 256),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(256, 1, bias=False)
                                    )


    def forward(self, batch, inv=False):
    

        graphs0 = (batch.constraint_features_s, 
                   batch.edge_index_s, 
                  batch.edge_attr_s, 
                  batch.variable_features_s, 
                  batch.variable_features_s_batch)
        
    
        graphs1 = (batch.constraint_features_t,
                   batch.edge_index_t, 
                  batch.edge_attr_t,
                  batch.variable_features_t,
                  batch.variable_features_t_batch)
        
        if inv:
            graphs0, graphs1 = graphs1, graphs0
            
        
        scores0 = self.forward_graphs(*graphs0)
        scores1 = self.forward_graphs(*graphs1)
        
        #print(torch.sigmoid(scores0-scores1).squeeze(1))
        
        return torch.sigmoid(scores0-scores1).squeeze(1)
         
        
        
       
    def forward_graphs(self, constraint_features, edge_indices, edge_features, variable_features, variable_batch):
        # First step: linear embedding layers to a common dimension (64)
  
        #constraint_features = self.cons_embedding(constraint_features)
        #edge_features = self.edge_embedding(edge_features)
        #variable_features = self.var_embedding(variable_features)
 
        # 1 half convolutions (is sufficient)
        #edge indice var to cons       
        edge_indices_cons_to_var = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        
        variable_conveds = [ conv((constraint_features, variable_features), 
                                  edge_indices_cons_to_var,
                                 size=(constraint_features.size(0), variable_features.size(0))) for conv in self.convs ]


        variable_pooleds = [ self.pool(variable_conved, variable_batch, self.k) for variable_conved in variable_conveds ]
        
        feature = torch.cat(variable_pooleds, dim=1) #B,nconv*K*emb
        score = self.final_mlp(feature)

        return score #B, F=1
    
        
    


 
