
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import dgl

class Deepset(nn.Module):
    def __init__(self):
        super().__init__()
        
        input_size = 2
        hidden_size = 50
        
        self.node_init = nn.Linear(input_size,hidden_size)
        self.hidden_layers = nn.ModuleList()
        
        for i in range(2):
            self.hidden_layers.append( 
                nn.Sequential(
                    nn.Linear(hidden_size*2,hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size,hidden_size), 
                    nn.ReLU(),
                    nn.Linear(hidden_size,hidden_size), 
                    nn.ReLU(),
                    nn.Linear(hidden_size,hidden_size), 
                    nn.ReLU(),
                    nn.Linear(hidden_size,hidden_size),
                    nn.BatchNorm1d(hidden_size)
                )
            )
            
    def forward(self,g):
        
        g.nodes['points'].data['hidden rep'] = self.node_init(g.nodes['points'].data['xy'])
        
        for layer_i, layer in enumerate(self.hidden_layers):
                                
            mean_of_node_rep = dgl.mean_nodes(g,'hidden rep',ntype = 'points')
            broadcasted_mean = dgl.broadcast_nodes(g,mean_of_node_rep,ntype = 'points')
            g.nodes['points'].data['global rep'] = broadcasted_mean

            input_to_layer = torch.cat([g.nodes['points'].data['global rep'],g.nodes['points'].data['hidden rep']],dim=1)
                      
            features = self.hidden_layers[layer_i](input_to_layer)
            
            g.nodes['points'].data['hidden rep'] = features