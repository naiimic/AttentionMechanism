
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import dgl
from torch.distributions.multivariate_normal import MultivariateNormal

from deepset import Deepset

class SlotAttention(nn.Module):
    def __init__(self):
        super().__init__()
        
        # same as before
        hidden_size = 50
        
        # point_size corresponds to hidden_rep + x + y
        point_size = 50 + 2
        # objects_size corresponds to hidden_rep + global_rep
        objects_size = 50 + 50

        key_size = 10
        query_size = 10
        
        # the values size have to be reflect the hidden size
        values_size = hidden_size
        
        self.dk = 64

        self.key = nn.Linear(point_size, key_size)
        self.query = nn.Linear(objects_size, query_size)
        self.values = nn.Linear(point_size, values_size)

        self.gru = nn.GRUCell(hidden_size, values_size)

        self.layer_norm = nn.LayerNorm(values_size)
        
        self.norm = 1/torch.sqrt(torch.FloatTensor([key_size]))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, self.dk),
            nn.ReLU(),
            nn.Linear(self.dk, hidden_size)
        )

    def edge_function(self, edges):

        attention = (edges.src['key'] * edges.dst['query']).sum(-1,keepdim = True) * self.norm
        values = edges.src['values']

        # this line is not required so we can visualize the data afterwards
        edges.data['attention weights'] = attention

        return {'attention' : attention, 'values' : values}

    def node_update(self,nodes):

        attention_weights = torch.softmax(nodes.mailbox['attention'], dim = 1)
        weighted_sum = torch.sum(attention_weights * nodes.mailbox['values'], dim = 1)
        
        #print(weighted_sum.shape)

        # update the hidden rep based on the existing rep and 
        new_hidden_rep = nodes.data['hidden rep'] + self.mlp(self.layer_norm(self.gru(weighted_sum,nodes.data['hidden rep'])))

        return {'hidden rep': new_hidden_rep }

    def forward(self, g):
        
        self.norm = self.norm.to(g.device)

        point_inputs = torch.cat([g.nodes['points'].data['hidden rep'],g.nodes['points'].data['xy']],dim=1)
        objects_input = torch.cat([g.nodes['predicted objects'].data['hidden rep'],g.nodes['predicted objects'].data['global rep']],dim=1)

        g.nodes['points'].data['key'] = self.key(point_inputs)
        g.nodes['points'].data['values'] = self.values(point_inputs)
        g.nodes['predicted objects'].data['query'] = self.query(objects_input)

        g.update_all(self.edge_function,self.node_update,etype='points_to_object')
                                                               
##########
##########
##########
##########
##########

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.deepset = Deepset()
        
        hidden_size = 50                                                               
        output_size_predictor = 3
        output_properties_prediction = 4
        max_N = 4

        # represent the predicted object as a vector of size hidden_size                                                       
        self.z_init = nn.Embedding(max_N, hidden_size)

        self.slotattentions = nn.ModuleList()
        
        for i in range(3):
            self.slotattentions.append( SlotAttention() )
        
        self.properties_prediction = nn.Sequential(
            nn.Linear(hidden_size,int(hidden_size/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size/2),int(hidden_size/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size/2),int(hidden_size/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size/2),output_properties_prediction)
        )

        self.size_predictor = nn.Sequential(
            nn.Linear(hidden_size,int(hidden_size/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size/2),int(hidden_size/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size/2),int(hidden_size/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size/2),output_size_predictor)
        )
    
    def create_output_graphs(self,input_g, num_points,num_objects):

        output_g = []
                                                                
        for n_points, n_objects in zip(num_points,num_objects):
            n_points, n_objects = n_points.item(), n_objects.item()

            num_nodes_dict = {
                'points' : n_points,
                'predicted objects' : n_objects,
            }

            edge_start = torch.repeat_interleave( torch.arange(n_points,device=input_g.device),n_objects)
            edge_end = torch.arange(n_objects,device=input_g.device).repeat(n_points)

            data_dict = {
                        ('points','points_to_object','predicted objects') : (edge_start,edge_end),            
                        }

            output_g.append( dgl.heterograph(data_dict,num_nodes_dict,device=input_g.device) )

        output_g = dgl.batch(output_g)

        output_g.nodes['points'].data['hidden rep'] = input_g.nodes['points'].data['hidden rep']
        output_g.nodes['points'].data['xy'] = input_g.nodes['points'].data['xy']
        output_g.nodes['points'].data['global rep'] = input_g.nodes['points'].data['global rep']

        return output_g

    def predict_objects(self,g):
        
        num_objects = g.batch_num_nodes('predicted objects')

        global_rep = dgl.mean_nodes(g,'global rep',ntype='points')
        g.nodes['predicted objects'].data['global rep'] = dgl.broadcast_nodes(g,global_rep,ntype='predicted objects')

        # for every batch we say how many objects there are
        z = torch.cat([torch.arange(N,device=g.device) for N in num_objects])
        
        g.nodes['predicted objects'].data['hidden rep'] = self.z_init( z )

        for slotattention in self.slotattentions:
            slotattention(g)

        g.nodes['predicted objects'].data['properties'] = self.properties_prediction( g.nodes['predicted objects'].data['hidden rep'] )
        
        return g

    def forward(self, g,use_target_size=False):
        
        self.deepset(g)

        global_rep = dgl.mean_nodes(g,'global rep',ntype='points')

        size_prediction = self.size_predictor(global_rep)

        if self.training or use_target_size:
            output_g = self.predict_objects(g)
        else:
           # sampling from softmax and converting 0,1,2 to 2,3,4
            num_objects = torch.torch.multinomial(torch.softmax(size_prediction,dim=1),1).view(-1)+2
            output_g = self.create_output_graphs(g, g.batch_num_nodes('points'),num_objects)
            self.predict_objects(output_g)

        return output_g, size_prediction