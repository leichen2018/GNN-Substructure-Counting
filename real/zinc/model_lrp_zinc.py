import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
import time
import dgl
import math

class LRP_PURE_layer(nn.Module):
    def __init__(self,
                 lrp_length = 16,
                 lrp_in_dim = 13,
                 lrp_out_dim = 13,
                 num_bond_type = 4
                 ):
        super(LRP_PURE_layer, self).__init__()

        coeffs_values_3 = lambda i, j, k: torch.randn([i, j, k])
        coeffs_values_4 = lambda i, j, k, l: torch.randn([i, j, k, l])
        self.weights = nn.Parameter(coeffs_values_3(lrp_out_dim, lrp_out_dim, lrp_length))
        self.bias = nn.Parameter(torch.zeros(1, lrp_out_dim))

        self.degnet_0, self.degnet_1 = nn.Linear(1, 2 * lrp_out_dim), nn.Linear(2 * lrp_out_dim, lrp_out_dim)

        self.lrp_length = lrp_length
        self.lrp_in_dim = lrp_in_dim
        self.lrp_out_dim = lrp_out_dim

        self.bond_encoder = nn.Embedding(num_bond_type, lrp_out_dim)
    
    def forward(self, graph, pooling_matrix, degs, n_to_perm_length_sp_matrix, e_to_perm_length_sp_matrix):
        nfeat = graph.ndata['h']
        efeat = self.bond_encoder(graph.edata['feat'])

        nfeat = torch_sparse.spmm(n_to_perm_length_sp_matrix[0], n_to_perm_length_sp_matrix[1], n_to_perm_length_sp_matrix[2], n_to_perm_length_sp_matrix[3], nfeat) + torch_sparse.spmm(e_to_perm_length_sp_matrix[0], e_to_perm_length_sp_matrix[1], e_to_perm_length_sp_matrix[2], e_to_perm_length_sp_matrix[3], efeat)
        nfeat = nfeat.transpose(0, 1).view(self.lrp_out_dim, -1, self.lrp_length).permute(1, 2, 0)
        nfeat = F.relu(torch.einsum('dab,bca->dc', nfeat, self.weights) + self.bias)
        nfeat = torch_sparse.spmm(pooling_matrix[0], pooling_matrix[1], pooling_matrix[2], pooling_matrix[3], nfeat)

        factor_degs = self.degnet_1(F.relu(self.degnet_0(degs.unsqueeze(1)))).squeeze()
        nfeat = torch.einsum('ab,ab->ab', nfeat, factor_degs)

        graph.ndata['h'] = nfeat
        return graph

class LRP_PURE_layer_alldegree(nn.Module):
    def __init__(self,
                 lrp_length = 16,
                 lrp_in_dim = 13,
                 lrp_out_dim = 13,
                 num_bond_type = 4
                 ):
        super(LRP_PURE_layer_alldegree, self).__init__()

        coeffs_values_3 = lambda i, j, k: torch.randn([i, j, k])
        coeffs_values_4 = lambda i, j, k, l: torch.randn([i, j, k, l])
        self.weights = nn.Parameter(coeffs_values_3(lrp_out_dim, lrp_out_dim, lrp_length))
        self.bias = nn.Parameter(torch.zeros(1, lrp_out_dim))

        self.len_seq = int(lrp_length**0.5)

        self.degnet_0, self.degnet_1 = nn.Linear(self.len_seq, 2 * lrp_out_dim), nn.Linear(2 * lrp_out_dim, lrp_out_dim)
        self.linear = nn.Linear(lrp_out_dim, lrp_out_dim)

        self.lrp_length = lrp_length
        self.lrp_in_dim = lrp_in_dim
        self.lrp_out_dim = lrp_out_dim

        self.bond_encoder = nn.Embedding(num_bond_type, lrp_out_dim)
    
    def forward(self, graph, pooling_matrix, degs, n_to_perm_length_sp_matrix, e_to_perm_length_sp_matrix):
        nfeat = graph.ndata['h']

        efeat = self.bond_encoder(graph.edata['feat'])

        nfeat = torch_sparse.spmm(n_to_perm_length_sp_matrix[0], n_to_perm_length_sp_matrix[1], n_to_perm_length_sp_matrix[2], n_to_perm_length_sp_matrix[3], nfeat) + torch_sparse.spmm(e_to_perm_length_sp_matrix[0], e_to_perm_length_sp_matrix[1], e_to_perm_length_sp_matrix[2], e_to_perm_length_sp_matrix[3], efeat)
        deg_perm_feat = torch_sparse.spmm(n_to_perm_length_sp_matrix[0], n_to_perm_length_sp_matrix[1], n_to_perm_length_sp_matrix[2], n_to_perm_length_sp_matrix[3], degs.unsqueeze(1))
        nfeat = nfeat.transpose(0, 1).view(self.lrp_out_dim, -1, self.lrp_length).permute(1, 2, 0)
        deg_perm_feat = deg_perm_feat.transpose(0, 1).view(1, -1, self.lrp_length).permute(1, 2, 0).squeeze()[:, list(range(0, self.lrp_length, self.len_seq + 1))]
        nfeat = self.linear(F.relu(torch.einsum('dab,bca->dc', nfeat, self.weights) + self.bias))
        deg_perm_feat = self.degnet_1(F.relu(self.degnet_0(deg_perm_feat)))
        nfeat = nfeat * deg_perm_feat
        nfeat = torch_sparse.spmm(pooling_matrix[0], pooling_matrix[1], pooling_matrix[2], pooling_matrix[3], nfeat)

        graph.ndata['h'] = nfeat
        return graph

class LRP_PURE(nn.Module):
    def __init__(self,
                 num_tasks = 1,
                 lrp_length = 16,
                 lrp_in_dim = 13,
                 hid_dim = 13,
                 num_layers = 4,
                 bn = False,
                 mlp = False,
                 num_atom_type = 28,
                 num_bond_type = 4,
                 alldegree = False
                 ):
        super(LRP_PURE, self).__init__()
        
        self.lrp_list = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                if not alldegree:
                    self.lrp_list.append(LRP_PURE_layer(lrp_length = lrp_length,
                                                    lrp_in_dim = lrp_in_dim,
                                                    lrp_out_dim = hid_dim,
                                                    num_bond_type = num_bond_type
                                                    ))
                else:
                    self.lrp_list.append(LRP_PURE_layer_alldegree(lrp_length = lrp_length,
                                                    lrp_in_dim = lrp_in_dim,
                                                    lrp_out_dim = hid_dim,
                                                    num_bond_type = num_bond_type
                                                    ))
            else:
                if not alldegree:
                    self.lrp_list.append(LRP_PURE_layer(lrp_length = lrp_length,
                                                    lrp_in_dim = hid_dim,
                                                    lrp_out_dim = hid_dim,
                                                    num_bond_type = num_bond_type
                                                    ))
                else:
                    self.lrp_list.append(LRP_PURE_layer_alldegree(lrp_length = lrp_length,
                                                    lrp_in_dim = hid_dim,
                                                    lrp_out_dim = hid_dim,
                                                    num_bond_type = num_bond_type
                                                    ))

        self.final_predict = nn.Linear(hid_dim, num_tasks)
    
        self.atom_encoder = nn.Embedding(num_atom_type, hid_dim)

        self.bn = bn

        if bn:
            self.bn_layers_0 = nn.ModuleList([nn.BatchNorm1d(hid_dim) for i in range(num_layers)])
            self.bn_layers_1 = nn.ModuleList([nn.BatchNorm1d(hid_dim) for i in range(num_layers)])

        self.mlp = mlp

        if mlp:
            self.mlp_layers = nn.ModuleList([nn.Linear(hid_dim, hid_dim) for i in range(num_layers)])

    def forward(self, graph, pooling_matrix, degs, n_to_perm_length_sp_matrix, e_to_perm_length_sp_matrix):

        graph.ndata['h'] = self.atom_encoder(graph.ndata['feat'])

        if self.bn and self.mlp:
            for lrp_layer, mlp_layer, bn0, bn1 in zip(self.lrp_list, self.mlp_layers, self.bn_layers_0, self.bn_layers_1):
                # residual
                h_prev = graph.ndata['h']
                graph = lrp_layer(graph, pooling_matrix, degs, n_to_perm_length_sp_matrix, e_to_perm_length_sp_matrix)
                graph.ndata['h'] = F.relu(bn0(graph.ndata['h']))
                graph.ndata['h'] = F.relu(bn1(mlp_layer(graph.ndata['h']) + h_prev))
        elif self.bn and not self.mlp:
            for lrp_layer, bn in zip(self.lrp_list, self.bn_layers):
                graph = lrp_layer(graph, pooling_matrix, degs, n_to_perm_length_sp_matrix, e_to_perm_length_sp_matrix)
                graph.ndata['h'] = bn(graph.ndata['h'])
        elif (not self.bn) and self.mlp:
            for lrp_layer, mlp_layer in zip(self.lrp_list, self.mlp_layers):
                graph = lrp_layer(graph, pooling_matrix, degs, n_to_perm_length_sp_matrix, e_to_perm_length_sp_matrix)
                graph.ndata['h'] = F.relu(mlp_layer(graph.ndata['h']))
        else:
            for lrp_layer in self.lrp_list:
                graph = lrp_layer(graph, pooling_matrix, degs, n_to_perm_length_sp_matrix, e_to_perm_length_sp_matrix)

        output = self.final_predict(dgl.mean_nodes(graph, 'h'))
        return output
