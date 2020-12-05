import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as tsp
import dgl

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

class LRP_synthetic_layer(nn.Module):
    def __init__(self,
                 lrp_length = 16,
                 lrp_e_dim = 2,
                 lrp_in_dim = 2,
                 lrp_out_dim = 128):
        super(LRP_synthetic_layer, self).__init__()

        coeffs_values_3 = lambda i, j, k: torch.randn([i, j, k])
        coeffs_values_4 = lambda i, j, k, l: torch.randn([i, j, k, l])
        self.weights = nn.Parameter(coeffs_values_3(lrp_in_dim, lrp_out_dim, lrp_length))

        self.bias = nn.Parameter(torch.zeros(1, lrp_out_dim))

        self.degnet_0, self.degnet_1 = nn.Linear(1, 2 * lrp_out_dim), nn.Linear(2 * lrp_out_dim, lrp_out_dim)

        self.lrp_length = lrp_length
        self.lrp_e_dim = lrp_e_dim
        self.lrp_in_dim = lrp_in_dim
        self.lrp_out_dim = lrp_out_dim
    
    def forward(self, graph, pooling_matrix, degs, n_to_perm_length_sp_matrix, e_to_perm_length_sp_matrix):
        nfeat = graph.ndata['h']
        efeat = graph.edata['feat']

        nfeat = tsp.mm(n_to_perm_length_sp_matrix, nfeat) + tsp.mm(e_to_perm_length_sp_matrix, efeat)

        nfeat = nfeat.transpose(0, 1).view(self.lrp_in_dim, -1, self.lrp_length).permute(1, 2, 0)

        nfeat = F.relu(torch.einsum('dab,bca->dc', nfeat, self.weights) + self.bias)
        nfeat = tsp.mm(pooling_matrix, nfeat)

        factor_degs = self.degnet_1(F.relu(self.degnet_0(degs.unsqueeze(1))))#.squeeze()

        nfeat = F.relu(torch.einsum('ab,ab->ab', nfeat, factor_degs))

        graph.ndata['h'] = nfeat
        return graph

class LRP_synthetic(nn.Module):
    def __init__(self,
                 num_tasks = 1,
                 lrp_length = 16,
                 lrp_in_dim = 2,
                 hid_dim = 128,
                 num_layers = 1,
                 bn = False,
                 mlp = False
                 ):
        super(LRP_synthetic, self).__init__()
        
        self.lrp_list = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.lrp_list.append(LRP_synthetic_layer(lrp_length = lrp_length,
                                                lrp_e_dim = lrp_in_dim,
                                                lrp_in_dim = lrp_in_dim,
                                                lrp_out_dim = hid_dim 
                                                ))
            else:
                self.lrp_list.append(LRP_synthetic_layer(lrp_length = lrp_length,
                                                lrp_e_dim = 2,
                                                lrp_in_dim = hid_dim,
                                                lrp_out_dim = hid_dim 
                                                ))

        self.final_predict = nn.Linear(hid_dim, num_tasks)

        self.bn = bn
        self.mlp = mlp

        if bn:
            self.bn_layers = nn.ModuleList([nn.BatchNorm1d(hid_dim) for i in range(num_layers)])

        if mlp:
            self.mlp_layers = nn.ModuleList([nn.Linear(hid_dim, hid_dim) for i in range(num_layers)])

    def forward(self, graph, pooling_matrix, degs, n_to_perm_length_sp_matrix, e_to_perm_length_sp_matrix):
        graph.ndata['h'] = graph.ndata['feat']
        if self.bn and self.mlp:
            for lrp_layer, mlp_layer, bn in zip(self.lrp_list, self.mlp_layers, self.bn_layers):
                graph = lrp_layer(graph, pooling_matrix, degs, n_to_perm_length_sp_matrix, e_to_perm_length_sp_matrix)
                graph.ndata['h'] = bn(graph.ndata['h'])
                graph.ndata['h'] = F.relu(mlp_layer(graph.ndata['h']))
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


        graph.ndata['h'] = self.final_predict(graph.ndata['h'])
        output = dgl.sum_nodes(graph, 'h')
        return output

if __name__ == "__main__":
    from dataset_synthetic import helper, build_batch_graph_node_to_perm_times_length
    from main_synthetic import build_perm_pooling_sp_matrix, np_sparse_to_pt_sparse

    G = dgl.DGLGraph()
    G.add_nodes(4)
    G.add_edges([0,0,1,1,2,2,2,3], [1,2,0,2,0,1,3,2])

    lrp_seq = helper(G)

    split_list = [len(ll) for ll in lrp_seq]
    pooling_matrix = build_perm_pooling_sp_matrix(split_list, "sum")

    n_to_pl, e_to_pl = build_batch_graph_node_to_perm_times_length([G], [lrp_seq])

    n_to_perm_length_sp_matrix = np_sparse_to_pt_sparse(n_to_pl)
    e_to_perm_length_sp_matrix = np_sparse_to_pt_sparse(e_to_pl)

    # print(pooling_matrix)

    model = LRP_synthetic(num_tasks = 1,
                 lrp_length = 16,
                 lrp_in_dim = 1,
                 hid_dim = 1,
                 num_layers = 1)

    G.ndata['feat'] = torch.ones((G.number_of_nodes(),1))
    G.edata['feat'] = torch.ones((G.number_of_edges(),1))

    degs = G.in_degrees(list(range(G.number_of_nodes()))).type(torch.FloatTensor)

    output = model(G, pooling_matrix, degs, n_to_perm_length_sp_matrix, e_to_perm_length_sp_matrix)

    print(output)