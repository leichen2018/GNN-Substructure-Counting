import torch
import numpy as np
import dgl
import os
import os.path as osp
from ogb.graphproppred import DglGraphPropPredDataset
from itertools import permutations, product
from dgl.data.utils import Subset, load_graphs
from dgl.data import GINDataset

from scipy.sparse import csr_matrix

from tqdm import tqdm

def Elist(graph):
    E_list = []
    for i in range(graph.number_of_nodes()):
        E_list.append(graph.successors(i).numpy())
    return E_list

def seq_generate_deep(E_list, start_node, depth = 1, node_per_layer = 1):
    current_seq_set = [[[], [start_node]]]
    # prev, this_depth
    current_depth = 0

    while current_depth < depth:
        new_seq_set = []
        for seq in current_seq_set:
            prev, this_depth = seq

            new_perm_set = [[]]
            for node in this_depth:
                new_new_perm_set = []

                for new_perm in new_perm_set:
                    new_node_children = list(set(E_list[node]) - set(prev) - set(this_depth) - set(new_perm))
                    all_perm = permutations(new_node_children, min(node_per_layer, len(new_node_children)))
                    
                    for p in all_perm:
                        new_new_perm_set.append(new_perm + list(p))

                new_perm_set = new_new_perm_set
            
            for p in new_perm_set:
                new_seq_set.append([prev + this_depth, p])

        current_seq_set = new_seq_set

        current_depth += 1

    seq_set = [p + q for p, q in current_seq_set]

    return seq_set

def seq_generate_easy(E_list, start_node, length = 4):
    all_perm = permutations(E_list[start_node], min(length, len(E_list[start_node])))
    return [[start_node] + list(p) for p in all_perm]

def seq_to_sp_indx(graph, one_perm, subtensor_length):
    dim_dict = {node:i for i, node in enumerate(one_perm)}

    node_to_length_indx_row = [i + i * subtensor_length for i in range(len(one_perm))]
    node_to_length_indx_col = one_perm

    product_one_perm = list(product(one_perm, one_perm))
    query_edge_id_src, query_edge_id_end = [edge[0] for edge in product_one_perm], [edge[1] for edge in product_one_perm]
    query_edge_result = graph.edge_ids(query_edge_id_src, query_edge_id_end, return_uv = True)

    edge_to_length_indx_row = [int(dim_dict[src.item()] * subtensor_length + dim_dict[end.item()]) for src, end, _ in zip(*query_edge_result)]
    edge_to_length_indx_col = [int(edge_id.item()) for edge_id in query_edge_result[2]]

    # return [np.array(node_to_length_indx_row), np.array(node_to_length_indx_col), np.array(edge_to_length_indx_row), np.array(edge_to_length_indx_col)]
    return [node_to_length_indx_row, node_to_length_indx_col, edge_to_length_indx_row, edge_to_length_indx_col]

def helper(graph, subtensor_length = 4, lrp_depth = 1, lrp_width = 3):
    num_of_nodes = graph.number_of_nodes()
    graph_Elist = Elist(graph)

    egonet_seq_graph = []

    for i in range(num_of_nodes):
        # this_node_perms = seq_generate(graph_Elist, i, 1, split_level = False)
        if lrp_depth == 1:
            this_node_perms = seq_generate_easy(graph_Elist, start_node = i, length = subtensor_length - 1)
        else:
            this_node_perms = seq_generate_deep(graph_Elist, start_node = i, depth = lrp_depth, node_per_layer = lrp_width)
        this_node_egonet_seq = []

        for perm in this_node_perms:
            this_node_egonet_seq.append(seq_to_sp_indx(graph, perm, subtensor_length))
        # this_node_egonet_seq = np.array(this_node_egonet_seq)
        egonet_seq_graph.append(this_node_egonet_seq)

    # print(egonet_seq_graph)
    # print(graph.in_degrees(list(range(graph.number_of_nodes()))))
    egonet_seq_graph = np.array(egonet_seq_graph)

    return egonet_seq_graph

class GINDataset_LRP(GINDataset):
    def __init__(self, dataset_name = "MUTAG", self_loop = False, degree_as_nlabel = False, lrp_save_path = "dataset", lrp_depth = 1, subtensor_length = 4, lrp_width = 3):
        super(GINDataset_LRP, self).__init__(dataset_name, self_loop, degree_as_nlabel)
        
        # if not embedding and feature == "full" and self.graphs[0].edata['feat'].shape[-1] < 4:
        #     print('Adding dimension of edge')
        #     for i in tqdm(range(len(self.graphs))):
        #         self.graphs[i].edata['feat'] = torch.cat((self.graphs[i].edata['feat'].type(torch.FloatTensor), torch.ones((self.graphs[i].number_of_edges(), 1))), dim = 1)
        #         self.graphs[i].ndata['feat'] = self.graphs[i].ndata['feat'].type(torch.FloatTensor)
        # elif not embedding and feature == "simple":
        #     print('Deleting additional feature')
        #     for i in tqdm(range(len(self.graphs))):
        #         self.graphs[i].edata['feat'] = torch.cat((self.graphs[i].edata['feat'][:,:2].type(torch.FloatTensor), torch.ones((self.graphs[i].number_of_edges(), 1))), dim = 1)
        #         self.graphs[i].ndata['feat'] = self.graphs[i].ndata['feat'][:,:2].type(torch.FloatTensor)

        self.subtensor_length = subtensor_length
        self.lrp_depth = lrp_depth
        self.lrp_width = lrp_width

        assert self.subtensor_length == self.lrp_depth * self.lrp_width + 1

        # self.output_dim = 1 + self.graphs[0].ndata['feat'].shape[-1] + self.graphs[0].edata['feat'].shape[-1]
        self.num_tasks = self.gclasses

        self.output_length = int(subtensor_length ** 2)
        self.lrp_save_path = lrp_save_path
        
        self.lrp_save_file = "lrp_" + dataset_name + "_dep" + str(lrp_depth) + "_wid" + str(lrp_width) + "_len" + str(subtensor_length)
        
        self.lrp_egonet_seq = np.array([])
        self.load_lrp()

        for i in tqdm(range(len(self.graphs))):
            self.graphs[i].ndata['attr'] = self.graphs[i].ndata['attr'].to(torch.float)

        # print(self.graphs[63].number_of_edges())
        # # self.graphs[63].edata['e'] = torch.ones((self.graphs[63].number_of_edges(), 1))
        # print(dgl.broadcast_edges(self.graphs[63], torch.ones((1))).size())
        # exit()

        if self.eclasses == 0:
            print("Adding dimension of edge data...")
            for i in tqdm(range(len(self.graphs))):
                self.graphs[i].edata['e'] = torch.ones((self.graphs[i].number_of_edges(), 1))

            assert self.graphs[0].edata['e'].size()[-1] == 1
            self.dim_efeats = 1
        
        # if self.graphs[0].ndata['feat'].shape[1] < 13:
        #     print('Adding dimension of data')
        #     for i in tqdm(range(len(self.graphs))):
        #         self.graphs[i].ndata['feat'] = torch.cat((self.graphs[i].ndata['feat'].type(torch.FloatTensor), torch.zeros((self.graphs[i].number_of_nodes(), 13 - self.graphs[i].ndata['feat'].shape[1]))), dim = 1)
        #         self.graphs[i].edata['feat'] = torch.cat((torch.zeros((self.graphs[i].number_of_edges(), 12 - self.graphs[i].edata['feat'].shape[1])), torch.ones((self.graphs[i].number_of_edges(), 1)), self.graphs[i].edata['feat'].type(torch.FloatTensor)), dim = 1)
    
    def load_lrp(self):
        print("Trying to load LRP!")
        lrp_save_file = osp.join(self.lrp_save_path, self.lrp_save_file + ".npz")
        
        if os.path.exists(lrp_save_file):
            print("LRP file exists!")
            read_file = np.load(lrp_save_file, allow_pickle=True)
            read_lrp = read_file['a']
            if len(read_lrp) == len(self.graphs):
                print("LRP file format correct! ", self.output_length)
                self.lrp_egonet_seq = read_lrp
            else:
                print("LRP file format WRONG!")
                self.LRP_preprocess()
        else: 
            print("LRP file does not exist!")
            self.LRP_preprocess()

    def save_lrp(self):
        print("Saving LRP!")
        np.savez(osp.join(self.lrp_save_path, self.lrp_save_file), a = self.lrp_egonet_seq)
        print("Saving LRP FINISHED!")

    def LRP_preprocess(self):
        print("Preprocessing LRP!")
        if len(self.lrp_egonet_seq) == 0:
            self.lrp_egonet_seq = []
            for g in tqdm(self.graphs):
                self.lrp_egonet_seq.append(helper(g, subtensor_length = self.subtensor_length, lrp_depth = self.lrp_depth, lrp_width = self.lrp_width))
        
            self.lrp_egonet_seq = np.array(self.lrp_egonet_seq)
            if len(self.lrp_egonet_seq) == len(self.graphs):
                print('LRP generated with correct format')
            else:
                print('LRP generated WRONG: PLEASE CHECK!')
                print(len(self.lrp_egonet_seq), len(self.graphs))
                exit()
            self.save_lrp()

    def __getitem__(self, idx):
        """Get datapoint with index"""

        if isinstance(idx, int):
            return self.graphs[idx], self.lrp_egonet_seq[idx], self.labels[idx]
        elif torch.is_tensor(idx) and idx.dtype == torch.long:
            if idx.dim() == 0:
                return self.graphs[idx], self.lrp_egonet_seq[idx], self.labels[idx]
            elif idx.dim() == 1:
                return Subset(self, idx.cpu())

        raise IndexError(
            'Only integers and long are valid '
            'indices (got {}).'.format(type(idx).__name__))

def build_batch_graph_node_to_perm_times_length(graphs, lrp_egonet):
    '''
        graphs: list of DGLGraph
        lrp_egonet: list of egonet of graph, dim: #graphs x #nodes x #perms x
                     (sparse index for length x #nodes or #edges)
    '''
    list_num_nodes_in_graphs = [g.number_of_nodes() for g in graphs]
    sum_num_nodes_before_graphs = [sum(list_num_nodes_in_graphs[:i]) for i in range(len(graphs))]
    list_num_edges_in_graphs = [g.number_of_edges() for g in graphs]
    sum_num_edges_before_graphs = [sum(list_num_edges_in_graphs[:i]) for i in range(len(graphs))]

    node_to_perm_length_indx_row = []
    node_to_perm_length_indx_col = []
    edge_to_perm_length_indx_row = []
    edge_to_perm_length_indx_col = []

    sum_row_number = 0

    for i, g_egonet in enumerate(lrp_egonet):
        for n_egonet in g_egonet:
            for perm in n_egonet:
                node_to_perm_length_indx_col.extend(np.array(perm[1]) + sum_num_nodes_before_graphs[i])
                node_to_perm_length_indx_row.extend(np.array(perm[0])+ sum_row_number)

                edge_to_perm_length_indx_col.extend(np.array(perm[3]) + sum_num_edges_before_graphs[i])
                edge_to_perm_length_indx_row.extend(np.array(perm[2]) + sum_row_number)

                sum_row_number += 16

    node_to_perm_length_size_row = sum_row_number
    node_to_perm_length_size_col = sum(list_num_nodes_in_graphs)
    edge_to_perm_length_size_row = sum_row_number
    edge_to_perm_length_size_col = sum(list_num_edges_in_graphs)

    # return node_to_perm_length_indx_row, node_to_perm_length_indx_col
    # return edge_to_perm_length_indx_row, edge_to_perm_length_indx_col
    data1 = np.ones((len(node_to_perm_length_indx_col, )))
    node_to_perm_length_sp_matrix = csr_matrix((data1, (node_to_perm_length_indx_row, node_to_perm_length_indx_col)), shape = (node_to_perm_length_size_row, node_to_perm_length_size_col))

    data2 = np.ones((len(edge_to_perm_length_indx_col, )))
    edge_to_perm_length_sp_matrix = csr_matrix((data2, (edge_to_perm_length_indx_row, edge_to_perm_length_indx_col)), shape = (edge_to_perm_length_size_row, edge_to_perm_length_size_col))

    return node_to_perm_length_sp_matrix, edge_to_perm_length_sp_matrix

def build_batch_graph_node_to_perm_times_length_index_form(graphs, lrp_egonet, subtensor_length = 4):
    '''
        graphs: list of DGLGraph
        lrp_egonet: list of egonet of graph, dim: #graphs x #nodes x #perms x
                     (sparse index for length x #nodes or #edges)
    '''
    list_num_nodes_in_graphs = [g.number_of_nodes() for g in graphs]
    sum_num_nodes_before_graphs = [sum(list_num_nodes_in_graphs[:i]) for i in range(len(graphs))]
    list_num_edges_in_graphs = [g.number_of_edges() for g in graphs]
    sum_num_edges_before_graphs = [sum(list_num_edges_in_graphs[:i]) for i in range(len(graphs))]

    node_to_perm_length_indx_row = []
    node_to_perm_length_indx_col = []
    edge_to_perm_length_indx_row = []
    edge_to_perm_length_indx_col = []

    sum_row_number = 0

    for i, g_egonet in enumerate(lrp_egonet):
        for n_egonet in g_egonet:
            for perm in n_egonet:
                node_to_perm_length_indx_col.extend(np.array(perm[1]) + sum_num_nodes_before_graphs[i])
                node_to_perm_length_indx_row.extend(np.array(perm[0])+ sum_row_number)

                edge_to_perm_length_indx_col.extend(np.array(perm[3]) + sum_num_edges_before_graphs[i])
                edge_to_perm_length_indx_row.extend(np.array(perm[2]) + sum_row_number)

                sum_row_number += int(subtensor_length ** 2)

    node_to_perm_length_size_row = sum_row_number
    node_to_perm_length_size_col = sum(list_num_nodes_in_graphs)
    edge_to_perm_length_size_row = sum_row_number
    edge_to_perm_length_size_col = sum(list_num_edges_in_graphs)

    # return node_to_perm_length_indx_row, node_to_perm_length_indx_col
    # return edge_to_perm_length_indx_row, edge_to_perm_length_indx_col
    data1 = np.ones((len(node_to_perm_length_indx_col, )))
    
    data2 = np.ones((len(edge_to_perm_length_indx_col, )))

    return np.array([node_to_perm_length_indx_row, node_to_perm_length_indx_col]), data1, node_to_perm_length_size_row, node_to_perm_length_size_col, np.array([edge_to_perm_length_indx_row, edge_to_perm_length_indx_col]), data2, edge_to_perm_length_size_row, edge_to_perm_length_size_col

def build_perm_pooling_sp_matrix_index_form(split_list, pooling = "sum"):
    dim0, dim1 = len(split_list), sum(split_list)
    col = np.arange(dim1)
    row = np.array([i for i, count in enumerate(split_list) for j in range(count)])

    if pooling == "sum":
        data = np.ones((dim1, ))
    elif pooling == "mean":
        data = np.array([1/s for s in split_list for i in range(s)])
    else:
        assert False
    
    return np.stack([row, col]), data, dim0, dim1

def collate_lrp_dgl_light(samples):
    graphs, lrp_egonets, labels = map(list, zip(*samples))
    n_to_pl, e_to_pl = build_batch_graph_node_to_perm_times_length(graphs, lrp_egonets)
    batched_graph = dgl.batch(graphs)
    return batched_graph, [len(node) for g in lrp_egonets for node in g], [n_to_pl, e_to_pl], torch.stack(labels)

def collate_lrp_dgl_light_index_form_wrapper(subtensor_length = 4):

    def collate_lrp_dgl_light_index_form(samples):
        graphs, lrp_egonets, labels = map(list, zip(*samples))
        egonet_to_perm_pl = build_batch_graph_node_to_perm_times_length_index_form(graphs, lrp_egonets, subtensor_length = subtensor_length)
        batched_graph = dgl.batch(graphs)
        split_list = [len(node) for g in lrp_egonets for node in g]
        perm_pooling_matrix = build_perm_pooling_sp_matrix_index_form(split_list, "mean")
        return batched_graph, perm_pooling_matrix, egonet_to_perm_pl, torch.LongTensor(labels)

    return collate_lrp_dgl_light_index_form

if __name__ == "__main__":
    g1 = dgl.DGLGraph()
    g1.add_nodes(4)
    g1.add_edges([0,1,2,3], [1,0,3,2])
    g2 = dgl.DGLGraph()
    g2.add_nodes(3)
    g2.add_edges([0,1,0,2], [1,0,2,0])
    print(build_batch_graph_node_to_perm_times_length([g1, g2], [helper(g1), helper(g2)]))
