import torch
import numpy as np 
import dgl
import os
import os.path as osp
from itertools import permutations, product
from dgl.data.utils import Subset, load_graphs

from scipy.sparse import csr_matrix

from tqdm import tqdm

dataset_path = './data/'
lrp_save_path = './data/lrp_save_path/'

def Elist(graph):
    E_list = []
    for i in range(graph.number_of_nodes()):
        E_list.append(graph.successors(i).numpy())
    return E_list

def seq_generate_easy(E_list, start_node, length = 4, full_permutation = False):
    if full_permutation:
        all_perm_full = permutations(E_list[start_node])
        all_perm = [[start_node] + list(p[:length]) for p in all_perm_full]
        return all_perm
    else:
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

    return [np.array(node_to_length_indx_row), np.array(node_to_length_indx_col), np.array(edge_to_length_indx_row), np.array(edge_to_length_indx_col)]

def helper(graph, subtensor_length = 4, full_permutation = False):
    num_of_nodes = graph.number_of_nodes()
    graph_Elist = Elist(graph)

    egonet_seq_graph = []

    for i in range(num_of_nodes):
        # this_node_perms = seq_generate(graph_Elist, i, 1, split_level = False)
        this_node_perms = seq_generate_easy(graph_Elist, start_node = i, length = subtensor_length - 1, full_permutation = full_permutation)
        this_node_egonet_seq = []

        for perm in this_node_perms:
            this_node_egonet_seq.append(seq_to_sp_indx(graph, perm, subtensor_length))
        this_node_egonet_seq = np.array(this_node_egonet_seq)
        egonet_seq_graph.append(this_node_egonet_seq)

    egonet_seq_graph = np.array(egonet_seq_graph)

    return egonet_seq_graph

class DglSyntheticDataset(object):
    def __init__(self, dataset = 1, task = 'star', subtensor_length = 4, full_permutation = False):
        super(DglSyntheticDataset, self).__init__()
        self.graphs = []
        self.task = task
        self.lrp_egonet_seq = np.array([])
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None
        self.labels = None
        self.full_permutation = full_permutation

        if dataset == 1:
            self.dataset_prefix = 'dataset1'
        elif dataset == 2:
            self.dataset_prefix = 'dataset2'
        else:
            print('Undefined dataset1')

        self.subtensor_length = subtensor_length
        self.output_length = int(subtensor_length ** 2)
        self.lrp_save_path = lrp_save_path
        
        if full_permutation:
            self.lrp_save_file = 'lrp_light_len' + str(subtensor_length) + '_' + self.dataset_prefix + '_full'
        else:
            self.lrp_save_file = 'lrp_light_len' + str(subtensor_length) + '_' + self.dataset_prefix

        self.load_graphs()
        self.load_lrp()

    def load_graphs(self):
        """Load graphs and labels for the task"""
        glist, all_labels = load_graphs(dataset_path + self.dataset_prefix + '.bin')

        if self.task == 'attributed_triangle': # adding node features according to parity of their node indices 
            for g in glist:
                g.ndata['feat'] = torch.FloatTensor([[i % 2, 1 - i % 2] for i in range(g.number_of_nodes())])
                g.edata['feat'] = torch.ones(g.number_of_edges(), 2)
                self.graphs.append(g)
        else:
            self.graphs = glist

        self.labels = all_labels[self.task]

        self.variance = np.std(self.labels.numpy()) ** 2
        print("Label variance: ", self.variance)

        self.train_idx = []
        with open(dataset_path + self.dataset_prefix + '_train.txt', "r") as f:
            for line in f:
                self.train_idx.append(int(line.strip()))

        self.val_idx = []
        with open(dataset_path + self.dataset_prefix + '_val.txt', "r") as f:
            for line in f:
                self.val_idx.append(int(line.strip()))

        self.test_idx = []
        with open(dataset_path + self.dataset_prefix + '_test.txt', "r") as f:
            for line in f:
                self.test_idx.append(int(line.strip()))

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
                if self.full_permutation:
                    self.lrp_egonet_seq.append(helper(g, subtensor_length = self.subtensor_length, full_permutation = True))
                else:
                    self.lrp_egonet_seq.append(helper(g, subtensor_length = self.subtensor_length))
        
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

    def __len__(self):
        """Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        """
        return len(self.graphs)

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))

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
                node_to_perm_length_indx_col.extend(perm[1] + sum_num_nodes_before_graphs[i])
                node_to_perm_length_indx_row.extend(perm[0] + sum_row_number)

                edge_to_perm_length_indx_col.extend(perm[3] + sum_num_edges_before_graphs[i])
                edge_to_perm_length_indx_row.extend(perm[2] + sum_row_number)

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

def collate_lrp_dgl_light(samples):
    graphs, lrp_egonets, labels = map(list, zip(*samples))
    n_to_pl, e_to_pl = build_batch_graph_node_to_perm_times_length(graphs, lrp_egonets)
    batched_graph = dgl.batch(graphs)
    return batched_graph, [len(node) for g in lrp_egonets for node in g], [n_to_pl, e_to_pl], torch.stack(labels)