import torch
import pickle
import torch.utils.data
import time
import os
import numpy as np

import csv

import dgl

from scipy import sparse as sp
import numpy as np

import os.path as osp
from dgl.data.utils import Subset, load_graphs
from dataset_lrp_gindataset import helper
from tqdm import tqdm

'''
    MoleculeDGL, MoleculeDatasetDGL, MoleculeDataset are from 
        ``https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/data/molecules.py''
'''

# Download `ZINC.pkl` following `https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/data/script_download_molecules.sh`
# Put ZINC.pkl under `./data`
# 

molecule_zip_dir = './data/'
molecule_fold_dir = './data/molecules'

class MoleculeDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, num_graphs):
        self.data_dir = data_dir
        self.split = split
        self.num_graphs = num_graphs
        
        with open(data_dir + "/%s.pickle" % self.split,"rb") as f:
            self.data = pickle.load(f)

        # loading the sampled indices from file ./zinc_molecules/<split>.index
        with open(data_dir + "/%s.index" % self.split,"r") as f:
            data_idx = [list(map(int, idx)) for idx in csv.reader(f)]
            self.data = [ self.data[i] for i in data_idx[0] ]
            
        assert len(self.data)==num_graphs, "Sample num_graphs again; available idx: train/val/test => 10k/1k/1k"
        
        """
        data is a list of Molecule dict objects with following attributes
        
          molecule = data[idx]
        ; molecule['num_atom'] : nb of atoms, an integer (N)
        ; molecule['atom_type'] : tensor of size N, each element is an atom type, an integer between 0 and num_atom_type
        ; molecule['bond_type'] : tensor of size N x N, each element is a bond type, an integer between 0 and num_bond_type
        ; molecule['logP_SA_cycle_normalized'] : the chemical property to regress, a float variable
        """
        
        self.graph_lists = []
        self.graph_labels = []
        self.n_samples = len(self.data)
        self._prepare()
    
    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))
        
        for molecule in self.data:
            node_features = molecule['atom_type'].long()
            
            adj = molecule['bond_type']
            edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list
            
            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(-1).long()
            
            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])
            g.ndata['feat'] = node_features
            
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata['feat'] = edge_features
            
            self.graph_lists.append(g)
            self.graph_labels.append(molecule['logP_SA_cycle_normalized'])
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]
    
    
class MoleculeDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name='Zinc'):
        t0 = time.time()
        self.name = name
        
        self.num_atom_type = 28 # known meta-info about the zinc dataset; can be calculated as well
        self.num_bond_type = 4 # known meta-info about the zinc dataset; can be calculated as well
        
        # data_dir='/home/lc3909/lei/ZINC/molecules'
        data_dir = molecule_fold_dir
        
        self.train = MoleculeDGL(data_dir, 'train', num_graphs=10000)
        self.val = MoleculeDGL(data_dir, 'val', num_graphs=1000)
        self.test = MoleculeDGL(data_dir, 'test', num_graphs=1000)
        print("Time taken: {:.4f}s".format(time.time()-t0))

class MoleculeDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading SBM datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        # data_dir = 'data/molecules/'
        # data_dir = '/home/lc3909/lei/ZINC/'
        data_dir = molecule_zip_dir
        with open(data_dir+name+'.pkl',"rb") as f:
            f = pickle.load(f)
            self.train = f[0]
            self.val = f[1]
            self.test = f[2]
            self.num_atom_type = f[3]
            self.num_bond_type = f[4]
        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels)).unsqueeze(1)
        #tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        #tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        #snorm_n = torch.cat(tab_snorm_n).sqrt()  
        #tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        #tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        #snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)       
        
        return batched_graph, labels
    
    # prepare dense tensors for GNNs using them; such as RingGNN, 3WLGNN
    def collate_dense_gnn(self, samples, edge_feat):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels)).unsqueeze(1)
        #tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        #tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        #snorm_n = tab_snorm_n[0][0].sqrt()  
        
        #batched_graph = dgl.batch(graphs)
    
        g = graphs[0]
        adj = self._sym_normalize_adj(g.adjacency_matrix().to_dense())        
        """
            Adapted from https://github.com/leichen2018/Ring-GNN/
            Assigning node and edge feats::
            we have the adjacency matrix in R^{n x n}, the node features in R^{d_n} and edge features R^{d_e}.
            Then we build a zero-initialized tensor, say T, in R^{(1 + d_n + d_e) x n x n}. T[0, :, :] is the adjacency matrix.
            The diagonal T[1:1+d_n, i, i], i = 0 to n-1, store the node feature of node i. 
            The off diagonal T[1+d_n:, i, j] store edge features of edge(i, j).
        """

        zero_adj = torch.zeros_like(adj)

        if edge_feat:
            # use edge feats also to prepare adj
            adj_with_edge_feat = torch.stack([zero_adj for j in range(self.num_atom_type + self.num_bond_type)])
            adj_with_edge_feat = torch.cat([adj.unsqueeze(0), adj_with_edge_feat], dim=0)

            us, vs = g.edges()      
            for idx, edge_label in enumerate(g.edata['feat']):
                adj_with_edge_feat[edge_label.item()+1+self.num_atom_type][us[idx]][vs[idx]] = 1

            for node, node_label in enumerate(g.ndata['feat']):
                adj_with_edge_feat[node_label.item()+1][node][node] = 1
            
            x_with_edge_feat = adj_with_edge_feat.unsqueeze(0)
            
            return None, x_with_edge_feat, labels
        
        else:
            # use only node feats to prepare adj
            adj_no_edge_feat = torch.stack([zero_adj for j in range(self.num_atom_type)])
            adj_no_edge_feat = torch.cat([adj.unsqueeze(0), adj_no_edge_feat], dim=0)

            for node, node_label in enumerate(g.ndata['feat']):
                adj_no_edge_feat[node_label.item()+1][node][node] = 1

            x_no_edge_feat = adj_no_edge_feat.unsqueeze(0)

            return x_no_edge_feat, None, labels
    
    def _sym_normalize_adj(self, adj):
        deg = torch.sum(adj, dim = 0)#.squeeze()
        deg_inv = torch.where(deg>0, 1./torch.sqrt(deg), torch.zeros(deg.size()))
        deg_inv = torch.diag(deg_inv)
        return torch.mm(deg_inv, torch.mm(adj, deg_inv))
    
    def _add_self_loops(self):
        
        # function for adding self loops
        # this function will be called only if self_loop flag is True
            
        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]

    def _add_positional_encodings(self, pos_enc_dim):
        
        # Graph positional encoding v/ Laplacian eigenvectors
        self.train.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.train.graph_lists]
        self.val.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.val.graph_lists]
        self.test.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.test.graph_lists]

class ZINC_LRP(object):
    def __init__(self, dataset_name = "ZINC", part = "train", lrp_save_path = "dataset", lrp_depth = 1, subtensor_length = 4, lrp_width = 3):
        super(ZINC_LRP, self).__init__()

        self.graphs = []
        self.labels = []

        self.part = part

        self.load_data()
    
        self.subtensor_length = subtensor_length
        self.lrp_depth = lrp_depth
        self.lrp_width = lrp_width

        assert self.subtensor_length == self.lrp_depth * self.lrp_width + 1

        self.num_tasks = 1

        self.output_length = int(subtensor_length ** 2)
        self.lrp_save_path = lrp_save_path
        
        self.lrp_save_file = "lrp_" + dataset_name + '_' + part + "_dep" + str(lrp_depth) + "_wid" + str(lrp_width) + "_len" + str(subtensor_length)
        
        self.lrp_egonet_seq = np.array([])
        self.load_lrp()

        self.num_atom_type = 28 # known meta-info about the zinc dataset; can be calculated as well
        self.num_bond_type = 4 # known meta-info about the zinc dataset; can be calculated as well

    def load_data(self):
        dataset = LoadData('ZINC')
        
        if self.part == "train":
            self.graphs = dataset.train.graph_lists
            self.labels = dataset.train.graph_labels
        elif self.part == "val":
            self.graphs = dataset.val.graph_lists
            self.labels = dataset.val.graph_labels
        elif self.part == "test":
            self.graphs = dataset.test.graph_lists
            self.labels = dataset.test.graph_labels
        else:
            print("WRONG SET")
            assert False
    
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
                node_to_perm_length_indx_col.extend(np.array(perm[1]) + sum_num_nodes_before_graphs[i])
                node_to_perm_length_indx_row.extend(np.array(perm[0])+ sum_row_number)

                edge_to_perm_length_indx_col.extend(np.array(perm[3]) + sum_num_edges_before_graphs[i])
                edge_to_perm_length_indx_row.extend(np.array(perm[2]) + sum_row_number)

                sum_row_number += 16

    node_to_perm_length_size_row = sum_row_number
    node_to_perm_length_size_col = sum(list_num_nodes_in_graphs)
    edge_to_perm_length_size_row = sum_row_number
    edge_to_perm_length_size_col = sum(list_num_edges_in_graphs)

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


def LoadData(DATASET_NAME):
    # handling for (ZINC) molecule dataset
    if DATASET_NAME == 'ZINC':
        return MoleculeDataset(DATASET_NAME)