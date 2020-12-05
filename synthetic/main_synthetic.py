import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import os, sys
from tqdm import tqdm
import argparse
import time
import random
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from dataset_synthetic import DglSyntheticDataset, collate_lrp_dgl_light
from torch.utils.data import DataLoader

from model_synthetic import LRP_synthetic

reg_criterion = torch.nn.MSELoss()

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def np_sparse_to_pt_sparse(matrix):
    coo = matrix.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def build_perm_pooling_sp_matrix(split_list, pooling = "sum"):
    dim0, dim1 = len(split_list), sum(split_list)
    col = np.arange(dim1)
    row = np.array([i for i, count in enumerate(split_list) for j in range(count)])
    data = np.ones((dim1, ))
    pooling_sp_matrix = csr_matrix((data, (row, col)), shape = (dim0, dim1))

    if pooling == "mean":
        pooling_sp_matrix = normalize(pooling_sp_matrix, norm='l1', axis=1)
    
    return np_sparse_to_pt_sparse(pooling_sp_matrix)

def train(model, device, loader, optimizer, args):
    model.train()
    epoch_loss = []

    for iter, (batch, split_list, sp_matrices, label) in tqdm(enumerate(loader)):
        # egonet: n_nodes x n_perm x length**2 x (1 + n_feats + e_feats)
        # print ('label', label)
        # print ('batch', batch.ndata['feat'])

        batch = batch.to(device)
        # batch.ndata['feat'] = torch.cat((torch.ones(batch.number_of_nodes(), 1), torch.zeros(batch.number_of_nodes(), 1)), dim=1).to(device)
        # batch.edata['feat'] = torch.cat((torch.zeros(batch.number_of_edges(), 1), torch.ones(batch.number_of_edges(), 1)), dim=1).to(device)
        if args.task == 'attributed_triangle':
            # batch.ndata['feat'] = torch.FloatTensor([[i % 2, 1 - i % 2] for i in range(batch.number_of_nodes())]).to(device)
            # # batch.ndata['feat'] = torch.ones(batch.number_of_nodes(), 1).to(device)
            # batch.edata['feat'] = torch.ones(batch.number_of_edges(), 2).to(device)
            pass
        else:
            batch.ndata['feat'] = torch.ones(batch.number_of_nodes(), 1).to(device)
            batch.edata['feat'] = torch.ones(batch.number_of_edges(), 1).to(device)
        # print ('ndata', batch.ndata['feat'].shape)
        # print ('edata', batch.edata['feat'].shape)
        # batch_input_egonet = [[torch.FloatTensor(ego).to(device), torch.FloatTensor([batch.in_degrees(i)]).to(device)] for i, ego in enumerate(egonets)]
        # split_list = [len(ego) for ego in egonets]
        # mean_pooling_matrix = build_perm_pooling_sp_matrix(split_list, "mean").to(device)
        mean_pooling_matrix = build_perm_pooling_sp_matrix(split_list, "mean").to(device)
        # egonets = np.concatenate(egonets, axis = 0)

        n_to_perm_length_sp_matrix = np_sparse_to_pt_sparse(sp_matrices[0]).to(device)
        e_to_perm_length_sp_matrix = np_sparse_to_pt_sparse(sp_matrices[1]).to(device)

        degs = batch.in_degrees(list(range(batch.number_of_nodes()))).type(torch.FloatTensor).to(device)
        
        # batch_input_egonet = [torch.FloatTensor(egonets).to(device), batch.in_degrees(list(range(batch.number_of_nodes()))).type(torch.FloatTensor).to(device), mean_pooling_matrix.to(device)]
        
        # pred = model(batch, batch.ndata['feat'].type(torch.cuda.FloatTensor), batch.edata['feat'].type(torch.cuda.FloatTensor))
        # pred = model(batch, batch_input_egonet, batch.edata['feat'].type(torch.cuda.FloatTensor))
        pred = model(batch, mean_pooling_matrix, degs, n_to_perm_length_sp_matrix, e_to_perm_length_sp_matrix)
        optimizer.zero_grad()
        # print(pred)
 
        # print(pred, label)
        # assert False
        # print(pred.size(), label.unsqueeze(1).size())
        # assert False
        loss = reg_criterion(pred.to(torch.float32), label.to(device).to(torch.float32).unsqueeze(1))
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.detach().item())
        # y_true.append(label.detach().cpu())
        # y_pred.append(pred.detach().cpu())

        # print ('y_true', label)
        # print ('y_pred', pred)

    # y_true = torch.cat(y_true, dim = 0).numpy()
    # y_pred = torch.cat(y_pred, dim = 0).numpy()


    # input_dict = {"y_true": y_true, "y_pred": y_pred}

    return np.mean(epoch_loss)

def eval(model, device, loader, args):
    model.eval()
    epoch_loss = []

    with torch.no_grad():
        for iter, (batch, split_list, sp_matrices, label) in tqdm(enumerate(loader)):
            batch = batch.to(device)
            # batch.ndata['feat'] = torch.cat((torch.ones(batch.number_of_nodes(), 1), torch.zeros(batch.number_of_nodes(), 1)), dim=1).to(device)
            # batch.edata['feat'] = torch.cat((torch.zeros(batch.number_of_edges(), 1), torch.ones(batch.number_of_edges(), 1)), dim=1).to(device)
            if args.task == 'attributed_triangle':
                # batch.ndata['feat'] = torch.FloatTensor([[i % 2, 1 - i % 2] for i in range(batch.number_of_nodes())]).to(device)
                # # batch.ndata['feat'] = torch.ones(batch.number_of_nodes(), 1).to(device)
                # batch.edata['feat'] = torch.ones(batch.number_of_edges(), 2).to(device)
                pass
            else:
                batch.ndata['feat'] = torch.ones(batch.number_of_nodes(), 1).to(device)
                batch.edata['feat'] = torch.ones(batch.number_of_edges(), 1).to(device)
            mean_pooling_matrix = build_perm_pooling_sp_matrix(split_list, "mean").to(device)

            n_to_perm_length_sp_matrix = np_sparse_to_pt_sparse(sp_matrices[0]).to(device)
            e_to_perm_length_sp_matrix = np_sparse_to_pt_sparse(sp_matrices[1]).to(device)

            degs = batch.in_degrees(list(range(batch.number_of_nodes()))).type(torch.FloatTensor).to(device)
            
            pred = model(batch, mean_pooling_matrix, degs, n_to_perm_length_sp_matrix, e_to_perm_length_sp_matrix)
    
            loss = reg_criterion(pred.to(torch.float32), label.to(device).to(torch.float32).unsqueeze(1))
            epoch_loss.append(loss.detach().item())

    return np.mean(epoch_loss)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='lrp_pure',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    parser.add_argument('--seed', type=int, default = 0)

    parser.add_argument('--lr', type=float, default = 0.001)
    parser.add_argument('--bn', action = 'store_true', default = False)
    parser.add_argument('--mlp', action = 'store_true', default = False)
    parser.add_argument('--dataset', type=int, default=1)
    parser.add_argument('--task', type=str, default='triangle')
    parser.add_argument('--full_permutation', action = 'store_true', default = False)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(0)
    np.random.seed(0)  
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    timestamp = int(time.time())
    save_model_name = 'save_model/lrp' + str(args.seed) + args.task + '_' + 'data' + str(args.dataset) + '_' + str(timestamp) + '.pkl'
    save_curve_name = 'results/lrp' + str(args.seed) + args.task + '_' + 'data' + str(args.dataset) + '_' + str(timestamp)

    print("Save model name:", save_model_name)
    print("Save curve name:", save_curve_name)

    if args.task not in ['star', 'triangle', 'tailed_triangle', 'chordal_cycle', 'attributed_triangle']:
        print("WRONG TASK!")
        exit()

    if args.dataset not in [1, 2]:
        print("WRONG DATASET!")
        exit()

    dataset = DglSyntheticDataset(dataset = args.dataset, task = args.task, full_permutation = args.full_permutation)

    train_idx, val_idx, test_idx = dataset.train_idx, dataset.val_idx, dataset.test_idx

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    train_loader = DataLoader(dataset[train_idx], batch_size=args.batch_size, shuffle=True, collate_fn=collate_lrp_dgl_light)
    valid_loader = DataLoader(dataset[val_idx], batch_size=args.batch_size, shuffle=False, collate_fn=collate_lrp_dgl_light)
    test_loader = DataLoader(dataset[test_idx], batch_size=args.batch_size, shuffle=False, collate_fn=collate_lrp_dgl_light)

    if args.task == 'attributed_triangle':
        hyperparam_list = {
                'lrp_length': 16,
                'num_tasks': 1,
                'lrp_in_dim': 2,
                'hid_dim': 128,
                'num_layers': 1,
                'bn': False,
                'lr': args.lr,
                'mlp': False
        }
    else:
        hyperparam_list = {
                'lrp_length': 16,
                'num_tasks': 1,
                'lrp_in_dim': 1,
                'hid_dim': 128,
                'num_layers': 1,
                'bn': False,
                'lr': args.lr,
                'mlp': False
        }

    model = LRP_synthetic(
        num_tasks = hyperparam_list['num_tasks'],
        lrp_length = hyperparam_list['lrp_length'],
        lrp_in_dim = hyperparam_list['lrp_in_dim'],
        hid_dim = hyperparam_list['hid_dim'],
        num_layers = hyperparam_list['num_layers'],
        bn = hyperparam_list['bn'],
        mlp = hyperparam_list['mlp']
    ).to(device)

    print('#params: ', count_parameters(model))
    print(hyperparam_list)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_curve = []
    valid_curve = []
    test_curve = []

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_loss = train(model, device, train_loader, optimizer, args)/dataset.variance

        print('Evaluating...')
        valid_loss = eval(model, device, valid_loader, args)/dataset.variance
        test_loss = eval(model, device, test_loader, args)/dataset.variance

        print({'Train': train_loss, 'Validation': valid_loss, 'Test': test_loss})

        if best_val_loss > valid_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), save_model_name)

        train_curve.append(train_loss)
        valid_curve.append(valid_loss)
        test_curve.append(test_loss)

    best_val_epoch = np.argmin(np.array(valid_curve))
    best_train = min(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    np.savez(save_curve_name, train = np.array(train_curve), val = np.array(valid_curve), test = np.array(test_curve), test_for_best_val = test_curve[best_val_epoch])

if __name__ == "__main__":
    main()