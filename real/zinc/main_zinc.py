import os
import torch
# from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

from tqdm import tqdm
import argparse
import time
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold

import random 

from dataset_zinc import ZINC_LRP, collate_lrp_dgl_light, collate_lrp_dgl_light_index_form_wrapper
from torch.utils.data import DataLoader

# from mpnn_custom import MPNNModel, LRP_MPNN
from model_lrp_zinc import LRP_PURE

import csv

reg_criterion = torch.nn.L1Loss()
cls_criterion = torch.nn.CrossEntropyLoss()

import sys

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

def output_csv(file_name, test_acc_list):
    with open(file_name, mode='w') as output_file:
        output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for acc in test_acc_list:
            output_writer.writerow([acc])

def separate_data(labels, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    return train_idx, test_idx

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

def train(model, device, loader, optimizer):
    time0 = time.time()
    model.train()
    epoch_loss = []

    # for iter, (batch, split_list, sp_matrices, label) in tqdm(enumerate(loader)):
    for iter, (batch, pooling_matrix, sp_matrices, label) in tqdm(enumerate(loader)):

        batch = batch.to(device)
        # mean_pooling_matrix = build_perm_pooling_sp_matrix(split_list, "mean").to(device)

        pooling_matrix = [torch.LongTensor(pooling_matrix[0]).to(device), torch.FloatTensor(pooling_matrix[1]).to(device), pooling_matrix[2], pooling_matrix[3]]

        # n_to_perm_length_sp_matrix = np_sparse_to_pt_sparse(sp_matrices[0]).to(device)
        # e_to_perm_length_sp_matrix = np_sparse_to_pt_sparse(sp_matrices[1]).to(device)

        n_to_perm = [torch.LongTensor(sp_matrices[0]).to(device), torch.FloatTensor(sp_matrices[1]).to(device), sp_matrices[2], sp_matrices[3]]
        e_to_perm = [torch.LongTensor(sp_matrices[4]).to(device), torch.FloatTensor(sp_matrices[5]).to(device), sp_matrices[6], sp_matrices[7]]

        degs = batch.in_degrees(list(range(batch.number_of_nodes()))).type(torch.FloatTensor).to(device)
        # pred = model(batch, mean_pooling_matrix, degs, n_to_perm_length_sp_matrix, e_to_perm_length_sp_matrix)
        pred = model(batch, pooling_matrix, degs, n_to_perm, e_to_perm)
        optimizer.zero_grad()

        label = label.unsqueeze(1)

        loss = reg_criterion(pred, label.to(device))
        loss.backward()

        optimizer.step()

        loss = loss.detach().cpu().numpy()
        epoch_loss.append(loss)

    print(time.time()-time0)

    return np.mean(epoch_loss)

def eval(model, device, loader):
    model.eval()
    epoch_loss = []

    with torch.no_grad():
        for iter, (batch, pooling_matrix, sp_matrices, label) in tqdm(enumerate(loader)):
            batch = batch.to(device)
            # mean_pooling_matrix = build_perm_pooling_sp_matrix(split_list, "mean").to(device)
            
            pooling_matrix = [torch.LongTensor(pooling_matrix[0]).to(device), torch.FloatTensor(pooling_matrix[1]).to(device), pooling_matrix[2], pooling_matrix[3]]

            # n_to_perm_length_sp_matrix = np_sparse_to_pt_sparse(sp_matrices[0]).to(device)
            # e_to_perm_length_sp_matrix = np_sparse_to_pt_sparse(sp_matrices[1]).to(device)

            n_to_perm = [torch.LongTensor(sp_matrices[0]).to(device), torch.FloatTensor(sp_matrices[1]).to(device), sp_matrices[2], sp_matrices[3]]
            e_to_perm = [torch.LongTensor(sp_matrices[4]).to(device), torch.FloatTensor(sp_matrices[5]).to(device), sp_matrices[6], sp_matrices[7]]

            degs = batch.in_degrees(list(range(batch.number_of_nodes()))).type(torch.FloatTensor).to(device)

            # pred = model(batch, mean_pooling_matrix, degs, n_to_perm_length_sp_matrix, e_to_perm_length_sp_matrix)

            output = model(batch, pooling_matrix, degs, n_to_perm, e_to_perm)

            # pred = output.max(1, keepdim=True)[1].to('cpu')
            # correct += pred.eq(label.view_as(pred)).sum().cpu().item()

            label = label.unsqueeze(1)

            loss = reg_criterion(output, label.to(device))
            loss = loss.detach().cpu().numpy()
            epoch_loss.append(loss)

    return np.mean(epoch_loss)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='lrp_pure',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=4,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--hid_dim', type=int, default=128,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ZINC",
                        help='dataset name (default: MUTAG)')

    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    parser.add_argument('--seed', type=int, default = 0)
    parser.add_argument('--decay_step', type=int, default = 50)

    parser.add_argument('--lr', type=float, default = 0.001)
    parser.add_argument('--bn', action = 'store_true', default = False)
    parser.add_argument('--mlp', action = 'store_true', default = False)
    parser.add_argument('--scheduler', action = 'store_true', default = False)
    parser.add_argument('--decay_rate', type=float, default = 0.5)
    parser.add_argument('--model_save_name', type=str, default="")
    parser.add_argument('--lrp_length', type=int, default=4)
    parser.add_argument('--lrp_depth', type=int, default=1)
    parser.add_argument('--lrp_width', type=int, default=3)
    parser.add_argument('--output_folder', type=str, default="results_gindataset")
    parser.add_argument('--output_file', type=str, default="0")
    parser.add_argument('--alldegree', action = 'store_true', default = False)
    
    parser.add_argument('--lrp_save_path', type = str, default = "./data/lrp_preprocessed/")
    args = parser.parse_args()
   
    print(args)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    model_save_name = "model_" + str(int(time.time())) 
    print("The new model name:", model_save_name)
    print("The old model name:", args.model_save_name)

    torch.manual_seed(args.seed)
    random.seed(0)
    np.random.seed(0)  
    os.environ['PYTHONHASHSEED'] = str(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.enabled=False
        torch.backends.cudnn.deterministic=True

    dataset_train = ZINC_LRP(part = "train", lrp_save_path = args.lrp_save_path, lrp_depth = args.lrp_depth, subtensor_length = args.lrp_length, lrp_width = args.lrp_width)
    dataset_val = ZINC_LRP(part = "val", lrp_save_path = args.lrp_save_path, lrp_depth = args.lrp_depth, subtensor_length = args.lrp_length, lrp_width = args.lrp_width)
    dataset_test = ZINC_LRP(part = "test", lrp_save_path = args.lrp_save_path, lrp_depth = args.lrp_depth, subtensor_length = args.lrp_length, lrp_width = args.lrp_width)

    num_tasks = dataset_val.num_tasks # obtaining the number of prediction tasks in a dataset

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_lrp_dgl_light_index_form_wrapper(args.lrp_length), num_workers = args.num_workers)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, collate_fn=collate_lrp_dgl_light_index_form_wrapper(args.lrp_length), num_workers = args.num_workers)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate_lrp_dgl_light_index_form_wrapper(args.lrp_length), num_workers = args.num_workers)

    hyperparam_list = {
        # 'lrp_length': 16,
        # 'num_tasks': dataset.num_tasks,
        # 'lrp_in_dim': 13,
        # 'hid_dim': 13,
        # 'num_layers': 4,
        # 'lr': args.lr
        'lrp_length': int(args.lrp_length ** 2),
        'num_tasks': num_tasks,
        'lrp_in_dim': 28, # node feats dim
        'hid_dim': args.hid_dim,
        'num_layers': args.num_layer,
        'bn': args.bn,
        'lr': args.lr,
        'mlp': args.mlp,
        'alldegree': args.alldegree
    }

    model = LRP_PURE(
        num_tasks = hyperparam_list['num_tasks'],
        lrp_length = hyperparam_list['lrp_length'],
        lrp_in_dim = hyperparam_list['lrp_in_dim'],
        hid_dim = hyperparam_list['hid_dim'],
        num_layers = hyperparam_list['num_layers'],
        bn = hyperparam_list['bn'],
        mlp = hyperparam_list['mlp'],
        alldegree = hyperparam_list['alldegree']
    ).to(device)

    print('#params: ', count_parameters(model))
    print(hyperparam_list)

    # exit()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 50], gamma=0.5)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size = args.decay_step, gamma=args.decay_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.5, patience = 10)

    if args.model_save_name != "":
        checkpoint = torch.load('checkpoint/' + args.model_save_name + '.pth')
        start_epoch = checkpoint['epoch']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']
    else:
        start_epoch = 0 

    train_curve = []
    val_curve = []
    test_curve = []

    for epoch in range(1 + start_epoch, args.epochs + 1 + start_epoch):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_loss = train(model, device, train_loader, optimizer)

        print('Evaluating...')
        train_loss = eval(model, device, train_loader)
        val_loss = eval(model, device, val_loader)
        test_loss = eval(model, device, test_loader)

        train_curve.append(train_loss)
        val_curve.append(val_loss)
        test_curve.append(test_loss)

        # print({'Train': train_loss})
        print({'Train loss:': train_loss, 'Validation loss:': val_loss, 'Test loss:': test_loss})

        if args.scheduler:
            scheduler.step(val_loss)
            print("current lr decay rate:", optimizer.param_groups[0]['lr']/args.lr)

        if optimizer.param_groups[0]['lr'] < 0.01 * args.lr:
            print("\n!! LR EQUAL TO MIN LR SET.")

            best_idx = np.argmin(val_curve)
            print("Best: ", {'Train loss:': train_curve[best_idx], 'Validation loss:': val_curve[best_idx], 'Test loss:': test_curve[best_idx]})
            assert False

        # exit()

        # train_curve.append(train_acc)
        # val_curve.append(val_acc)

        # print(test_multitask_loss)
   
        # checkpoint = {
        #     'epoch': epoch,
        #     'model': model,
        #     'optimizer': optimizer,
        #     'scheduler': scheduler,
        # }
        # torch.save(checkpoint, 'checkpoint/'+model_save_name+'.pth')

    # output_csv(args.output_folder + '/' + args.dataset + '_' + args.output_file + '_' + str(args.fold_idx) + '.csv', test_acc_list)

if __name__ == "__main__":
    main()
