import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from data_loaders.load_graph_cnmr_hnmr_alignment import graph_nmr_alignment_data, custom_collate_fn, CustomBatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import time
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse
import os
from GraphModel.GNN_2d_hsolvent import GNNNodeEncoder, NodeEncodeInterface

def train_model(model, dataloaders, optimizer, scheduler, checkpoint_path, num_epochs=1, train_c=True):
    best_loss = 1e10

    model = model.cuda()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0
            for i, batch in enumerate(dataloaders[phase]):
                graph, cnmr, hnmr, filename = batch

                graph = graph.cuda()
                cnmr = cnmr.cuda() if cnmr is not None else None
                hnmr = hnmr.cuda() if hnmr is not None else None

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    try:
                        c_shifts, h_shifts = model(graph)
                        if graph[0].has_h:
                            if graph[0].has_c:
                                loss = (nn.MSELoss()(c_shifts, cnmr) + nn.MSELoss()(h_shifts, hnmr))/2
                            else:
                                loss = nn.MSELoss()(h_shifts, hnmr)
                        else:
                            loss = nn.MSELoss()(c_shifts, cnmr)
                    except Exception as e:
                        print(filename)
                        print(e)

                    loss *= 100
                    epoch_loss += loss
                    if torch.isnan(loss):
                        print(filename)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
            print(i)
            epoch_loss = epoch_loss / i
            print(phase + 'loss', epoch_loss)
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

            # save the model weights
            if phase == 'val' and epoch_loss < best_loss:
                #print(f"saving best model to {checkpoint_path}")
                best_loss = epoch_loss
                torch.save(model.state_dict(), checkpoint_path)

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # save the last trained model
    torch.save(model.state_dict(), 'final_%s'%checkpoint_path)

    # load best model weights
    model.load_state_dict(torch.load(checkpoint_path))
    return model

def main(args):

    graph_path = '/scratch0/haox/2DNMR_prediction_gt/Datasets/graph3d/'
    nmr_path = '/scratch0/haox/yunruili/'
    csv_cnmr = './data_csv/1dnmr/filtered_cnmr_smile_dataset_22k.csv'
    csv_hnmr = './data_csv/1dnmr/filtered_hnmr_smile_dataset_67.csv'
    csv_common = './data_csv/1dnmr/filtered_common_smile_dataset_1600.csv'
    dataset_c = graph_nmr_alignment_data(csv_cnmr, graph_path, nmr_path, type='c')
    dataset_h = graph_nmr_alignment_data(csv_hnmr, graph_path, nmr_path, type='h')
    dataset_ch = graph_nmr_alignment_data(csv_common, graph_path, nmr_path, type='both')

    # Set the seed for reproducibility
    torch.manual_seed(0)  # Replace your_seed with your chosen seed value

    # Define the proportions or absolute sizes for your train and val sets
    train_size_c = int(0.8 * len(dataset_c))
    val_size_c = len(dataset_c) - train_size_c

    train_size_h = int(0.8 * len(dataset_h))
    val_size_h = len(dataset_h) - train_size_h

    train_size_ch = int(0.8 * len(dataset_ch))
    val_size_ch = len(dataset_ch) - train_size_ch

    # Split the datasets
    train_dataset_c, val_dataset_c = random_split(dataset_c, [train_size_c, val_size_c])
    train_dataset_h, val_dataset_h = random_split(dataset_h, [train_size_h, val_size_h])
    train_dataset_ch, val_dataset_ch = random_split(dataset_ch, [train_size_ch, val_size_ch])

    # Create DataLoaders for training and validation sets
    train_dataloader_c = DataLoader(train_dataset_c, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader_c = DataLoader(val_dataset_c, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)

    train_dataloader_h = DataLoader(train_dataset_h, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader_h = DataLoader(val_dataset_h, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)

    train_dataloader_ch = DataLoader(train_dataset_ch, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader_ch = DataLoader(val_dataset_ch, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)

    # Now you can create custom loaders for each set
    train_custom_loader = CustomBatchSampler(train_dataloader_c, train_dataloader_ch, train_dataloader_h, n1=7, n2=80)
    val_custom_loader = CustomBatchSampler(val_dataloader_c, val_dataloader_ch, val_dataloader_h, n1=14, n2=141)

    print('c', len(train_dataloader_c))
    print('h', len(train_dataloader_h))
    print('ch', len(train_dataloader_ch))

    print('c', len(val_dataloader_c))
    print('h', len(val_dataloader_h))
    print('ch', len(val_dataloader_ch))

    dataloaders = {'train': train_custom_loader, 'val': val_custom_loader}

    # 2d gnn model
    #nodeEncoder = GNNNodeEncoder(args.num_layers, args.hidden_channels, JK="last", gnn_type=args.type, aggr='add')
    #model = NodeEncodeInterface(nodeEncoder, hidden_channels=args.hidden_channels, c_out_hidden=args.c_out_hidden, h_out_hidden=args.h_out_hidden, solvent_emb_dim = args.sol_emb_dim)
    # 3d gnn model
    
     model = ComENet(in_embed_size=3, out_channels=1, agg_method='sum', hidden_channels=args.hidden_channels, c_out_hidden=args.c_out_hidden,\
                     # h_out_hidden=args.h_out_hidden, num_layers=args.num_layers, num_output_layers=args.num_output_layers)

    print(model)

    count_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model parameter: %d" % count_param)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=8, gamma=0.9)

    ckpt_path = 'model_1dnmr.pt'

    model = train_model(model, dataloaders, optimizer_ft, exp_lr_scheduler, ckpt_path, num_epochs=args.n_epoch, train_c=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--batch_size', type=int, default=32, help='batch size')
    args.add_argument('--n_epoch', type=int, default=200, help='number of epochs')
    args.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    args.add_argument('--type', type=str, default='gin', help='GNN type')
    args.add_argument('--hidden_channels', type=int, default=512, help='embedding dimension of gnn')
    args.add_argument('--num_layers', type=int, default=5, help='number of layers for GNN')
    args.add_argument('--agg_method', type=str, default='sum', help='aggregation method for GNN')
    args.add_argument('--c_out_hidden', default=[128, 64], type=int, nargs="+", help='hidden dims of projection')
    args.add_argument('--h_out_hidden', default=[128, 64], type=int, nargs="+", help='hidden dims of projection')
    args.add_argument('--sol_emb_dim', type=int, default=32, help='solvent embedding dimension')
 
    args = args.parse_args()
    main(args)
