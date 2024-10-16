# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from data_loaders.load_graph_nmr_chpeaks import graph_nmr_data_2d_peak, custom_collate_fn
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
            for batch in dataloaders[phase]:
                graph, cnmr, hnmr, filename = batch
                # print(filename)
                graph = graph.cuda()
                cnmr = cnmr.cuda()
                hnmr = hnmr.cuda()

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    c_shifts, h_shifts = model(graph)
                    c_nodes = (graph.x[:,0]==5).nonzero(as_tuple=True)[0]
                    h_nodes = (graph.x[:, 0] == 0).nonzero(as_tuple=True)[0] 

                    c_nodes_connected_to_h = []
                    # Check each C node for connection to any H node
                    for c_node in c_nodes:
                        # Get indices of edges involving the C node
                        edges_of_c = (graph.edge_index[0] == c_node) | (graph.edge_index[1] == c_node)

                        # Get all nodes that are connected to this C node
                        connected_nodes = torch.cat((graph.edge_index[0][edges_of_c], graph.edge_index[1][edges_of_c])).unique()

                        # Check if any of these connected nodes are H nodes
                        if any(node in h_nodes for node in connected_nodes):
                            c_nodes_connected_to_h.append(c_node.item())
                    # Convert to a tensor
                    c_nodes_connected_to_h = torch.tensor(c_nodes_connected_to_h).cuda()
                    c_index = [i for i, x in enumerate(c_nodes) if x in c_nodes_connected_to_h]
                    c_shifts = c_shifts[c_index, :]
                    # h_shifts = h_shifts[c_index, :]
                    
                    if train_c:
                        # print('use both loss')
                        loss = nn.MSELoss()(c_shifts, cnmr) + nn.MSELoss()(h_shifts, hnmr)
                    else:
                        # print('only counting H error')
                        loss = nn.MSELoss()(h_shifts, hnmr)
                        # print(loss)

                    # print(c_shifts)
                    # print(cnmr)
                    loss *= 100
                    epoch_loss += loss
                    # print(loss)
                    if torch.isnan(loss):
                        print(filename)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
            epoch_loss = epoch_loss / (len(dataloaders[phase]))
            print(phase + 'loss', epoch_loss)
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

            # save the model weights
            if phase == 'val' and epoch_loss < best_loss:
                print(f"saving best model to {checkpoint_path}")
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

    nmr_path = 'data/data_2d/nmr_2dcsv_chmatched%s/'%args.matching #nmr_2dcsv_expanded
    print(nmr_path)
    graph_path = 'data/data_2d/graph_3d'
    csv_file = 'code/data_csv/nmr_smile_solventclass_filtered4_3dgnn.csv'
    # csv_file = 'code/data_csv/case_largemol.csv'

    # Set a seed for reproducibility
    torch.manual_seed(0)

    dataset = graph_nmr_data_2d_peak(csv_file, graph_path, nmr_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    if args.initial_match:

        # 2d gnn model
        nodeEncoder = GNNNodeEncoder(args.num_layers, args.hidden_channels, JK="last", gnn_type=args.type, aggr='add')
        trained_model = NodeEncodeInterface(nodeEncoder, hidden_channels=args.hidden_channels, c_out_hidden=args.c_out_hidden, h_out_hidden=args.h_out_hidden, c_solvent_emb_dim = args.c_sol_emb_dim, h_solvent_emb_dim = args.h_sol_emb_dim, use_solvent=args.use_solvent)

        ckpt_path = 'model_1dnmr.pt' 
        msg = trained_model.load_state_dict(torch.load(ckpt_path))
        print(msg, ckpt_path)
    else:
        nodeEncoder = GNNNodeEncoder(args.num_layers, args.hidden_channels, JK="last", gnn_type=args.type, aggr='add')
        trained_model = NodeEncodeInterface(nodeEncoder, hidden_channels=args.hidden_channels, c_out_hidden=args.c_out_hidden, h_out_hidden=args.h_out_hidden, c_solvent_emb_dim = args.c_sol_emb_dim, h_solvent_emb_dim = args.h_sol_emb_dim, h_out_channels=2, use_solvent=args.use_solvent)

        ckpt_path = 'model_2dnmr.pt' 
        msg = trained_model.load_state_dict(torch.load(ckpt_path))
        print(msg, ckpt_path)

    # ##TODO can seperate model instances of GNN and H/C MLP layers so that MLP layers can adapt to different datasets by retraining. 
    # model = ComENet(in_embed_size=3, c_out_channels=1, h_out_channels=2, agg_method='sum', \
    #                 hidden_channels=args.hidden_channels, c_out_hidden=args.c_out_hidden,\
    #                  h_out_hidden=args.h_out_hidden, num_layers=args.num_layers, num_output_layers=args.num_output_layers)
    
    # 2d gnn model
    nodeEncoder = GNNNodeEncoder(args.num_layers, args.hidden_channels, JK="last", gnn_type=args.type, aggr='add')
    model = NodeEncodeInterface(nodeEncoder, hidden_channels=args.hidden_channels, c_out_hidden=args.c_out_hidden, h_out_hidden=args.h_out_hidden, c_solvent_emb_dim = args.c_sol_emb_dim, h_solvent_emb_dim = args.h_sol_emb_dim, h_out_channels=2, use_solvent=args.use_solvent)
    
    
    # load parameters from trained model, for H-MLP layer, duplicate the weights
    if args.initial_match:
        print('initial match')
        for name, module in trained_model.named_children():
            # Check if the current model has the component
            if hasattr(model, name):
                # Special handling for the 'h_projection' component
                if name == 'lin_out_h':
                    # Ensure the h_projection module in the current model also has a 'model' attribute
                    if hasattr(getattr(model, name), 'model'):
                        # Copy all layers of 'h_projection.model' except the last one
                        h_proj_model = getattr(model, name).model
                        for idx, sub_module in enumerate(module.model):
                            if idx < len(module.model) - 1:  # Skip the last layer
                                h_proj_model[idx].load_state_dict(sub_module.state_dict())
                            else:
                                # Handle the last layer
                                source_layer = sub_module
                                target_layer = h_proj_model[idx]

                                with torch.no_grad():
                                    # Duplicate the weights and biases for each output channel
                                    # Assuming the output channels are doubled
                                    duplicated_weights = torch.repeat_interleave(source_layer.weight.data, 2, dim=0)
                                    duplicated_bias = torch.repeat_interleave(source_layer.bias.data, 2, dim=0)

                                    # Check if the dimensions match
                                    if duplicated_weights.shape == target_layer.weight.data.shape and \
                                    duplicated_bias.shape == target_layer.bias.data.shape:
                                        target_layer.weight.data = duplicated_weights
                                        target_layer.bias.data = duplicated_bias
                                    else:
                                        print(f"Dimension mismatch in layer {name}")
                    print('copied parameter for layer: ', name)

                else:
                    # For other components, directly copy the state dict
                    getattr(model, name).load_state_dict(module.state_dict())
                    print('copied parameter for layer: ', name)
    else:
        print('autoregressive match')
        for name, module in trained_model.named_children():
            if hasattr(model, name):
                getattr(model, name).load_state_dict(module.state_dict())
                print('copied parameter for layer: ', name)

    print(model)

    count_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model parameter: %d" % count_param)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=8, gamma=0.9)

    ckpt_path = 'model_2dnmr.pt' 
    model = train_model(model, dataloaders, optimizer_ft, exp_lr_scheduler, ckpt_path, num_epochs=args.n_epoch, train_c=True)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--batch_size', type=int, default=32, help='batch size')
    args.add_argument('--n_epoch', type=int, default=40, help='num of epoches')
    args.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    args.add_argument('--type', type=str, default='gin', help='GNN type')
    args.add_argument('--hidden_channels', type=int, default=512, help='hidden channel of gnn')
    args.add_argument('--num_layers', type=int, default=5, help='number of layers for GNN')
    # args.add_argument('--num_output_layers', type=int, default=2, help='number of layers for GNN')
    args.add_argument('--agg_method', type=str, default='sum', help='aggregation method for GNN')
    args.add_argument('--c_out_hidden', default=[128, 64], type=int, nargs="+", help='hidden dims of projection')
    args.add_argument('--h_out_hidden', default=[128, 64], type=int, nargs="+", help='hidden dims of projection')
    args.add_argument('--c_sol_emb_dim', type=int, default=16, help='carbon solvent embedding dimension')
    args.add_argument('--h_sol_emb_dim', type=int, default=32, help='hydrogen solvent embedding dimension')

    
    args = args.parse_args()

    args.initial_match = False # True

    args.use_solvent = True

    if args.initial_match:
        args.matching = ''
    else:
        args.matching = '_5th' 
        args.prev = '_4th'

    main(args)
# %%
