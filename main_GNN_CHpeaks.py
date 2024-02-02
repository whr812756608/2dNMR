# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from data_loaders.load_graph_nmr_chpeaks import graph_nmr_data_2d_peak, custom_collate_fn

from GraphModel.Comenet_NMR_multitask import ComENet
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import time
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse
import os


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
            # feed text and images into diffusion prior network
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

                    ##### calculate the indices of C node connected to H
                    # Initialize a list to store C nodes connected to H
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

    nmr_path = '/scratch0/yunruili/2dnmr_30k/nmr_2dcsv_chmatched/' #nmr_2dcsv_expanded
    graph_path = '/scratch0/yunruili/2dnmr_30k/graph_3d/'
    csv_file = './data_csv/nmr_smile_solvent_filtered2_3dgnn.csv'

    # Set a seed for reproducibility
    torch.manual_seed(0)

    dataset = graph_nmr_data_2d_peak(csv_file, graph_path, nmr_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    ## current approach: train multi task alignment model, then finetune. H layer now predicts 2 values instead of 1.
    trained_model = ComENet(in_embed_size=3, c_out_channels=1, h_out_channels=1, agg_method='sum', \
                            hidden_channels=args.hidden_channels, c_out_hidden=args.c_out_hidden,\
                     h_out_hidden=args.h_out_hidden, num_layers=args.num_layers, num_output_layers=args.num_output_layers)

    ckpt_path = 'gnn3d_multi_align_%s_hiddendim_%d_nlayers_%d_noutlayers_%d_couthidden_%s_houthidden_%s.pt' % \
        (args.agg_method, args.hidden_channels, args.num_layers, args.num_output_layers, ''.join(str(i) for i in args.c_out_hidden), ''.join(str(i) for i in args.h_out_hidden))
    
    msg = trained_model.load_state_dict(torch.load(ckpt_path))
    print(msg)

    ##TODO can seperate model instances of GNN and H/C MLP layers so that MLP layers can adapt to different datasets by retraining. 
    model = ComENet(in_embed_size=3, c_out_channels=1, h_out_channels=2, agg_method='sum', \
                    hidden_channels=args.hidden_channels, c_out_hidden=args.c_out_hidden,\
                     h_out_hidden=args.h_out_hidden, num_layers=args.num_layers, num_output_layers=args.num_output_layers)
    
    # load parameters from trained model, for H-MLP layer, duplicate the weights
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
    
    # # check if the parameters are copied correctly
    # for idx, (model_layer, trained_layer) in enumerate(zip(model.lin_out_h.model, trained_model.lin_out_h.model)):
    #     if idx < len(model.lin_out_h.model) - 1:  # Skip the last layer
    #         # Compare parameters of the current layer in both models
    #         are_same = all((torch.equal(a, b) for a, b in zip(model_layer.parameters(), trained_layer.parameters())))
            
    #     else:
    #         # For the last layer, compare only the first half of the parameters
    #         are_same = True
    #         for a, b in zip(model_layer.parameters(), trained_layer.parameters()):
    #             half_size = a.data.size(0) // 2  # Assuming the output features are doubled
    #             are_same = are_same and torch.equal(a.data[:half_size], b.data)
    #     print(f"Layer {idx} parameters are {'identical' if are_same else 'different'}")


    # trained model attributes: feature1 feature2 emb interaction_blocks lins lin_out, lin_out_h(out channel = 1)
    freeze_list = ['feature1', 'feature2', 'emb', 'interaction_blocks', 'lins']

    # freeze layers
    # Freeze the specified named children
    for name, child in model.named_children():
        if name in freeze_list:
            for param in child.parameters():
                param.requires_grad = False

    print(model)

    count_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model parameter: %d" % count_param)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"Layer: {name}, number of params: {param.numel()}")

    # Fine-tuning setup
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=8, gamma=0.9)

    ckpt_path = 'gnn3d_2dch_%s_hiddendim_%d_nlayers_%d_noutlayers_%d_couthidden_%s_houthidden_%s.pt' % \
        (args.agg_method, args.hidden_channels, args.num_layers, args.num_output_layers, ''.join(str(i) for i in args.c_out_hidden), ''.join(str(i) for i in args.h_out_hidden))
    print(ckpt_path)
    print( 'final_%s'%ckpt_path)

    # if os.path.exists(ckpt_path):
    #     msg = model.load_state_dict(torch.load(ckpt_path))
    #     print(msg)
    #     print('model loaded')
    model = train_model(model, dataloaders, optimizer_ft, exp_lr_scheduler, ckpt_path, num_epochs=args.n_epoch, train_c=True)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--batch_size', type=int, default=32, help='batch size')
    args.add_argument('--n_epoch', type=int, default=100, help='batch size')
    args.add_argument('--lr', type=float, default=0.0001, help='batch size')
    args.add_argument('--hidden_channels', type=int, default=256, help='hidden channel of gnn')
    args.add_argument('--num_layers', type=int, default=3, help='number of layers for GNN')
    args.add_argument('--num_output_layers', type=int, default=2, help='number of layers for GNN')
    args.add_argument('--agg_method', type=str, default='sum', help='aggregation method for GNN')
    args.add_argument('--c_out_hidden', default=[128, 64, 64], type=int, nargs="+", help='hidden dims of projection')
    args.add_argument('--h_out_hidden', default=[128, 64, 64], type=int, nargs="+", help='hidden dims of projection')
    
    args = args.parse_args()
    main(args)
# %%
