# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from load_graph_nmr import graph_nmr_data
from Comenet_NMR import ComENet
from torch_geometric.data import DataLoader
from torch.utils.data import random_split
from torch.cuda.amp import autocast, GradScaler
import time
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse
import os


def train_model(model, dataloaders, optimizer, scheduler, checkpoint_path, num_epochs=1):
    best_loss = 1e10
    scaler = GradScaler()

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
                graph, nmr, nmr_noise, filename = batch
                graph = graph.cuda()
                nmr = nmr.cuda()
                nmr_noise = nmr_noise.cuda()

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # print(next(model.parameters()).is_cuda)
                    # print(smile.get_device())
                    # print(nmr_seq.get_device())
                    # with autocast():
                    loss, out_agg = model(graph, nmr_noise)
                    # print(loss)
                    loss *= 100
                    epoch_loss += loss
                    # print(loss)
                    if torch.isnan(loss):
                        print(filename)
                        print(torch.isnan(out_agg).any())
                    if phase == 'train':
                        # scaler.scale(loss).backward()
                        # scaler.step(optimizer)
                        # scaler.update()

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

    csv_file = 'nmr_smile_solvent_filtered_3dgnn.csv'
    graph_path = '/scratch0/yunruili/2dnmr_30k/graph_3d/'
    nmr_path = '/scratch0/yunruili/2dnmr_30k/nmr_1dcsv_30k/'
    dataset = graph_nmr_data(csv_file, graph_path, nmr_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    # Create an instance of the network
    model = ComENet(in_embed_size=3, out_channels=1001, \
                    agg_method=args.agg_method, hidden_channels=args.hidden_channels, out_hidden=args.out_hidden, \
                        num_layers=args.num_layers, num_output_layers=args.num_output_layers)
    print(model)

    count_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model parameter: %d" % count_param)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"Layer: {name}, number of params: {param.numel()}")

    # Fine-tuning setup
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=8, gamma=0.9)

    ckpt_path = 'gnn3d_ch_%s_hiddendim_%d_nlayers_%d_noutlayers_%d_outhidden_%s.pt' % \
        (args.agg_method, args.hidden_channels, args.num_layers, args.num_output_layers, ''.join(str(i) for i in args.out_hidden))
    print(ckpt_path)
    print( 'final_%s'%ckpt_path)

    # if os.path.exists(ckpt_path):
    #     msg = model.load_state_dict(torch.load(ckpt_path))
    #     print(msg)
    #     print('model loaded')
    model = train_model(model, dataloaders, optimizer_ft, exp_lr_scheduler, ckpt_path, num_epochs=args.n_epoch)


if __name__ == '__main__':
    # csv_file = 'nmr_smile_solvent_filtered_3dgnn.csv'
    # graph_path = './graph_3d/'
    # nmr_path = '/scratch0/yunruili/2dnmr_30k/nmr_1dcsv_30k/'  #'/Users/siriusxiao/Documents/Github/2DNMR_data/nmr_1dcsv_30k/'
    # dataset = graph_nmr_data(csv_file, graph_path, nmr_path)
    # data_loader = DataLoader(dataset, batch_size=10, shuffle= True)

    # model = ComENet(in_embed_size=3, out_channels=1001, agg_method='sum', num_layers=2, num_output_layers=2, hidden_channels=128, out_hidden=[256])
    # # msg = model.load_state_dict(torch.load('final_gnn3d_sum_hiddendim_512_nlayers_3_noutlayers_3_outhidden_512256.pt'))
    # # print(msg)
    # # model = model.cuda()

    # total_loss = 0
    # nan_graph = []
    # for graph, nmr, nmr_noise, filename in data_loader:
    #     print(graph.x.shape)
    #     # graph = graph.cuda()
    #     # nmr_noise = nmr_noise.cuda()
    #     loss, out_agg = model(graph, nmr_noise)
    #     if torch.isnan(loss):
    #         nan_graph.append(filename)
    #     else:
    #         print(loss)
    #         # total_loss += loss
    #     # nmr = nmr.detach().numpy()
    #     # plt.figure()
    #     # plt.plot(np.arange(1001), nmr[0])
    #     # plt.figure()
    #     # nmr2 = nmr2.detach().numpy()
    #     # plt.plot(np.arange(1001), nmr2[0])
    #     # print(out)
    #     # print(out.shape)
    #     # print(energy.shape)
    #     break
    # # print(nan_graph)

    args = argparse.ArgumentParser()
    args.add_argument('--batch_size', type=int, default=64, help='batch size')
    args.add_argument('--n_epoch', type=int, default=200, help='batch size')
    args.add_argument('--lr', type=float, default=0.0001, help='batch size')
    args.add_argument('--hidden_channels', type=int, default=256, help='hidden channel of gnn')
    args.add_argument('--num_layers', type=int, default=4, help='number of layers for GNN')
    args.add_argument('--num_output_layers', type=int, default=3, help='number of layers for GNN')
    args.add_argument('--agg_method', type=str, default='sum', help='aggregation method for GNN')
    args.add_argument('--out_hidden', default=[256, 512], type=int, nargs="+", help='hidden dims of projection')
    
    args = args.parse_args()
    main(args)
# %%
