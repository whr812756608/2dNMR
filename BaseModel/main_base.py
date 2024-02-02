# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from load_fpt_nmr import fpt_nmr_data
from BaseModel import NMRNetwork_Base
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import time
import matplotlib.pyplot as plt
import numpy as np


def train_model(model, dataloaders, optimizer, scheduler, checkpoint_path, num_epochs=100):
    best_loss = 1e10
    scaler = GradScaler()

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
                fpt, nmr, nmr_noise = batch
                fpt = fpt.cuda()
                nmr = nmr.cuda()
                nmr_noise = nmr_noise.cuda()

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # print(next(model.parameters()).is_cuda)
                    # print(smile.get_device())
                    # print(nmr_seq.get_device())
                    with autocast():
                        loss, out = model(fpt, nmr_noise)
                        # print(loss)
                        loss *= 100
                        epoch_loss += loss
                    print(loss)
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        # loss.backward()
                        # optimizer.step()

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

    # load best model weights
    model.load_state_dict(torch.load(checkpoint_path))
    return model

def main():
    # Example usage
    input_size = 2048      # Size of the input layer (number of bits in the fingerprint)
    output_size = 1001      # Desired size of the output representation 'd'
    initial_hidden_size = 512 # Size of the first hidden layer
    n_layers = 2           # Number of hidden layers
    batch_size = 2

    csv_file = 'nmr_smile_solvent_web_sat_combined2.csv'
    fpt_path = './fingerprint/'
    nmr_path = '/Users/siriusxiao/Documents/GitHub/2DNMR_data/data_websat/nmr_1dcsv_30k/'
    dataset = fpt_nmr_data(csv_file, fpt_path, nmr_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    # Create an instance of the network
    model = NMRNetwork_Base(input_size, output_size, initial_hidden_size, n_layers)
    print(model)
    count_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model parameter: %d" % count_param)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name}, number of params: {param.numel()}")

    # Fine-tuning setup
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=8, gamma=0.9)

    ckpt_path = 'basemodel_nlayer%d_hdim%d.pt' % (n_layers, initial_hidden_size)
    print(ckpt_path)
    model = train_model(model, dataloaders, optimizer_ft, exp_lr_scheduler, ckpt_path, num_epochs=100)


if __name__ == '__main__':
    # csv_file = 'nmr_smile_solvent_web_sat_combined2.csv'
    # fpt_path = './fingerprint/'
    # nmr_path = '/Users/siriusxiao/Documents/GitHub/2DNMR_data/data_websat/nmr_1dcsv_30k/'
    # dataset = fpt_nmr_data(csv_file, fpt_path, nmr_path)
    # data_loader = DataLoader(dataset, batch_size=1, shuffle= True)

    # for fpt, nmr, nmr2 in data_loader:
    #     nmr = nmr.detach().numpy()
    #     plt.figure()
    #     plt.plot(np.arange(1001), nmr[0])
    #     plt.figure()
    #     nmr2 = nmr2.detach().numpy()
    #     plt.plot(np.arange(1001), nmr2[0])
    #     break
    main()
# %%
