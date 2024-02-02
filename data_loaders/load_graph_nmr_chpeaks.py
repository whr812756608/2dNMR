import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import numpy as np
import scipy
from torch_geometric.data import Batch

class graph_nmr_data_2d_peak(Dataset):
    '''
    Excludes Test_*.csv data
    Returns 
        graph data as usual
        c_peaks of shape [N, 1]
        h_peaks of shape [N, 2]
        filename as usual
    '''
    def __init__(self, csv_file, graph_path, nmr_path):
        df = pd.read_csv(csv_file)
        self.file_list = df['File_name'].to_list()
        # filter out Test_*.csv
        self.file_list = [x for x in self.file_list if not x.startswith('Test_')]

        self.nmr_path = nmr_path
        self.graph_path = graph_path
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, item):
        filename = self.file_list[item].split('.')[0]

        graph_file = os.path.join(self.graph_path, filename + '.pickle')
        graph_data = pickle.load(open(graph_file, 'rb'))
        graph_data.x = graph_data.x.float()

        # added feature to use ComENet for multitask
        graph_data.has_c = True
        graph_data.has_h = True

        # use processed file to load c and h peaks
        cnmr = os.path.join(self.nmr_path, filename + '_c.pickle')
        cnmr_data = pickle.load(open(cnmr, 'rb'))
        hnmr = os.path.join(self.nmr_path, filename + '_h.pickle')
        hnmr_data = pickle.load(open(hnmr, 'rb'))

        c_peaks = torch.tensor(cnmr_data).view(-1, 1)
        h_peaks = torch.tensor(hnmr_data)

        return graph_data, c_peaks, h_peaks, filename
    
def custom_collate_fn(batch):
    # Separate graph data, NMR data, and filenames
    graphs, c_peaks, h_peaks, filenames = zip(*batch)

    # Use torch_geometric's Batch to handle graph data
    batched_graph = Batch.from_data_list(graphs)

    # Concatenate NMR data into a single tensor
    batched_cnmr_data = torch.cat([data for data in c_peaks], dim=0)
    batched_hnmr_data = torch.cat([data for data in h_peaks], dim=0)

    return batched_graph, batched_cnmr_data, batched_hnmr_data, filenames


# nmr_path = '/scratch0/yunruili/2dnmr_30k/nmr_2dcsv_expanded/'
# graph_path = '/scratch0/yunruili/2dnmr_30k/graph_3d/'
# csv_file = 'nmr_smile_solvent_filtered2_3dgnn.csv'

# data = graph_nmr_data_2d_peak(csv_file, graph_path, nmr_path)
# dataset = DataLoader(data, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)
# for graph_data, cnmr_data, hnmr_data, filename in dataset:
#     print(graph_data.pos.shape)
#     print(graph_data.x.shape)
#     print(cnmr_data.shape)
#     print(hnmr_data.shape)
#     print(cnmr_data)
#     break