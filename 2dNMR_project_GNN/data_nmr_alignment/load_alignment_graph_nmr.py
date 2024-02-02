import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import numpy as np
import scipy
from torch_geometric.data import Batch

class graph_nmr_data(Dataset):
    def __init__(self, csv_file, graph_path, nmr_path):
        df = pd.read_csv(csv_file)
        self.file_list = df['cnmr'].to_list()
        self.nmr_path = nmr_path
        self.graph_path = graph_path
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, item):
        filename = self.file_list[item].split('.')[0]
        graph_file = os.path.join(self.graph_path, filename + '.pickle')
        graph_data = pickle.load(open(graph_file, 'rb'))
        graph_data.x = graph_data.x.float()
        nmr_file = os.path.join(self.nmr_path, filename + '.csv')
        nmr_data = pd.read_csv(nmr_file)['ppm']
        nmr_data = torch.tensor(nmr_data).float() / 200.0
        return graph_data, nmr_data, filename


def custom_collate_fn(batch):
    # Separate graph data, NMR data, and filenames
    graphs, nmr_data_list, filenames = zip(*batch)

    # Use torch_geometric's Batch to handle graph data
    batched_graph = Batch.from_data_list(graphs)

    # Concatenate NMR data into a single tensor
    batched_nmr_data = torch.cat([data.unsqueeze(1) for data in nmr_data_list], dim=0)

    return batched_graph, batched_nmr_data, filenames


# nmr_path = '/scratch0/haox/yunruili/cnmr_alignment'
# graph_path = '/scratch0/yunruili/nmr_alignment/graph3d'
# csv_file = 'data_nmr_alignment/filtered_cnmr_smile_3d.csv'

# data = graph_nmr_data(csv_file, graph_path, nmr_path)
# dataset = DataLoader(data, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)
# for graph_data, nmr_data, filename in dataset:
#     print(graph_data.pos.shape)
#     print(graph_data.x.shape)
#     print(nmr_data.shape)
#     print(nmr_data)
#     break
