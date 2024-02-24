import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import numpy as np
import scipy
from torch_geometric.data import Batch

class graph_nmr_alignment_data(Dataset):
    '''
    type: c- cnmr only; b - both cnmr and hnmr; h - hnmr only (not used)'''
    def __init__(self, csv_file, graph_path, nmr_path, type = 'c'):
        df = pd.read_csv(csv_file)
        # df['file'] = df['file'].astype(str).str.zfill(9)
        df['file'] = df['id'].astype(str)
        self.file_list = df['file'].to_list()
        self.nmr_path = nmr_path
        self.graph_path = graph_path
        self.type = type
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, item):
        filename = self.file_list[item]
        graph_file = os.path.join(self.graph_path, filename + '.pickle')
        graph_data = pickle.load(open(graph_file, 'rb'))
        graph_data.x = graph_data.x.float()
        if self.type=='c':
            hnmr_data = None
            cnmr_file = os.path.join(self.nmr_path, 'cnmr_alignment_csv_mpnn', filename + '.csv')
            cnmr_data = pd.read_csv(cnmr_file)['ppm']
            cnmr_data = torch.tensor(cnmr_data).float() / 200.0
            graph_data.has_c = True
            graph_data.has_h = False
            graph_data.solvent_class = 8 # 8 is the "unknown" solvent class
        elif self.type=='h':
            cnmr_data = None
            hnmr_file = os.path.join(self.nmr_path, 'hnmr_alignment_csv_mpnn', filename + '.csv.csv')
            hnmr_data = pd.read_csv(hnmr_file)['ppm']
            hnmr_data = torch.tensor(hnmr_data).float() / 10.0
            graph_data.has_c = False
            graph_data.has_h = True
            graph_data.solvent_class = 8 # 8 is the "unknown" solvent class
        else:
            hnmr_file = os.path.join(self.nmr_path, 'hnmr_alignment_csv_mpnn', filename + '.csv.csv')
            cnmr_file = os.path.join(self.nmr_path, 'cnmr_alignment_csv_mpnn', filename + '.csv')
            hnmr_data = pd.read_csv(hnmr_file)['ppm']
            cnmr_data = pd.read_csv(cnmr_file)['ppm']
            cnmr_data = torch.tensor(cnmr_data).float() / 200.0
            hnmr_data = torch.tensor(hnmr_data).float() / 10.0
            nmr_data = [cnmr_data, hnmr_data]
            graph_data.has_c = True
            graph_data.has_h = True
            graph_data.solvent_class = 8 # 8 is the "unknown" solvent class
        return graph_data, cnmr_data, hnmr_data, filename


def custom_collate_fn(batch):
    # Separate graph data, NMR data, and filenames
    graphs, cnmr_data, hnmr_data, filenames = zip(*batch)

    # Use torch_geometric's Batch to handle graph data
    batched_graph = Batch.from_data_list(graphs)

    # Concatenate NMR data into a single tensor
    if graphs[0].has_h:
        batched_hnmr_data = torch.cat([data.unsqueeze(1) for data in hnmr_data], dim=0)
        if graphs[0].has_c:
            batched_cnmr_data = torch.cat([data.unsqueeze(1) for data in cnmr_data], dim=0)
        else:
            batched_cnmr_data = None
    else:
        batched_cnmr_data = torch.cat([data.unsqueeze(1) for data in cnmr_data], dim=0)
        batched_hnmr_data = None

    return batched_graph, batched_cnmr_data, batched_hnmr_data, filenames

class CustomBatchSampler:
    def __init__(self, dataloader_a, dataloader_b, dataloader_c, n1, n2):
        self.dataloader_a = dataloader_a
        self.dataloader_b = dataloader_b
        self.dataloader_c = dataloader_c
        self.n1 = n1
        self.n2 = n2

    def __iter__(self):
        self.iter_a = iter(self.dataloader_a)
        self.iter_b = iter(self.dataloader_b)
        self.iter_c = iter(self.dataloader_c)
        self.current_count = -1
        return self

    def __next__(self):
        if self.current_count % self.n1 == 0:
            try:
                # Fetch batch from dataloader_b
                # print('fetch from b')
                batch = next(self.iter_b)
            except StopIteration:
                # Reset dataloader_b if exhausted
                # print('RE- fetch from b')
                self.iter_b = iter(self.dataloader_b)
                batch = next(self.iter_b)
            self.current_count += 1
            return batch
        
        if self.current_count % self.n2 == 0:
            try:
                # Fetch batch from dataloader_c
                # print('fetch from c')
                batch = next(self.iter_c)
            except StopIteration:
                # Reset dataloader_c if exhausted
                # print('RE- fetch from c')
                self.iter_c = iter(self.dataloader_c)
                batch = next(self.iter_c)
            self.current_count += 1
            return batch

        # Fetch batch from dataloader_a
        try:
            batch = next(self.iter_a)
            self.current_count += 1
        except StopIteration:
            raise StopIteration
        
        return batch

# graph_path = '/scratch0/haox/2DNMR_prediction_gt/Datasets/graph3d/'
# nmr_path = '/scratch0/haox/yunruili/'
# # cnmr_path = '/scratch0/haox/yunruili/cnmr_alignment'
# # hnmr_path = '/scratch0/haox/yunruili/hnmr_alignment'
# csv_cnmr = 'cnmr_smile_dataset_27k.csv'
# # csv_hnmr = 'hnmr_smile_dataset_144.csv'
# csv_common = 'common_smile_dataset_5k.csv'

# data = graph_nmr_alignment_data(csv_common, graph_path, nmr_path, type='both')
# dataset = DataLoader(data, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)
# for graph_data, cnmr_data, hnmr_data, filename in dataset:
#     print(graph_data.pos.shape)
#     print(graph_data.x.shape)
#     print(graph_data.has_c)
#     print(graph_data.has_h)
#     print(cnmr_data.shape)
#     print(cnmr_data)
#     print(hnmr_data)
#     break
