import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import pickle
import numpy as np
import scipy

class graph_nmr_data(Dataset):
    def __init__(self, csv_file, graph_path, nmr_path):
        df = pd.read_csv(csv_file)
        self.file_list = df['File_name'].to_list()
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
        nmr_data = pd.read_csv(nmr_file)['13C']
        nmr_data_noise = self.add_noise(nmr_data, 2)
        nmr_data = torch.tensor(nmr_data).float()/100
        nmr_data_noise = torch.tensor(nmr_data_noise).float()/100
        return graph_data, nmr_data, nmr_data_noise
    def add_noise(self, nmr_data, std_dev):
        peak_locations = np.where(nmr_data > 0)[0]  # Adjust the condition as needed
        # Create a noise vector initialized to zero
        nmr_blur = np.zeros_like(nmr_data)

        # Apply Gaussian smoothing at each peak
        for peak in peak_locations:
            # Generate a Gaussian distribution centered at the peak
            gaussian = scipy.signal.gaussian(2 * std_dev * 3, std_dev)
            # gaussian = gaussian / gaussian.sum()  # Normalize the Gaussian
            gaussian *= nmr_data[peak]
            # Determine the range for applying the Gaussian
            start = max(peak - std_dev * 3, 0)
            end = min(peak + std_dev * 3, nmr_data.shape[0])

            # Apply Gaussian smoothing to the range
            for i in range(start, end):
                if nmr_blur[i] < gaussian[i - start]:
                    nmr_blur[i] = gaussian[i - start]
            # nmr_blur[start:end] += smoothed_range * gaussian[std_dev * 3 - (peak - start): std_dev * 3 + (end - peak)]

        # Make sure peak values remain unchanged
        nmr_blur[peak_locations] = nmr_data[peak_locations]
        return nmr_blur