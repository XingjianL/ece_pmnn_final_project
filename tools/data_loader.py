import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class PMNNDataset(Dataset):
    def __init__(self, start_time = 0, stop_time = 4, time_gap = 2, batch_window = 50):
        self.all_sequences = self.get_all_sequences()
        self.start_time = start_time
        self.stop_time = stop_time
        self.batch_window = batch_window
        self.time_gap = time_gap # number of items to skip between timesteps in sequences
    def __len__(self):
        return len(self.all_sequences)
    
    def __getitem__(self, index):
        seq_file_name = self.all_sequences[index]
        seq_data = pd.read_csv(seq_file_name).values
        t = torch.tensor(seq_data[::self.time_gap,  -1],dtype=torch.float32)    # time
        valid_ind = (t >= self.start_time) & (t < self.stop_time)
        u = torch.tensor(seq_data[::self.time_gap,  :8],dtype=torch.float32)    # all motor inputs
        x = torch.tensor(seq_data[::self.time_gap,  8:-1],dtype=torch.float32)  # orientation stuff
        
        return u[valid_ind], x[valid_ind], t[valid_ind]
    

    def get_all_sequences(self):
        all_sequences = sorted(['./parsed_sequence_data/'+f for f in os.listdir('./parsed_sequence_data')])
        return all_sequences
    
# Parse window sizes of the signals
# output is (# of batches, window_size, # of features), time is 1 feature, u is 8 features, x is 13 features
def dynamic_collate_fn(batch, window_size):
    batch_t0 = torch.stack([t[0] for u,x,t in batch])
    batch_u0 = torch.stack([u[0] for u,x,t in batch])
    batch_x0 = torch.stack([x[0] for u,x,t in batch])
    # use all entries (note: only for batch_size = 1)
    if window_size < 0:
        assert len(batch) == 1 # Batch size must be 1 for stacking variable length signals, or specify a window_size
        batch_u = batch[0][0][1:]
        batch_x = batch[0][1][1:]
        batch_t = batch[0][2][1:]
        return [batch_t0, batch_u0, batch_x0, batch_t, batch_u, batch_x]
    # if window size is specified
    batch_t = torch.stack([t[1:window_size+1] for u,x,t in batch])
    batch_u = torch.stack([u[1:window_size+1] for u,x,t in batch])
    batch_x = torch.stack([x[1:window_size+1] for u,x,t in batch])
    for u,x,t in batch:
        max_t0 = len(t) - window_size - 1
        num_of_windows = len(t) // window_size - 1
        seq_t0 = np.random.choice(range(max_t0),num_of_windows,replace=False)
        batch_t0 = torch.cat([batch_t0, torch.stack([t[st0] for st0 in seq_t0])])
        batch_u0 = torch.cat([batch_u0, torch.stack([u[st0] for st0 in seq_t0])])
        batch_x0 = torch.cat([batch_x0, torch.stack([x[st0] for st0 in seq_t0])])
        batch_t = torch.cat([batch_t, torch.stack([t[st0+1:st0+window_size+1] for st0 in seq_t0])])
        batch_u = torch.cat([batch_u, torch.stack([u[st0+1:st0+window_size+1] for st0 in seq_t0])])
        batch_x = torch.cat([batch_x, torch.stack([x[st0+1:st0+window_size+1] for st0 in seq_t0])])
    return [batch_t0, batch_u0, batch_x0, batch_t, batch_u, batch_x]