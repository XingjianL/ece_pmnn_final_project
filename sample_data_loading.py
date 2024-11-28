from tools.data_loader import PMNNDataset, dynamic_collate_fn
from torch.utils.data import DataLoader
from functools import partial
import matplotlib.pyplot as plt

training_dataset = PMNNDataset(time_gap=2, batch_window=40)
training_loader = DataLoader(
    training_dataset, 
    batch_size=16, # Note: this batch size is the number of files, more batches will be added by the window size and time_gap
    shuffle=True, 
    collate_fn=partial(dynamic_collate_fn,window_size=training_dataset.batch_window)
    )
val_dataset = PMNNDataset(start_time=4, stop_time=5, time_gap=1)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=partial(dynamic_collate_fn, window_size=-1))

# Iterate through the data
for batch in training_loader:
    t0, u0, x0, t,u,x = batch
    print(t0.shape, u0.shape, x0.shape, t.shape, u.shape, x.shape)
    plt.subplot(121)
    plt.plot(t[20,:],u[10,:])
    plt.subplot(122)
    plt.plot(t[20,:],x[10,:])
    plt.show()
    break
print('val')
for batch in val_loader:
    t0, u0, x0, t,u,x = batch
    print(t0.shape, u0.shape, x0.shape, t.shape, u.shape, x.shape)
    break