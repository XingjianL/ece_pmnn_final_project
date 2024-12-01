from tools.data_loader import PMNNDataset, dynamic_collate_fn
from torch.utils.data import DataLoader
from functools import partial
import matplotlib.pyplot as plt
from model.tanh_model import ODEFunc
import torch.optim as optim
import torch.nn as nn
import torch
from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm

training_dataset = PMNNDataset(time_gap=3, batch_window=40)
training_loader = DataLoader(
    training_dataset, 
    batch_size=4, # Note: this batch size is the number of files, more batches will be added by the window size and time_gap
    shuffle=True, 
    collate_fn=partial(dynamic_collate_fn,window_size=training_dataset.batch_window),
    num_workers=4
    )
val_dataset = PMNNDataset(start_time=4, stop_time=5, time_gap=1)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=partial(dynamic_collate_fn, window_size=-1))

func = ODEFunc().cuda()
optimizer = optim.AdamW(func.parameters(), lr=1e-3)
criterion = nn.MSELoss()
losses = []
# Iterate through the data
for i in range(10):
    print(f"--it {i}: total batches: {len(training_loader)}")
    for batch in tqdm(training_loader):
        t0, u0, x0, t,u,x = batch
        #print(t0.shape, u0.shape, x0.shape, t.shape, u.shape, x.shape)
        y0 = torch.cat([u0,x0],1)
        y = torch.cat([u,x],2)
        preds = []
        for j in range(t0.shape[0]):
            pred = odeint(func,y0[j].cuda(),torch.squeeze(t[j]-t0[j]).cuda(),method="rk4")
            preds.append(pred)
        loss = criterion(torch.stack(preds),torch.squeeze(y).cuda())
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(func.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
    plt.plot(losses)
    plt.show()
print('val')
for batch in val_loader:
    t0, u0, x0, t,u,x = batch
    print(t0.shape, u0.shape, x0.shape, t.shape, u.shape, x.shape)
    break