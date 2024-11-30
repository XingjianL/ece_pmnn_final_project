from tools.data_loader import PMNNDataset, dynamic_collate_fn
from torch.utils.data import DataLoader
from functools import partial
import matplotlib.pyplot as plt
from model.sample_model import ODEFunc
import torch.optim as optim
import torch.nn as nn
import torch
from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm
training_dataset = PMNNDataset(time_gap=7, batch_window=-1)
training_loader = DataLoader(
    training_dataset, 
    batch_size=1, # Note: this batch size is the number of files, more batches will be added by the window size and time_gap
    shuffle=True, 
    collate_fn=partial(dynamic_collate_fn,window_size=training_dataset.batch_window),
    num_workers=4
    )
val_dataset = PMNNDataset(start_time=4, stop_time=5, time_gap=1)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=partial(dynamic_collate_fn, window_size=-1))

func = ODEFunc().cuda()
optimizer = optim.AdamW(func.parameters(), lr=1e-4)
criterion = nn.MSELoss()
losses = [0]*10
plt.figure(figsize=(8,2))
# Iterate through the data
for i in range(10):
    print(f"--it {i}: total batches: {len(training_loader)}")
    batch_n = 0
    for batch in tqdm(training_loader):
        t0, u0, x0, t,u,x = batch
        optimizer.zero_grad()
        #print(t0.shape, u0.shape, x0.shape, t.shape, u.shape, x.shape)
        y0 = torch.cat([u0,x0],1)
        y = torch.cat([u,x],1)
        pred = odeint(func,y0.cuda(),t.cuda(),method="rk4")
        #print(torch.squeeze(pred).shape, y.shape,torch.squeeze(pred)[-1],y[-1])
        loss = criterion(torch.squeeze(pred),y.cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(func.parameters(), max_norm=1.0)
        optimizer.step()
        
        if batch_n % 1000 == 0:
            #print(f"Saving model as model_{i}_{batch_n}.pth")
            torch.save(func.state_dict(), f"model_{i}.pth")
            tqdm.write(f"loss at {batch_n}: {loss.item()}. avg loss for epoch {i}: {losses[i]}")
        plt.clf()
        plt.subplot(141)
        plt.plot(t,torch.squeeze(pred.cpu().detach()))
        plt.title("pred")
        plt.subplot(142)
        plt.plot(t,y)
        plt.title("GT")
        plt.subplot(143)
        plt.plot(t,torch.squeeze(pred.cpu().detach())-y)
        plt.title("pred - GT")
        plt.subplot(144)
        plt.plot(losses)
        plt.title("losses")
        plt.show(block=False)
        plt.pause(0.1)
        batch_n += 1
        losses[i] += (loss.item() - losses[i])/batch_n
    print(losses)
    print(f"Saving model as model_{i}.pth")
    torch.save(func.state_dict(), f"model_{i}.pth")
print('val')
for batch in val_loader:
    t0, u0, x0, t,u,x = batch
    print(t0.shape, u0.shape, x0.shape, t.shape, u.shape, x.shape)
    break