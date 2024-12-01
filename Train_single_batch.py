from tools.data_loader import PMNNDataset, dynamic_collate_fn
from torch.utils.data import DataLoader
from functools import partial
import matplotlib.pyplot as plt
from model.relu_model import ODEFunc
import torch.optim as optim
import torch.nn as nn
import torch
from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm
training_dataset = PMNNDataset(time_gap=10, batch_window=-1)
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
func.load_state_dict(torch.load("relu_model_0.pth", weights_only=True))
optimizer = optim.AdamW(func.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
criterion = nn.MSELoss()
avg_losses = [[]]*10
fig = plt.figure(figsize=(8,2))
# Iterate through the data
for i in range(1,10):
    print(f"--it {i}: total batches: {len(training_loader)}")
    batch_n = 0
    for batch in tqdm(training_loader):
        t0, u0, x0, t,u,x = batch
        optimizer.zero_grad()
        #print(t0.shape, u0.shape, x0.shape, t.shape, u.shape, x.shape)
        y0 = torch.cat([u0,x0],1)
        y = torch.cat([u,x],1)
        pred = odeint(func,y0.cuda(),t.cuda(),method="rk4")
        loss = criterion(torch.squeeze(pred),y.cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(func.parameters(), max_norm=1.0)
        optimizer.step()
        batch_n += 1
        if len(avg_losses[i]) == 0:
            avg_losses[i].append(loss.item())
        else:
            avg_losses[i].append(avg_losses[i][-1]+(loss.item() - avg_losses[i][-1])/batch_n)
        if batch_n % 200 == 1:
            torch.save(func.state_dict(), f"relu_model_{i}.pth")
            tqdm.write(f"loss at {batch_n}: {loss.item()}. avg loss for epoch {i}: {avg_losses[i][-1]}")
        if batch_n % 1 == 0:
            plt.clf()
            fig.suptitle(f"Epoch {i}. Batch {batch_n}")
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
            plt.plot(torch.tensor(avg_losses).T)
            plt.title("losses")
            plt.show(block=False)
            plt.pause(0.1)
        
    scheduler.step()
    #print(avg_losses)
    print(f"Saving model as relu_model_{i}.pth")
    torch.save(func.state_dict(), f"relu_model_{i}.pth")
print('val')
for batch in val_loader:
    t0, u0, x0, t,u,x = batch
    print(t0.shape, u0.shape, x0.shape, t.shape, u.shape, x.shape)
    break