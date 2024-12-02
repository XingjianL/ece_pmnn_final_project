from tools.data_loader import PMNNDataset, dynamic_collate_fn
from torch.utils.data import DataLoader
from functools import partial
from model.relu_model import ODEFunc
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm
val_dataset = PMNNDataset(start_time=4, stop_time=5, time_gap=1)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=partial(dynamic_collate_fn, window_size=-1))
func = ODEFunc().cuda()
func.load_state_dict(torch.load("ReLU_model_3.pth", weights_only=True))
fig = plt.figure(figsize=(12,3))
batch_n = 0
avg_losses = [[]]*10

criterion = nn.MSELoss()
with torch.no_grad():
    for batch in tqdm(val_loader):
        batch_n += 1
        t0, u0, x0, t,u,x = batch
        #print(t-t0)
        #print(t0.shape, u0.shape, x0.shape, t.shape, u.shape, x.shape)
        y0 = torch.cat([u0,x0],1)
        y = torch.cat([u,x],1)
        pred = odeint(func,y0.cuda(),(t-t0).cuda(),method="rk4")
        loss = criterion(torch.squeeze(pred)[8:],y[8:].cuda())
        if len(avg_losses[0]) == 0:
            avg_losses[0].append(loss.item())
        else:
            avg_losses[0].append(avg_losses[0][-1]+(loss.item() - avg_losses[0][-1])/batch_n)
        #print(loss.item())
        
        if batch_n % 10 == 0:
            tqdm.write(f"batch {batch_n}: avg loss {avg_losses[0][-1]}")
            plt.clf()
            fig.suptitle(f"Epoch {0}. Batch {batch_n}")
            plt.subplot(141)
            plt.plot(t,torch.squeeze(pred.cpu().detach())[:,8:])
            plt.title("pred")
            plt.subplot(142)
            plt.plot(t,y[:,8:])
            plt.title("GT")
            plt.subplot(143)
            plt.plot(t,torch.squeeze(pred.cpu().detach())-y)
            plt.title("pred - GT")
            plt.subplot(144)
            plt.plot(torch.tensor(avg_losses).T)
            plt.title("avg losses")
            plt.ylim([0,max(.1,torch.tensor(avg_losses).min()*2)])
            plt.show(block=False)
            plt.pause(0.1)
print(avg_losses[0][-1])