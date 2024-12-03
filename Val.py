from tools.data_loader import PMNNDataset, dynamic_collate_fn
from torch.utils.data import DataLoader
from functools import partial
from model.tanh_model import ODEFunc
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm
val_dataset = PMNNDataset(start_time=4, stop_time=5, time_gap=10)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=partial(dynamic_collate_fn, window_size=-1))
func = ODEFunc().cuda()
func.load_state_dict(torch.load("tanh_model_4.pth", weights_only=True))
fig = plt.figure(figsize=(12,3))
batch_n = 0
avg_l2 = []
avg_loss = []
criterion = nn.MSELoss()
with torch.no_grad():
    for batch in tqdm(val_loader):
        batch_n += 1
        t0, u0, x0, t,u,x = batch
        #print(t-t0)
        #print(t0.shape, u0.shape, x0.shape, t.shape, u.shape, x.shape,torch.tensor([[0,0,0]]).shape)
        y0 = torch.cat([u0,torch.tensor([[0,0,0]]),x0[:,3:]],1)
        y = torch.cat([u,x[:,:3]-x0[:,:3],x[:,3:]],1)
        pred = odeint(func,y0.cuda(),(t-t0).cuda(),method="rk4")
        loss = criterion(torch.squeeze(pred.cpu().detach())[:,8:],y[:,8:])
        #print(torch.squeeze(pred.cpu().detach()).shape,y.shape)
        # if torch.isnan(loss):
        #     break
        L2 = torch.sqrt(torch.sum(torch.square(torch.squeeze(pred.cpu().detach())[:,8:11]-y[:,8:11]),dim=1))[-1]
        #print((torch.squeeze(pred)[8:11] - y[8:11].cuda()).square().sum(dim=0))
        avg_l2.append(L2)
        #print(loss.item())
        avg_loss.append(loss.item())
        if batch_n % 1 == 0:
            # print(torch.cat([torch.squeeze(pred.cpu().detach())[:,8:11]+ x0[:,:3], torch.squeeze(pred.cpu().detach())[:,11:]], 1).shape)
            # print(torch.cat([y[:,8:11] + x0[:,:3], y[:,11:]],1).shape)
            # print()
            #tqdm.write(f"batch {batch_n}: avg loss {avg_losses[0][-1]}")
            plt.clf()
            fig.suptitle(f"Epoch {1}. Batch {batch_n}")
            plt.subplot(141)
            plt.plot(t,torch.cat([torch.squeeze(pred.cpu().detach())[:,8:11]+ x0[:,:3], torch.squeeze(pred.cpu().detach())[:,11:]], 1))
            plt.title("pred")
            plt.subplot(142)
            plt.plot(t,torch.cat([y[:,8:11] + x0[:,:3], y[:,11:]],1))
            plt.title("GT")
            plt.subplot(143)
            plt.plot(t,torch.squeeze(pred.cpu().detach())-y)
            plt.title("pred - GT")
            plt.subplot(144)
            plt.hist(torch.tensor(avg_l2), bins=100)
            plt.title("L2 distance for position at 5 second")
            #plt.ylim([0,max(.1,torch.tensor(avg_losses).min()*2)])
            plt.show(block=False)
            plt.pause(0.01)
            tqdm.write(f"Mean L2: {torch.tensor(avg_l2).mean()}. Max L2: {torch.tensor(avg_l2).max()}. Min L2: {torch.tensor(avg_l2).min()}\n\
                       Mean Loss: {torch.tensor(avg_loss).mean()}")
plt.hist(torch.tensor(avg_l2), bins=100)
plt.title("L2 distance for position at 5 second")
            #plt.ylim([0,max(.1,torch.tensor(avg_losses).min()*2)])
plt.show()
print(f"Mean L2: {torch.tensor(avg_l2).mean()}. Max L2: {torch.tensor(avg_l2).max()}. Min L2: {torch.tensor(avg_l2).min()}")