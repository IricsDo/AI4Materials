import torch
import numpy as np
from torch.utils.data import DataLoader
net= torch.jit.load('./model_scripted.pt')
val_ds=torch.load('./val_ds.pt')
val_loader = DataLoader(val_ds, 1, drop_last=True, shuffle=True, num_workers=0)
net.eval()

dev=torch.device("cpu")
with torch.no_grad():
    for xb, yb in val_loader:
        xb_g = xb.to(dev)
        yb_g_split = np.split(yb, [2], 1)
        yb_g_split_0 = (yb_g_split[0]).to(dev) #class
        yb_g_split_1 = (yb_g_split[1]).to(dev) #Tc
        pred = net(xb_g)
