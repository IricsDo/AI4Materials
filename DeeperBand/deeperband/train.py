#!/usr/bin/env python3

"""
Bonito training.
"""

import random,os,sys
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from deeperband.model import Net
import h5py
import torch
import numpy as np

__dir__ = os.path.dirname(os.path.realpath(__file__))
__data__ = os.path.join(__dir__, "dataset/")
__models__ = os.path.join(__dir__, "pretrain/")
def set_manual_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        print("the used gpu: ", torch.cuda.device_count(), torch.cuda.is_available())
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def load_dataset(h5pyfn: str, data_augment=3, batch_size=32):
    import torch.utils.data as Data
    fast5_data=h5py.File(h5pyfn, 'r')
    reads = np.random.permutation(np.array(sorted(fast5_data.items()), dtype=object))
    traindata=[]
    for read in reads:
        traindata.append([torch.tensor(read[1]['sc_bands'][:], dtype=torch.float),
                torch.log(torch.tensor(read[1].attrs['Tc']+1, dtype=torch.float))])
        if data_augment>0:
            traindata.append([torch.tensor(read[1]['sc_bands'][:], dtype=torch.float).flip(dims=[1]),
                    torch.log(torch.tensor(read[1].attrs['Tc']+1, dtype=torch.float))])
            traindata.append([torch.tensor(read[1]['sc_bands'][:], dtype=torch.float).flip(dims=[2]),
                    torch.log(torch.tensor(read[1].attrs['Tc']+1, dtype=torch.float))])
            traindata.append([torch.tensor(read[1]['sc_bands'][:], dtype=torch.float).flip(dims=[3]),
                    torch.log(torch.tensor(read[1].attrs['Tc']+1, dtype=torch.float))])
        if data_augment>1:
            traindata.append([torch.tensor(read[1]['sc_bands'][:], dtype=torch.float).permute(0, 2, 3, 1),
                    torch.log(torch.tensor(read[1].attrs['Tc']+1, dtype=torch.float))])
            traindata.append([torch.tensor(read[1]['sc_bands'][:], dtype=torch.float).permute(0, 3, 1, 2),
                    torch.log(torch.tensor(read[1].attrs['Tc']+1, dtype=torch.float))])
            traindata.append([torch.tensor(read[1]['sc_bands'][:], dtype=torch.float).permute(0, 1, 3, 2),
                    torch.log(torch.tensor(read[1].attrs['Tc']+1, dtype=torch.float))])
            traindata.append([torch.tensor(read[1]['sc_bands'][:], dtype=torch.float).permute(0, 2, 1, 3),
                    torch.log(torch.tensor(read[1].attrs['Tc']+1, dtype=torch.float))])
            traindata.append([torch.tensor(read[1]['sc_bands'][:], dtype=torch.float).permute(0, 3, 2, 1),
                    torch.log(torch.tensor(read[1].attrs['Tc']+1, dtype=torch.float))])
        if data_augment>2:    
            traindata.append([torch.tensor(read[1]['sc_bands'][:], dtype=torch.float).permute(0, 2, 3, 1).flip(dims=[1]),
                    torch.log(torch.tensor(read[1].attrs['Tc']+1, dtype=torch.float))])
            traindata.append([torch.tensor(read[1]['sc_bands'][:], dtype=torch.float).permute(0, 2, 3, 1).flip(dims=[2]),
                    torch.log(torch.tensor(read[1].attrs['Tc']+1, dtype=torch.float))])
            traindata.append([torch.tensor(read[1]['sc_bands'][:], dtype=torch.float).permute(0, 2, 3, 1).flip(dims=[3]),
                    torch.log(torch.tensor(read[1].attrs['Tc']+1, dtype=torch.float))])
            traindata.append([torch.tensor(read[1]['sc_bands'][:], dtype=torch.float).permute(0, 3, 1, 2).flip(dims=[1]),
                    torch.log(torch.tensor(read[1].attrs['Tc']+1, dtype=torch.float))])
            traindata.append([torch.tensor(read[1]['sc_bands'][:], dtype=torch.float).permute(0, 3, 1, 2).flip(dims=[2]),
                    torch.log(torch.tensor(read[1].attrs['Tc']+1, dtype=torch.float))])
            traindata.append([torch.tensor(read[1]['sc_bands'][:], dtype=torch.float).permute(0, 3, 1, 2).flip(dims=[3]),
                    torch.log(torch.tensor(read[1].attrs['Tc']+1, dtype=torch.float))])
            traindata.append([torch.tensor(read[1]['sc_bands'][:], dtype=torch.float).permute(0, 1, 3, 2).flip(dims=[1]),
                    torch.log(torch.tensor(read[1].attrs['Tc']+1, dtype=torch.float))])
            traindata.append([torch.tensor(read[1]['sc_bands'][:], dtype=torch.float).permute(0, 1, 3, 2).flip(dims=[2]),
                    torch.log(torch.tensor(read[1].attrs['Tc']+1, dtype=torch.float))])
            traindata.append([torch.tensor(read[1]['sc_bands'][:], dtype=torch.float).permute(0, 1, 3, 2).flip(dims=[3]),
                    torch.log(torch.tensor(read[1].attrs['Tc']+1, dtype=torch.float))])
            traindata.append([torch.tensor(read[1]['sc_bands'][:], dtype=torch.float).permute(0, 2, 1, 3).flip(dims=[1]),
                    torch.log(torch.tensor(read[1].attrs['Tc']+1, dtype=torch.float))])
            traindata.append([torch.tensor(read[1]['sc_bands'][:], dtype=torch.float).permute(0, 2, 1, 3).flip(dims=[2]),
                    torch.log(torch.tensor(read[1].attrs['Tc']+1, dtype=torch.float))])
            traindata.append([torch.tensor(read[1]['sc_bands'][:], dtype=torch.float).permute(0, 2, 1, 3).flip(dims=[3]),
                    torch.log(torch.tensor(read[1].attrs['Tc']+1, dtype=torch.float))])
            traindata.append([torch.tensor(read[1]['sc_bands'][:], dtype=torch.float).permute(0, 3, 2, 1).flip(dims=[1]),
                    torch.log(torch.tensor(read[1].attrs['Tc']+1, dtype=torch.float))])
            traindata.append([torch.tensor(read[1]['sc_bands'][:], dtype=torch.float).permute(0, 3, 2, 1).flip(dims=[2]),
                    torch.log(torch.tensor(read[1].attrs['Tc']+1, dtype=torch.float))])
            traindata.append([torch.tensor(read[1]['sc_bands'][:], dtype=torch.float).permute(0, 3, 2, 1).flip(dims=[3]),
                    torch.log(torch.tensor(read[1].attrs['Tc']+1, dtype=torch.float))])

    print("> completed reads: %s\n" % len(traindata))
    traindata, testdata = Data.random_split(traindata,
                            lengths=[int(0.8 * len(traindata)),
                            len(traindata) - int(0.8 * len(traindata))],
                            generator=torch.Generator().manual_seed(1))
    train_loader = torch.utils.data.DataLoader(traindata, batch_size=batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(testdata, batch_size=batch_size)
    return train_loader,val_loader

def train_model(model, train_loader, criterion, optimizer, device,num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0    
        for inputs, labels in train_loader:
            input, label = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input).squeeze()
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')
    return model

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            input, label = inputs.to(device), labels.to(device)
            outputs = model(input).squeeze()
            loss = criterion(outputs, label)
            val_loss += loss.item()
    print(f'Validation Loss: {val_loss / len(val_loader):.4f}')
    return val_loss / len(val_loader)

def main(args):

    workdir = os.path.expanduser(args.training_directory)
    os.makedirs(workdir, exist_ok=True)
    if os.path.exists(workdir) and not args.force:
        print("[error] %s exists, use -f to force continue training." % workdir)
        exit(1)
    set_manual_seed(args.seed)
    device = torch.device(args.device)
    model=Net()
    
    if args.pretrained:
        print("[using pretrained model {}]".format(args.pretrained))
        pretrained_path = args.pretrained
        if not os.path.exists(pretrained_path):
                pretrained_path = os.path.join(__models__, args.pretrained)
        
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint)
        print(f"[successfully loaded pretrained weights from {pretrained_path}]")
        
    model.to(device)
    dirname=args.dataset
    if not os.path.exists(dirname) and os.path.exists(os.path.join(__data__, dirname)):
        dirname = os.path.join(__data__, dirname)
    print("[loading data]",dirname)
    try:
        train_loader,val_loader=load_dataset(dirname, 
                        data_augment=args.augment,batch_size=args.batch)
    except FileNotFoundError:
        print('Please run deeperband download --training first to get the training set')

    import torch.nn as nn
    import torch.optim as optim
    lr = float(args.lr)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    args.epochs
    
    for i in range(args.epochs):
        model=train_model(model, train_loader, criterion, optimizer, device,num_epochs=args.epochs)
        evaluate_model(model, val_loader, criterion,device)
        PATH = workdir+'/train_'+str(i)+'.pt'
        torch.save(model.state_dict(), PATH)


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("training_directory")
    parser.add_argument('--pretrained')
    parser.add_argument("--dataset", default="0724.hdf5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", default='2e-3')
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch", default=64, type=int)
    parser.add_argument("--augment", default=0, type=int)
    parser.add_argument("-f", "--force", action="store_true", default=False)
    return parser
