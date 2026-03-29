# make model

"""

Modified by Daniel Kaplan
"""
import os
print(os.getcwd())
from periodictable import elements
#import csv
import pandas as pd
import re
import numpy as np
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
#from torchvision import datasets, transforms
# from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset
#mport matplotlib.pyplot as plt

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {dev} device")
#torch.no_grad()
def is_nan(x):
    return (x != x)

exit
def create_template():
    num_elem = len(elements._element)+1
    value_list = [0.0]*num_elem
    res = np.array(value_list)
    res = np.expand_dims(res, axis=1)
    return res
# check if string contain int or float number
# or it is a true string

def string_or_number(s):
    try:
        z = int(s)
        return z
    except ValueError:
        try:
            z = float(s)
            return z
        except ValueError:
            return s

#dataset = []

def split_elename_and_value(row_data):
    """input name with shape into name and shape."""

    """
        MATCHING NUMBER:
        1. matches first digit ('\d')
         - \d matches all digits
        2. if present, matches decimal point ('\.?')
         - \. matches '.'
         - '?' means 'match 1-or-0 times'
        3. matches subsequent decimal digits ('\d*')
         - \d matches all digits
         - '*' means 'match >=0 times'

        MATCHING ELEMENT NAME:
        1. matches letters
         - [a-z A-Z] matches all letters
         - + matches 1-or-more times
    """
    name_pattern = r"\d\.?\d*|[a-z A-Z]+"
    splits = re.findall(name_pattern, row_data)

    # splits is to in format of pair of element name, value (all in strng)
    # the number of pairs varies
    #print(elements.symbol)
    # locate all symbol positions

    symbol_pos = []
    index = 0
    while index < len(splits):
        ele_sym = splits[index]
        #print(type(ele_sym))
        if type(string_or_number(ele_sym)) == str:
            try:
                ele = elements.symbol(ele_sym)
                symbol_pos.append(index)
            except ValueError as msg:
                print(str(msg))

        index = index+1

    element_id = []
    element_value = []
    total_value = 0
    for pos in symbol_pos:
        init_pos = pos
        ele = elements.symbol(splits[init_pos])
        element_id.append(ele.number)
        init_pos = init_pos+1
        if init_pos < len(splits):
            value = string_or_number(splits[init_pos])
            if isinstance(value, int) or isinstance(value, float):
                element_value.append(max(0, value))
                total_value = total_value + value
            else:
                element_value.append(0)
        else:
            element_value.append(0)

    if  total_value >= 1:
        element_value[:] = [x / total_value for x in element_value]

    total_value = min(1, max(0, sum(element_value)))

    if element_value.count(0) > 0:
        split_value = (1-total_value)/element_value.count(0)
        element_value = [split_value if item == -1 else item for item in element_value]


    return list(zip(element_id, element_value))


data = pd.read_csv(open('stable_materials_summary.csv', 'rb'))
f_struct = next(os.walk('.'))[1]
predicts_CNN=np.zeros((data.shape[0],len(f_struct)*2),dtype="float")
predicts_FC=np.zeros((data.shape[0],len(f_struct)*2),dtype="float")
jj=0
for index, row in data.iterrows():
    ### Pick element ### 
    #if(np.mod(jj,1000) == 0):
    print(jj,flush=True)
    if(jj > 2):
        exit()
    if(is_nan(row.Compound)==False):
        data_point = create_template()
        ret = split_elename_and_value(row.Compound)
    for item in ret:
        data_point[item[0]] = item[1]
    data_point = np.expand_dims(data_point, axis=0)   
    x_set=data_point.reshape((-1, 1, 10, 12))*100
    x_set = torch.tensor(x_set)
    mat_ds = TensorDataset(x_set,x_set)
    ### READY FOR INSERION INTO NETWORKS 
    for kk in enumerate(f_struct):
        ### CNN ### 
        res=next(os.walk(kk[1]+'/CNN/'))[1]
        net=torch.jit.load(kk[1]+'/CNN/'+res[0]+'/model_scripted.pt')
        net.eval()
        with torch.no_grad():
            for xb,yb in mat_ds:
                xb_g = xb.to(dev)
                pred = net(xb_g.float())
                pred_data = pred[0].to(dev)
                pred_data = pred_data.numpy().flatten()
                class_data = pred[1].to(dev)
                pred_result = 1
                if class_data[0][0] > class_data[0][1]:
                    pred_result = 0
                predicts_CNN[jj,2*kk[0]] = pred_data[0]*pred_result
                predicts_CNN[jj,2*kk[0]+1] = pred_result
        res=next(os.walk(kk[1]+'/FC/'))[1]
        net= torch.jit.load(kk[1]+'/FC/'+res[0]+'/model_scripted.pt')
        net.eval()
        with torch.no_grad():
            for xb,yb in mat_ds:
                xb_g = xb.to(dev)
                pred = net(xb_g.float())
                pred_data = pred[0].to(dev)
                pred_data = pred_data.numpy().flatten()
                class_data = pred[1].to(dev)
                pred_result = 1
                if class_data[0][0] > class_data[0][1]:
                    pred_result = 0
                predicts_FC[jj,2*kk[0]] = pred_data[0]*pred_result
                predicts_FC[jj,2*kk[0]+1] = pred_result
    jj=jj+1     
np.savetxt("CNN_predictions.csv", predicts_CNN)
np.savetxt("FC_predictions.csv", predicts_FC)               
