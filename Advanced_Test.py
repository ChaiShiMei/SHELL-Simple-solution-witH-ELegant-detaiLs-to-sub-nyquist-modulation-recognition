"""
Basic challenge: 1-signal modulation recognition
*********************************************



"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset
from torchmetrics import ConfusionMatrix 
import pandas as pd
from scipy.io import savemat
from copy import deepcopy
from utils.mix_aug import *
from utils.utils import mk_if_missing, seed_everything, count_parameters, get_likely_index, number_of_correct, split_weights, init_weights, WarmUpLR
# from utils.loss import OhemCrossEntropy
from dataset.GBS import CustomGBSDataset, create_dataloader
from models.M5 import M5
from models.pnn1 import Res1dNet31


import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb
import random 
import h5py
import argparse
import math


def parse_args():
    parser = argparse.ArgumentParser()
    '''Test'''
    parser.add_argument('--batch_size', type=int, default=256)   #batch size
    parser.add_argument('--testset_root_path', type=str, default="./data_2/data_2_test.h5") # root path to the unpublished test dataset
    parser.add_argument('--save_root_path', type=str, default='Pretrained_model')  # root path to the pretrained model
    args = parser.parse_args()
    return args

args = parse_args()

######################################################################
# create dataset and its loader for training and testing
def load_dataset(filename = args.testset_root_path):
    with h5py.File(filename, "r") as f:
        Waveform = f['X'][()]
    #normalization 
    max_norm, min_norm = 16382,-16384#get the maximum and minimum of the dataset
    Waveform = np.float32(((Waveform-min_norm)/(max_norm-min_norm)) * 2 -1)#normalize dataset to [-1,1]
    data_set= CustomGBSDataset(Waveform, [])
    # create loaders 
    data_loader = create_dataloader(data_set,args.batch_size,train_or_test='test', collate_fn=None)
    return data_loader
######################################################################

######################################################################

def test_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# use gpu if available otherwise use cpu
    print("GPU is available"  if torch.cuda.is_available() else "Only CPU is available")
    # Pretrained model for spectrum sensing
    model_subband = M5(n_input=16, n_output=24)# load the M5 
    model_subband.to(device)#move model to cuda
    model_subband.load_state_dict(torch.load(os.path.join(args.save_root_path,'best_checkpoint_ss.pt')))#load the pretrained model
    model_subband.eval()
    # Pretrained model for modulations detection

    Model = eval('Res1dNet31')#load the ResNet
    model_modu = Model(in_channel=16, classes_num=26)
    model_modu.to(device)#move model to gpu
    model_modu.load_state_dict(torch.load(os.path.join(args.save_root_path,'best_checkpoint_mr.pt')))#load the pretrained model
    model_modu.eval()
    print("Loading data")
    test_loader = load_dataset(filename = args.testset_root_path)#load the dataset
    #
    subbands_all = []# to store all the detected subbands
    modus_all = []# to store the detected modulations
    # with torch.no_grad():
    for data in test_loader:
        data = data.to(device)# move data to gpu
        output_subband = model_subband(data)# the output 1*24 vector determining the subbands 
        
        output_modu = model_modu(data)# output 1*26 voector determining the modulations
        modu1 = torch.argmax(output_modu[:,:13], dim=-1)# modulation of the 1st sub-band: index of the maximum element in the first 13 elements
        modu2 = torch.argmax(output_modu[:,13:], dim=-1)# modulation of the 2nd sub-band: index of the maximum element in the last 13 elements
        modus = torch.cat((modu1[:,None],modu2[:,None]),-1)# concatenate the detected 2 modulations together 
    
        subbands_all.append(output_subband.squeeze().cpu().detach().numpy())
        modus_all.append(modus.squeeze().cpu().detach().numpy())
    subbands_all = np.concatenate(subbands_all)
    modus_all = np.concatenate(modus_all)+1
    Detect = []
    for ii in range(len(modus_all)):
        subband_i = (subbands_all[ii,:]>-1)*1 #the element whose value > -1 means its corresponding sub-band is occupied
        if subband_i.sum()==1:#if 1-sub-band (channel conflict) signal, choose the first detected modulation
            Detect.append(subband_i*modus_all[ii,0])
        elif subband_i.sum()==2:# if if 2-sub-band signal, assign the two detected modulations to the corresponding subbands.
            idx = np.argmax(subband_i)
            Detect.append(np.concatenate((subband_i[:idx+1]*modus_all[ii,0],subband_i[idx+1:]*modus_all[ii,1])))
        elif subband_i.sum()==0:# If no sub-band is detected (all the elements<-1), we regard it as 1-sub-band signal and choose the index of largest element as the subband
            idx = np.argmax(subbands_all[ii,:])
            subband_i=np.zeros(24)
            subband_i[idx] = modus_all[ii,0]
            Detect.append(subband_i)
        elif subband_i.sum()>2:# If more than 2 subbands are detected, we choose the indices of the largest two elements
            idices = np.argsort(subbands_all[ii,:])[-2:]
            subband_i=np.zeros(24)
            subband_i[idices] = modus_all[ii,:]
            Detect.append(subband_i)
    savemat('AdvancedRecog.mat',{'Y':Detect})# save the detection to 'AdvancedRecog.mat'

def main():
    args = parse_args()
    test_analysis()
if __name__ == "__main__":
    main()
