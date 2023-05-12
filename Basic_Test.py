"""
Basic Challenge: 1-signal Modulation Recognition
*********************************************
This is the code for the evaluation on the unpublished test dataset. Tester just need to set '--testset_root_path' as the path to the test dataset. 
If the error  CUDA out of memory shows up, tester just need to reduce the '--batch_size'.

"""


import torch
import torch.nn.functional as F
from scipy.io import savemat
from utils.mix_aug import *
from dataset.GBS import CustomGBSDataset, create_dataloader
from models.pnn1 import Res1dNet311
import os
import numpy as np
import h5py
import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    '''Test'''
    parser.add_argument('--batch_size', type=int, default=256)  #batch size
    parser.add_argument('--testset_root_path', type=str, default="./data_1/data_1_test.h5") # root path to the unpublished test dataset
    parser.add_argument('--save_root_path', type=str, default='Pretrained_model')# root path to the pretrained model
    args = parser.parse_args()
    return args

args = parse_args()


######################################################################
# create dataset and its loader for training and testing
def load_dataset(filename = args.testset_root_path):
    #load the dataset
    with h5py.File(filename, "r") as f:
        Waveform = f['X'][()]
    #normalization 
    max_norm, min_norm = 8191, -9348# the maximum and minimum of the train dataset
    Waveform = np.float32(((Waveform-min_norm)/(max_norm-min_norm)) * 2 -1)#normalize dataset to [-1,1]
    # create dataset
    data_set= CustomGBSDataset(Waveform, [])
    # create loaders 
    data_loader = create_dataloader(data_set,args.batch_size,train_or_test='test', collate_fn=None)
    return data_loader
######################################################################

######################################################################

def test_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# use GPU if available otherwise use cpu
    # Pretrained model for modulations detection
    print("GPU is available"  if torch.cuda.is_available() else "Only CPU is available")
    Model = eval('Res1dNet311')#load the ResNet
    model_modu = Model(in_channel=16, classes_num=13)
    model_modu.to(device) # move the model to GPU
    model_modu.load_state_dict(torch.load(os.path.join(args.save_root_path,'best_checkpoint_basic.pt')))# load the best pretrained model
    model_modu.eval()
    print("Loading data")
    test_loader = load_dataset(filename = args.testset_root_path)# load the test dataset

    modus_all = []
    # with torch.no_grad():
    for data in test_loader:
        data = data.to(device)# move the data to the GPU
        output_modu = model_modu(data)
        modus = torch.argmax(output_modu, dim=-1) # the modulation is the index of the maximum element in the 1*13 output vector
        modus_all.append(modus.squeeze().cpu().detach().numpy())
    modus_all = np.concatenate(modus_all)+1#covert to 1-13
    savemat('BasicRecog.mat',{'Y':modus_all})#save to 'BasicRecog.mat'

def main():
    args = parse_args()
    test_analysis()
if __name__ == "__main__":
    main()
