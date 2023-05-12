"""
Basic Challenge: 1-signal Modulation Recognition
*********************************************
This is the code for training.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.mix_aug import *
from utils.utils import mk_if_missing, count_parameters, get_likely_index, number_of_correct, init_weights, WarmUpLR
from dataset.GBS import CustomGBSDataset, create_dataloader
from models.pnn1 import Res1dNet311
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    '''train'''
    parser.add_argument('--batch_size', type=int, default=1024) 
    parser.add_argument('--warm_up_epoch', type=int, default=5)# epoches for warmup start         
    parser.add_argument('--num_class', type=int, default=13)# num of modulations   
    parser.add_argument('--log_interval', type=int, default=40)# interval to print training stats    
    parser.add_argument('--n_epoch', type=int, default=200)  # num of training epoches
    parser.add_argument('--input_channel', type=int, default=16)  # channel of input data [I1,Q1,...,I8,Q8]
    parser.add_argument('--save_root_path', type=str, default='Pretrained_model')  #folder to save the trained model
    parser.add_argument('--root_path_train_set', type=str, default="./data_1/data_1_train.h5") #path to the train dataset
    parser.add_argument('--root_path_test_set', type=str, default="./data_1/data_1_test.h5")   # path to the test dataset
    args = parser.parse_args()
    return args
args = parse_args()
######################################################################
# create dataset and its loader for training and testing
def load_dataset(filename = args.root_path_train_set,train_or_test='test'):
    with h5py.File(filename, "r") as f:# open the h5 file
        Waveform = f['X'][()]
        Label = f['Y'][()] - 1
    #normalization 
    max_norm, min_norm = 8191, -9348# maximum and minimum of the train dataset
    Waveform = np.float32(((Waveform-min_norm)/(max_norm-min_norm)) * 2 -1)#normalize dataset to [-1,1]
    #create dataset
    data_set= CustomGBSDataset(Waveform, Label)
    # create loaders 
    data_loader = create_dataloader(data_set,args.batch_size,train_or_test=train_or_test)#If train_or_test='train',dataset will be shuffled. If 'test', no shuffle.
    return data_loader


######################################################################

######################################################################
def trainer():
    print('data loading')
    train_loader = load_dataset(args.root_path_train_set,train_or_test='train')#load the train dataset
    test_loader = load_dataset(args.root_path_test_set,train_or_test='test')#load the test dataset
    # model definition
    Model = eval('Res1dNet311')# load the ResNet
    model = Model(in_channel=args.input_channel, 
        classes_num=args.num_class)# define the model
    model = init_weights(model) #initial the weights of the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use GPU if available otherwise use cpu
    print("GPU is available"  if torch.cuda.is_available() else "Only CPU is available")
    model.to(device)# move the model to GPU
    print(model)
    n = count_parameters(model)
    print("Number of parameters: %s" % n)

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001) #ADAM 0.01
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch-args.warm_up_epoch)#CosineAnnealing scheduler
    warmup_scheduler = WarmUpLR(optimizer, args.warm_up_epoch)    #Warmup start scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()#Cross Entropy loss
    accuracy = [0]*args.n_epoch#list to store the accuracy on test set
    cur_iters = 0
    pbar_update = 1 / (len(train_loader) + len(test_loader))
    losses = []
    mk_if_missing(args.save_root_path) #create the save path folder if unavailable
    with tqdm(total=args.n_epoch) as pbar:
        for epoch in range(1, args.n_epoch + 1):
            ################training###################
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                cur_iters += 1
                data = data.to(device)#move data to GPU
                target = target.to(device)#move the target to GPU
                output = model(data)#output 1*13 vector
                loss = criterion(output.squeeze(), target)# calculate loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print training stats
                if batch_idx % args.log_interval == 0:
                    print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
                # update progress bar
                pbar.update(pbar_update)
                # record loss
                losses.append(loss.item())
            ################testing###################
            model.eval()
            correct = 0
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                pred = get_likely_index(output)# get the index of the maximum element as the modulation
                correct += number_of_correct(pred, target)# get the number of correctly detetcted modulations
                # update progress bar
                pbar.update(pbar_update)
            print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)\n")  
            accuracy[epoch-1] =  100. * correct / len(test_loader.dataset)# store the accuracy

            if accuracy[epoch-1] >= max(accuracy):#if current accuracy is best, save the trained model
                torch.save(model.state_dict(),os.path.join(args.save_root_path,'best_checkpoint_basic1.pt'))
            if epoch<args.warm_up_epoch:#first warm_up_epoch for warmup start
                warmup_scheduler.step()
            else:#after warmup, run CosineAnnealing
                scheduler.step()
        print(f'best accuracy is {max(accuracy)} at epoch {accuracy.index(max(accuracy))+1}')
        
def main():
    args = parse_args()
    trainer()
if __name__ == "__main__":
    main()