"""
Advanced ChallengeL: 2-signl Modulation Recognition
*********************************************
This is the training code for the first step: spectrum sensing.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.mix_aug import *
from utils.utils import mk_if_missing, count_parameters, WarmUpLR
from dataset.GBS import CustomGBSDataset, create_dataloader
from models.M5 import M5
import os
import numpy as np
from tqdm import tqdm
import h5py
import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    '''train'''
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_class', type=int, default=24)#24 sub-bands
    parser.add_argument('--log_interval', type=int, default=100) # interval to print training stats     
    parser.add_argument('--n_epoch', type=int, default=30)# num of training epoches
    parser.add_argument('--warm_up_epoch', type=int, default=5)# epoches for warmup start 
    parser.add_argument('--input_channel', type=int, default=16) # channel of input data [I1,Q1,...,I8,Q8]
    parser.add_argument('--save_root_path', type=str, default='Pretrained_model') #folder to save the trained model
    parser.add_argument('--root_path_train_set', type=str, default="./data_2/data_2_train.h5") #path to the train dataset
    parser.add_argument('--root_path_test_set', type=str, default="./data_2/data_2_test.h5")   # path to the test dataset
    args = parser.parse_args()
    return args
args = parse_args()

######################################################################
# create dataset and its loader for training and testing
def load_dataset(filename,train_or_test):
    with h5py.File(filename, "r") as f:# open the h5 file
        Waveform = f['X'][()]
        Label = f['Y'][()]
    #normalization 
    max_norm, min_norm = 16382,-16384# the maximum and minimum of the train dataset
    Waveform = np.float32(((Waveform-min_norm)/(max_norm-min_norm)) * 2 -1)#normalize dataset to [-1,1]
    #create the dataset
    data_set= CustomGBSDataset(Waveform, Label)
    # create loaders 
    data_loader = create_dataloader(data_set,args.batch_size,train_or_test=train_or_test)# If train_or_test='train',dataset will be shuffled. If 'test', no shuffle.
    return data_loader

######################################################################

######################################################################
def cal_accuracy( target, output):
    count = 0 # num of groundtruth subbands in the targets of test dataset
    detect = 0 # num of detected subbands
    TP = 0# true positive
    target = np.squeeze(np.concatenate(target, 0))
    output = np.squeeze(np.concatenate(output, 0))
    predidx=np.argsort(-output, axis=- 1)# get the indices after sorting the elements of each 1*24 output vector in descending order. As argsort can only sort in ascending order, so we use -ouput.

    for ii in range(len(target)):
        target_i = np.nonzero(target[ii,:])[0]# get the indices of nonzero elements of each target, target_i has is 1 or 2 elements.
        count += len(target_i)
        detect_i = min((output[ii, :] > -1).sum(), 2)#num of detected subbands in each output vector, the threshold is seting as -1. It cannot be lager than 2 as at most 2 in the target
        detect += detect_i
        predidx_i = predidx[ii, 0:detect_i]#indices of detected subbands
        if detect_i > 0:# if at least one element is >-1, there is at least one detected suband
            
            if len(target_i) == 2 and detect_i == 2:#if there are 2 subbands in both target and output,we need to compare all of them
                correct_i = float((predidx_i[0] == target_i[0]) + (predidx_i[0] == target_i[1])) + float(
                    (predidx_i[1] == target_i[0]) + (predidx_i[1] == target_i[1]))
            else:
                correct_i = float(predidx_i[0] == target_i[0])#otherwise we just need to  compare the first subband
        else:#if no element is greater than -1, we regard it as channel conflict case and choose the index of largest output element as the detected subband
            correct_i = float(predidx[ii, 0] == target_i[0])
        TP += correct_i
    precision = TP/detect
    recall = TP / count#recall is the accuracy 2 in the paper
    F1= 2*precision*recall/(precision+recall)
    return F1,precision,recall
def test_analysis():
    model = M5(n_input=args.input_channel, n_output=args.num_class)#load the M5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# use gpu if available otherwise use cpu
    model.to(device)#Move model to GPU
    model.load_state_dict(torch.load(os.path.join(args.save_root_path,'best_checkpoint_ss1.pt')))#load the pretrained model
    model.eval()


    test_loader = load_dataset(filename = args.root_path_test_set,train_or_test='test')#load the test dataset
    target_all = []
    output_all = []
    for data, target in test_loader:
        data = data.to(device)
        output = model(data)
        target_all.append(target.numpy())
        output_all.append(output.squeeze().cpu().detach().numpy())
    F1, precision, recall = cal_accuracy(target_all,output_all)
    print('F1:', F1, 'Precision:',  precision, 'Recall:', recall)


def trainer():
    print('data loading')
    train_loader = load_dataset(filename = args.root_path_train_set,train_or_test='train')# load the train dataset
    test_loader = load_dataset(filename = args.root_path_test_set,train_or_test='test')#load the test dataset
    print('data loaded')
    print('Nums of train and test samples:',len(train_loader.dataset), len(test_loader.dataset))
    # model definition
    model = M5(n_input=args.input_channel, n_output=args.num_class)#load M5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# use gpu if available otherwise use cpu
    print("GPU is available"  if torch.cuda.is_available() else "Only CPU is available")
    model.to(device)#move model to GPU
    print(model)
    n = count_parameters(model)# calculate the amount of parameters
    print("Number of parameters: %s" % n)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)# Adam optimizer
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch - args.warm_up_epoch)#cosine annealing learning rate scheduler
    warmup_scheduler = WarmUpLR(optimizer, args.warm_up_epoch)# warmup start scheduler
    criterion = nn.BCEWithLogitsLoss().cuda() #binary cross entropy loss with logits for multi-label classification
    label_smoothing = 0.025#label smoothing constant

    Fscore = [0]*args.n_epoch
    pbar_update = 1 / (len(train_loader) + len(test_loader))
    losses = []

    mk_if_missing(args.save_root_path) #create the save path folder if unavailable
    with tqdm(total=args.n_epoch) as pbar:
        for epoch in range(1, args.n_epoch + 1):
            ################training###################
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = torch.ne(target, 0).float()#Computes target not equal to 0 element-wise. The output is boolean tensor with the same shape with target
                target = target*(1-label_smoothing)+label_smoothing/args.num_class#label smoothing
                target = target.to(device)# move target to GPU
                
                output = model(data)
                loss = criterion(output.squeeze(), target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print training stats
                if batch_idx % args.log_interval == 0:
                    print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
                # record loss
                losses.append(loss.item())
            # update progress bar
            pbar.update(pbar_update)

            ################testing###################
            model.eval()
            target_all = []
            output_all = []
            for data, target in test_loader:
                
                data = data.to(device)
                output = model(data)
                target_all.append(target.numpy())
                output_all.append(output.squeeze().cpu().detach().numpy())
            # update progress bar
            pbar.update(pbar_update)
            F1, precision, recall = cal_accuracy(target_all, output_all)#calculate the F1 score, precision, and recall
            print('F1:', F1, 'Precision:', precision, 'Recall:', recall)
            Fscore[epoch-1] =  F1

            if Fscore[epoch-1] >= max(Fscore):# save the pretrained model for the best F1 score
                torch.save(model.state_dict(),os.path.join(args.save_root_path, 'best_checkpoint_ss1.pt'))
            if epoch < args.warm_up_epoch:#first warm_up_epoch for warmup start
                warmup_scheduler.step()
            else:
                scheduler.step()#after warmup, run CosineAnnealing
        print(f'best Fscore is {max(Fscore)} at epoch {Fscore.index(max(Fscore))+1}')

def main():
    args = parse_args()
    trainer()
    test_analysis()
if __name__ == "__main__":
    main()