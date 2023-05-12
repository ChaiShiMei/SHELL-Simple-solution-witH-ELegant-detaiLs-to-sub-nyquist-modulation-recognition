"""
Advanced ChallengeL: 2-signl Modulation Recognition
*********************************************
This is the training code for the second step: modulations recognition.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.utils import mk_if_missing, count_parameters, init_weights, WarmUpLR
from dataset.GBS import CustomGBSDataset, create_dataloader
from models.pnn1 import Res1dNet31
import os
import numpy as np
from tqdm import tqdm
import h5py
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    '''train'''      
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--warm_up_epoch', type=int, default=5)  # epoches for warmup start    
    parser.add_argument('--num_class', type=int, default=26)#26 modulations for two signals
    parser.add_argument('--log_interval', type=int, default=60) # interval to print training stats 
    parser.add_argument('--n_epoch', type=int, default=200)  # num of training epoches
    parser.add_argument('--input_channel', type=int, default=16) # channel of input data [I1,Q1,...,I8,Q8]
    parser.add_argument('--save_root_path', type=str, default='Pretrained_model')#folder to save the trained model
    parser.add_argument('--root_path_train_set', type=str, default="./data_2/data_2_train.h5") #path to the train dataset
    parser.add_argument('--root_path_test_set', type=str, default="./data_2/data_2_test.h5") # path to the test dataset

    args = parser.parse_args()
    return args

args = parse_args()

######################################################################
# encode the modulation of 2-signals to onehot
def double_one_hot(TrainLabel):
    TrainLabel_one_hot = []
    for ii in range(len(TrainLabel)):
        target_i = TrainLabel[ii, :]
        idx = target_i > 0#indices of nonzero elements
        modu = target_i[idx] - 1# get the value of nozero elements and convert to 0-12
        
        one_hot = np.zeros(26)#modulation for 2 signals
        if len(modu) == 1:
            one_hot[modu[0]] = 1#If channel conflict case, there is only 1 signal.
        else:#if 2 signal
            one_hot[modu[0]] = 1#encode the modulation of 1st signal in the first 13 elements
            one_hot[modu[1] + 13] = 1#encode the modulation of 2nd signal in the last 13 elements
        TrainLabel_one_hot.append(one_hot)
    return np.stack(TrainLabel_one_hot, axis=0)
# create dataset and its loader for training and testing
def load_dataset(filename,train_or_test):
    with h5py.File(filename, "r") as f:#open the h5 file
        Waveform = f['X'][()]
        Label = f['Y'][()]
    if train_or_test=='train':#encode the target to onehot for the train dataset. For the test dataset, no need.
        Label = double_one_hot(Label)
    # normalization
    max_norm, min_norm = 16382,-16384# the maximum and minimum of the train dataset
    Waveform = np.float32(((Waveform - min_norm) / (max_norm - min_norm)) * 2 - 1)#normalize dataset to [-1,1]
    data_set = CustomGBSDataset(Waveform, Label)
    # create loaders 
    data_loader = create_dataloader(data_set, args.batch_size, train_or_test=train_or_test)# If train_or_test='train',dataset will be shuffled. If 'test', no shuffle.
    return data_loader


######################################################################

######################################################################
#Accuracy 2 calculation
def cal_accuracy( target, output):
    count = 0 # num of groundtruth subbands in the targets of test dataset
    TP = 0#num of true positive
    target = np.squeeze(np.concatenate(target, 0))
    output = np.squeeze(np.concatenate(output, 0))

    modu1 = np.argmax(output[:,0:13], axis=- 1)# indices of the maximum of the first 13 elements in each output vector
    modu2 = np.argmax(output[:, 13:], axis=- 1)# indices of the maximum of the last 13 elements in each output vector
    predidx = np.stack((modu1,modu2),axis=1)

    for ii in range(len(target)):
        target_modu = target[ii, :]
        idx = target_modu > 0
        target_i = target_modu[idx] - 1#get target modulations and convert to [0 12]
        count += len(target_i)
        predidx_i = predidx[ii, :]
        if len(target_i) == 1:#if channel confilct case, just compare the 1st detected modualtion
            correct_i = float(predidx_i[0] == target_i[0])
        else:# if 2-signal case, compare both
            correct_i = float(predidx_i[0] == target_i[0]) + float(predidx_i[1] == target_i[1])
        TP += correct_i
    accuracy = TP / count# accuracy 2
    return accuracy
def test_analysis():
    # model  definition
    Model = eval('Res1dNet31')#load the ResNet
    model = Model(in_channel=args.input_channel,
                  classes_num=args.num_class)#define the ResNEt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# use gpu if available otherwise use cpu
    model.to(device)#move model to gpu
    model.load_state_dict(torch.load(os.path.join(args.save_root_path,'best_checkpoint_mr.pt')))#load the pretrained model
    test_loader = load_dataset(filename = args.root_path_test_set,train_or_test='test')#load the test dataset
    model.eval()
    target_all = []# store all the targets
    output_all = []# store all the outputs
    # with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)#move data to gpu
        output = model(data)
        target_all.append(target.numpy())
        output_all.append(output.squeeze().cpu().detach().numpy())
    accuracy = cal_accuracy(target_all, output_all)# calculate accuracy 2
    print('Accuracy:', accuracy)



def trainer():
    print('data loading')
    train_loader = load_dataset(filename = args.root_path_train_set,train_or_test='train')# load the train dataset
    test_loader = load_dataset(filename = args.root_path_test_set,train_or_test='test')#load the test dataset
    print('data loaded')
    print('Nums of train and test samples:',len(train_loader.dataset), len(test_loader.dataset))
    # model definition  
    Model = eval('Res1dNet31')#load ResNet
    model = Model(in_channel=args.input_channel, 
        classes_num=args.num_class)#define the ResNet

    model = init_weights(model) #initial the weights of ResNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# use gpu if available otherwise use cpu
    print("GPU is available"  if torch.cuda.is_available() else "Only CPU is available")
    model.to(device)#move model to gpu
    print(model)
    n = count_parameters(model)
    print("Number of parameters: %s" % n)


    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001) #ADAM optimizer
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch-args.warm_up_epoch)#cosine annealing learning rate scheduler
    warmup_scheduler = WarmUpLR(optimizer, args.warm_up_epoch)# warmup start scheduler
    criterion = nn.BCEWithLogitsLoss().cuda()#binary cross entropy loss with logits for multi-label classification
    label_smoothing = 0.026#label smoothing constant
    Accu = [0]*args.n_epoch#record accuracies on test dataset for every epoch
    cur_iters = 0
    pbar_update = 1 / (len(train_loader) + len(test_loader))
    losses = []
    mk_if_missing(args.save_root_path) #to save checkpoints
    with tqdm(total=args.n_epoch) as pbar:
        for epoch in range(1, args.n_epoch + 1):
            ################training###################
            model.train()

            for batch_idx, (data, target) in enumerate(train_loader):
                cur_iters += 1
                data = data.to(device)#move data to gpu
                target = target * (1 - label_smoothing) + label_smoothing / args.num_class#label smoothing
                target = target.to(device)#move target tp gpu
                output = model(data)
                loss = criterion(output.squeeze(), target)#calculate the loss

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
                data = data.to(device)#move data to gpu
                output = model(data)
                target_all.append(target.numpy())
                output_all.append(output.squeeze().cpu().detach().numpy())
            # update progress bar
            pbar.update(pbar_update)
            accuracy = cal_accuracy(target_all, output_all)#calculate the accuracy 2
            print("Train Epoch:", epoch, 'Accuracy:', accuracy)
            Accu[epoch - 1] = accuracy

            if Accu[epoch - 1] >= max(Accu):# save the pretrained model for the best accuracy
                torch.save(model.state_dict(), os.path.join(args.save_root_path,'best_checkpoint_mr1.pt'))
            if epoch < args.warm_up_epoch:#first warm_up_epoch for warmup start
                warmup_scheduler.step()
                print(warmup_scheduler.get_lr())#print the current learning rate
            else:
                scheduler.step()#after warmup, run CosineAnnealin
                print(scheduler.get_last_lr()[0])#print the current learning rate
        print(f'best accuracy is {max(Accu)} at epoch {Accu.index(max(Accu)) + 1}')

def main():
    args = parse_args()
    trainer()
    test_analysis()
if __name__ == "__main__":
    main()
