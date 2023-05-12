from torch.utils.data import Dataset
import torch
import numpy as np
GBS_LABEL = ["APSK16","APSK32","APSK64","ASK8","BPSK","OQPSK","PSK16","PSK8","QAM128","QAM16","QAM256","QAM64","QPSK"]

class CustomGBSDataset(Dataset):
    def __init__(self, data, data_label, transform=None, target_transform=None):
        self.waveform_data = data
        self.waveform_label = data_label
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.waveform_data)

    def __getitem__(self, idx):
        i_waveform = torch.from_numpy(np.transpose(self.waveform_data[idx,:,:]))
        
        if self.transform:
            i_waveform = self.transform(i_waveform)
        if len(self.waveform_label)>0:
            label = torch.from_numpy(self.waveform_label[idx, :]).squeeze()
            if self.target_transform:
                label = self.target_transform(label)
            return i_waveform, label
        else:
            return i_waveform



def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(GBS_LABEL.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return GBS_LABEL[index]


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, label in batch:
        tensors += [waveform]
        targets += [label]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets

def create_dataloader(dataset, batch_size, train_or_test='train',collate_fn=collate_fn, device='cuda'):
    
    if device == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False
    if train_or_test=='train':
        shuffle = True
        drop_last= True
    else:
        shuffle = False
        drop_last= False
    
    generated_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return generated_dataloader