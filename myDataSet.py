#coding=utf-8

from torch.utils.data import Dataset
import torch
import load_data as ld
import numpy as np


k = 3
stride = 1
print("kmer", k)
print("stride", stride)

#只包含原始输入
class MyDataSet(Dataset):
    def __init__(self, input, label, model):
        self.input_seq = input
        self.output = label
        self.model = model

    def __getitem__(self, index):
        input_seq_origin = self.input_seq[index]
        input_seq = np.array(ld.k_mer_stride(input_seq_origin, k, stride)).T    #(100, 99) 
        input_seq_OH = np.array(ld.OH_mer_stride(input_seq_origin, k, stride))  #(99, 64)

        input_seq = torch.from_numpy(input_seq).type(torch.FloatTensor).cuda()
        input_seq_OH = torch.from_numpy(input_seq_OH).type(torch.FloatTensor).cuda()

        input_seq = input_seq.unsqueeze(0)  #(1, 100, 99)
        input_seq = self.model.embedding(input_seq).transpose(1,2)  #(1, 99, 64)
        input_seq = input_seq.squeeze(0)  #(99, 64)

        input_seq = torch.cat((input_seq, input_seq_OH), 1) #(128,99)
        input_seq = input_seq.transpose(0,1)

        output_seq = self.output[index]
        output_seq = torch.Tensor([output_seq]).cuda()
        return input_seq, output_seq

    def __len__(self):
        return len(self.input_seq)

