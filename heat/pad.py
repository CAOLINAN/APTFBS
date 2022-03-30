#coding=utf-8

'''
比较经过pad的数据是否会产生验证结果的偏离
序列只能规定在80-202bps
85  <= dna <= 101bps的，在序列2端padding 'N'
101 < dna <= 186bps的，直接取以中心段的101bps序列
185 < dna <= 202bps的，以中心为准切成2段，分别对每段进行padding，预测结果取2段的最大值。
'''
import torch
import pandas as pd
import numpy as np
import models
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import load_data as ld
from sys import argv
import time



def slide(dna):
    '''
    按照原则对dna进行切片
    '''
    dnas = []
    if len(dna) <= 101:
        dnas.append(dna)
    elif 101 < len(dna) <= 186:
        dna1 = dna[int((len(dna)/2))-50 : int((len(dna)/2))+51]
        dnas.append(dna1)
    else:
        dna1 = dna[0:int((len(dna)/2))]
        dna2 = dna[int((len(dna)/2)):]
        dnas.append(dna1)
        dnas.append(dna2)
    return dnas

def heat(weight, name):
    #heatmap
    fig, ax = plt.subplots(figsize=(12,2))
    ax.set_title('DNA HEATMAP')
    ax.set_xlabel('Position of Feature Sequence')
    sns_plot = sns.heatmap(weight.detach().numpy(), cmap="RdBu", yticklabels=False , ax=ax)
    fig.savefig(str(name) + '.jpg')
    # time.sleep(20)


if __name__ == "__main__":
    model = models.arc()
    # dna = 'GTCCCAGACAGTGGCAGGTCAGCCAGCAGGTGGAGTGGGGTTATTGGAGGGGAGACCTGAATCCAGCCGACAGCTGCGAGCCGGGAGAGGTGGCTCC'
    # dna = 'TTCTGCATCACAAAACCTCCCAGAGCCAGGCTGGAGGTGGCTCCAGAACTCCTCCTTGTGGTTGGGGCAGGACCTGGCTGG'
    # dna = 'TTCTGCATCACAAAACCTCCCAGAGCCAGGCTGGAGGTGGCTCCAGAACTCCTCCTTGTGGTTGGGGCAGGACCTGGCTGGCCATTCCCAGTTCCTTCCTGAAGAATCCTGAAGAATCCTGAAGAATCCTGAAGATCCTGAAGAATCCTGAAGAATCCTGAAGAATCCTGAAGAATCCTGAAGAAGGGGG'
    # dna = 'CTCATGTCCTGATTCGGAAAGCTTTTTGCTCCTGCCTGTGGGTGACACCCAAGAGTTATAGCCTACTCATTCTAGTCAATCCATTAATAGCTGGAATTCAG'
    # dna = 'GCAGGAGAGCTTGGCTCTAGATGGTGCTGCTGTCTGTGGTAGCTAAAGCCCCTACAGGTGTTTGTGACTATTTGCCTGTGAAGTAACTTCATCAGCTGAAC'
    dna = 'AGACACAAGGCCAGATGGCCTGGTCTCTTGTTGGCCTGTCAAGGTCACCGCTCTGCACTGCCGGCAGCGGAACATGCTGCAAAGGGCCCCATCAACAGCCC'
    # dna = argv[1]

    dnas = slide(dna)

    result = 0
    weights = []
    for dna in dnas:
        index = 0
        flag = 0
        path = r'a549.pkl' 
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))   
        #encoding
        input_seq = torch.from_numpy(np.array(ld.k_mer_stride(dna, 3, 1)).T).type(torch.FloatTensor) #(99, 64)
        input_seq_OH = torch.from_numpy(np.array(ld.OH_mer_stride(dna, 3, 1))).type(torch.FloatTensor)  #(99, 64)
        # input_seq_OH = torch.from_numpy(ld.OH_mer_stride(dna, 3, 1)).type(torch.FloatTensor)  #(99, 64)

        input_seq = input_seq.unsqueeze(0)  #(1, 100, 99)
        input_seq = model.embedding(input_seq).transpose(1,2)
        input_seq = input_seq.squeeze(0)  #(99, 64)

        input_seq = torch.cat((input_seq, input_seq_OH),1).T
        input_seq = input_seq.unsqueeze(0)  #(1, 100, 99)

        #predict
        ans, weight = model(input_seq)
        weights.append(weight)
        if ans.item() > result :
            result = ans.item()
            flag = index
        index += 1
    lctime = time.strftime("p%Y-%m-%d-%H-%M-%S", time.localtime())
    heat(weights[flag], lctime)
    print(str(result)+lctime) 

