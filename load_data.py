#coding=utf-8
import pandas as pd
import os
import numpy as np
# from dna2vec.multi_k_model import MultiKModel
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

workDir = './'

dataDir = workDir + 'data/rawdata/'

modelDir = workDir + 'model/'

onemerDir = workDir + 'data/pretrained/1mer.txt'
threemerDir = workDir + 'data/pretrained/3mer.txt'

w2vDir = workDir + 'data/pretrained/DNA2Vec_dict.npy'
oneDir = workDir + 'data/pretrained/1mer.npy'
threeDir = workDir + 'data/pretrained/3mer.npy'

plotDir = workDir + 'plot/'
id_seq_dict = {}
DNA2Vec = np.load(w2vDir, allow_pickle=True).item()
threemerOH = np.load(threeDir, allow_pickle=True).item()


def create_list(path):
    list = []
    for i, j, k in os.walk(path):
        for item in j:
            list.append(item)
    return list


def readfile(in_file, out_file, label):
    i = 0
    for line in in_file:
        if i % 2 == 0:
            seqID = line.strip("\n")
        elif i % 2 == 1:
            sequence = line.strip("\n")
            if not sequence.__contains__('N'):
                out_file.write(sequence + "\t" + str(label) + "\n")
                id_seq_dict[sequence] = seqID
        i = i + 1


def make_file(makeDir, name):
    positive_file = open(makeDir + name + 'positive.fasta', 'r')
    negative_file = open(makeDir + name + 'negative.fasta', 'r')
    out_file = open(makeDir + name + 'all_data.txt', 'w')
    out_file.write('sequence' + '\t' + 'label' + '\n')
    readfile(positive_file, out_file, 1)
    readfile(negative_file, out_file, 0)
    np.save(makeDir + name + 'id_seq_dict.npy', id_seq_dict)
    id_seq_dict.clear()


def make_DNA2Vec_dict(Dirname):
    DNA2Vec = {}
    all_data = pd.read_csv(Dirname, header=None, sep='\t')
    for i in range(0, all_data.shape[0]):
        line = all_data[0][i].split()
        DNA = line[0]
        vec = line[1:]
        a = np.array(vec)
        a[a == ''] = 0.0
        a = a.astype(np.double)
        DNA2Vec[DNA] = a
    np.save(Dirname, DNA2Vec)


def k_mer_stride(dna, k, s):
    l = []
    dna_length = len(dna)
    j = 0
    for i in range(dna_length):
        t = dna[j:j + k]
        if (len(t)) == k:
            vec = DNA2Vec[t]
            l.append(vec)
        j += s
    return l

def OH_mer_stride(dna, k, s):
    l = []
    dna_length = len(dna)
    j = 0
    for i in range(dna_length):
        t = dna[j:j + k]
        if (len(t)) == k:
            vec = threemerOH[t]
            l.append(vec)
        j += s
    return l


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)


def seq2bag(dna, window_size, stride):
    bag = ''
    dna_length = len(dna)
    for i in range(0, dna_length, stride):
        dna_instance = dna[i:i + window_size]
        bag += dna_instance
        if (i + window_size) == dna_length:
            break
    return bag

