#coding=utf-8
import pandas as pd
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

w2vDir = r'data/pretrained/DNA2Vec_dict.npy'
threeDir = r'data/pretrained/3mer.npy'
id_seq_dict = {}
DNA2Vec = np.load(w2vDir, allow_pickle=True).item()
threemerOH = np.load(threeDir, allow_pickle=True).item()

def k_mer_stride(dna, k, s):
    if len(dna) < 101:
        padnum = 101 - len(dna)
        if padnum % 2 != 0:
            paddingl = 'N' * (int(padnum / 2))
            paddingr = 'N' * (int(padnum / 2) + 1) 
            dna = paddingl + dna + paddingr
        else:
            padding = 'N' * (int(padnum / 2))
            dna = padding + dna + padding
    l = []
    dna_length = len(dna)
    j = 0
    for i in range(dna_length):
        t = dna[j:j + k]
        if (len(t)) == k:
            if 'N' in t:
                vec = np.array([0] * 100) 
                l.append(vec)
            else:
                vec = DNA2Vec[t]
                l.append(vec)
        j += s
    return l

def OH_mer_stride(dna, k, s):
    if len(dna) < 101:
        padnum = 101 - len(dna)
        if padnum % 2 != 0:
            paddingl = 'N' * (int(padnum / 2))
            paddingr = 'N' * (int(padnum / 2) + 1) 
            dna = paddingl + dna + paddingr
        else:
            padding = 'N' * (int(padnum / 2))
            dna = padding + dna + padding
    l = []
    dna_length = len(dna)
    j = 0
    for i in range(dna_length):
        t = dna[j:j + k]
        if (len(t)) == k:
            if 'N' in t:
                vec = np.zeros(64)
                l.append(vec)
            else:
                vec = threemerOH[t]
                l.append(vec)
        j += s
    return l
