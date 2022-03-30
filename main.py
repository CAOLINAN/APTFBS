#coding=utf-8
import datetime
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import myDataSet as ms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import load_data as ld
from models import arc
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, recall_score, f1_score, \
    precision_recall_curve
from sklearn.model_selection import StratifiedKFold, KFold
import util


def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 

def train(myDataLoader, path, fold):
    best = 0
    train_file = open('train_files'+str(fold)+'.txt','a')
    train_file.write("train_epoch" + '\t' + 'loss' + '\n')
    for epoch in range(Epoch):
        for step, (x, y) in enumerate(myDataLoader):
            model.train()
            output = model(x)
            loss = loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_file.write(str(epoch) + '\t' + str(loss.item()) + '\n')
        ROC, PR, F1, test_loss, accuracy = validate(validate_DataLoader, epoch)
        if ROC > best:
            best = ROC
            model_path = ld.modelDir + str(path)
            model_name = ld.modelDir + path + '/validate_params_' + str(fold) + '.pkl'
            mkdir(model_path)
            torch.save(model.state_dict(), model_name)
    scheduler.step(test_loss)
    train_file.close()
    print(model_name)
    return model_name


def validate(myDataLoader, epoch):
    output_list = []
    output_result_list = []
    correct_list = []
    test_loss = 0
    for step, (x, y) in enumerate(myDataLoader):
        model.eval()
        output = model(x)
        loss = loss_func(output, y)
        test_loss += float(loss)
        output_list += output.cpu().detach().numpy().tolist()
        output = (output > 0.5).int()
        output_result_list += output.cpu().detach().numpy().tolist()
        correct_list += y.cpu().detach().numpy().tolist()
    y_pred = np.array(output_result_list)
    y_true = np.array(correct_list)
    accuracy = accuracy_score(y_true, y_pred)
    test_loss /= myDataLoader.__len__()
    print('Validate set: Average loss:{:.4f}\tAccuracy:{:.3f}'.format(test_loss, accuracy))
    ROC, PR, F1 = util.get_ROC_Curve(output_list, output_result_list, correct_list)
    print('第{}折_第{}轮_ROC:{}\tPR:{}\tF1:{} '.format(fold, epoch, ROC, PR, F1))
    return ROC, PR, F1, test_loss, accuracy


def test(myDataLoader, path, fold, best_model_name):
    name = 'validate_params_' + str(fold)
    model.load_state_dict(torch.load(best_model_name))
    output_list = []
    output_result_list = []
    correct_list = []
    for step, (x, y) in enumerate(myDataLoader):
        model.eval()
        output = model(x)
        output_list += output.cpu().detach().numpy().tolist()
        output = (output > 0.5).int()
        output_result_list += output.cpu().detach().numpy().tolist()
        correct_list += y.cpu().detach().numpy().tolist()
    ROC, PR, F1 = util.draw_ROC_Curve(output_list, output_result_list, correct_list, path + '/' + name)

    tp, fp, tn, fn, accuracy, precision, sensitivity, recall, specificity, MCC, f1_score, Q9, ppv, npv = util.calculate_performace(len(correct_list), output_result_list, correct_list)

    return ROC, PR, F1, sensitivity,  specificity, accuracy, MCC


def getDataSet(train_index, test_index, model):
    x_train = X[train_index]
    y_train = y[train_index]
    x_test = X[test_index]
    y_test = y[test_index]
    x_train_, x_validate_, y_train_, y_validate_ = train_test_split(
        x_train, y_train, test_size=0.125, stratify=y_train)
    x_train_ = x_train_.reset_index(drop=True)
    x_validate_ = x_validate_.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_train_ = y_train_.reset_index(drop=True)
    y_validate_ = y_validate_.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    train_DataSet = ms.MyDataSet(input=x_train_, label=y_train_, model=model)
    validate_DataSet = ms.MyDataSet(input=x_validate_, label=y_validate_, model=model)
    test_DataSet = ms.MyDataSet(input=x_test, label=y_test, model=model)

    train_DataLoader = DataLoader(dataset=train_DataSet, batch_size=Batch_Size)
    validate_DataLoader = DataLoader(dataset=validate_DataSet, batch_size=test_Batch_Size)
    test_DataLoader = DataLoader(dataset=test_DataSet, batch_size=test_Batch_Size)

    return train_DataLoader, validate_DataLoader, test_DataLoader

def get_retrain(retrainPath):
    retrain = pd.read_csv(retrainPath, sep='\s+', names=['name','pre'])
    Xs = retrain['name']
    Ys = retrain['pre']
    retrainList = []
    for i in range(len(retrain)):
        retrainList.append(Xs[i])
    return retrainList

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    now_time = datetime.date.today()
    result_file = open(str(now_time) + '_1stride.txt', 'a')
    result_file.write('NAME' + '\t' + 'ROC' + '\t' + 'PR' + '\t' + 'F1' + '\t' + 'SEN' + '\t' + 'SPE' + '\t' + 'ACC' + '\t' + 'MCC' + '\n')
    Batch_Size = 64
    test_Batch_Size = 128
    LR = 0.001
    Epoch = 2
    K_Fold = 3
    print("Batch_Size", Batch_Size)
    print("Epoch", Epoch)
    print("K_Fold", K_Fold)
    Retrain = open("Retrain.txt",'w')
    file_list = ld.create_list(ld.dataDir)
    # file_list = get_retrain(retrainPath)
    # print(file_list)
    file_list.sort()
    for path in file_list:
        # path = 'wgEncodeAwgTfbsHaibA549Creb1sc240V0416102Dex100nmUniPk'
        print(path)        
        all_data = pd.read_csv(ld.dataDir + path + '/all_data_demo.txt', sep='\s+')
        X = all_data['sequence']
        y = all_data['label']
        kf = StratifiedKFold(n_splits=K_Fold, shuffle=True)
        fold = 1
        roc_total = []
        pr_total = []
        F1_total = []
        sen_total = []
        spe_total = []
        acc_total = []
        mcc_total = []
        for train_index, validate_index in kf.split(X, y):
            model = arc().to(device)
            train_DataLoader, validate_DataLoader, test_DataLoader = getDataSet(train_index, validate_index, model)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
            loss_func = nn.BCELoss().to(device)
            best_model_name = train(train_DataLoader, path, fold)
            print(best_model_name)
            ROC, PR, F1, sensitivity,  specificity, accuracy, MCC = test(test_DataLoader, path, fold, best_model_name)
            roc_total.append(ROC)
            pr_total.append(PR)
            F1_total.append(F1)
            sen_total.append(sensitivity)
            spe_total.append(specificity)
            acc_total.append(accuracy)
            mcc_total.append(MCC)
            fold += 1
        # 获得三折的平均AUC值
        roc_average = np.mean(roc_total)
        pr_average  = np.mean(pr_total)
        f1_average  = np.mean(F1_total)
        sen_average = np.mean(sen_total)
        spe_average = np.mean(spe_total)
        acc_average = np.mean(acc_total)
        mcc_average = np.mean(mcc_total)
        if roc_average < 0.99:
            Retrain.write(path + '\n')
        print(path)
        print("Average ROC:{}\tPR:{}\tF1:{}".format(roc_average, pr_average, f1_average))
        print("#################################")
        result_file.write(path + '\t')
        result_file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(roc_average, pr_average, f1_average, sen_average, spe_average, acc_average, mcc_average))
    result_file.close()
    Retrain.close()