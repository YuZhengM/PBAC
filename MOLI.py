from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score, average_precision_score
import pandas as pd
import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from Model import MyModel
import numpy as np
import torch.utils.data as Data
from torch.autograd import Variable
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch.nn.functional as F
torch.manual_seed(100)
# 定义自动编码器模型
# paclitaxel:0.00001,20
# bortezomib:0.0001,30 6818
# cisplatin
# DOCETAXEL:0.0001,45 54818
class MOLI(nn.Module):
    def __init__(self):
        super(MOLI, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5418, 2592),
            nn.ReLU(),
            nn.Linear(2592, 648),
            nn.ReLU(),
            nn.Linear(648, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 648),
            nn.ReLU(),
            nn.Linear(648, 2592),
            nn.ReLU(),
            nn.Linear(2592, 5418),
            nn.Sigmoid()
        )
        self.classifier = nn.Linear(16, 2)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        output = self.classifier(encoded)
        return decoded, output

def Get_data(Datafile_name, Clinicalfile_name):
    set_data = pd.read_csv(Datafile_name)
    set_data = set_data.iloc[:, 1:]
    set_clinical = pd.read_csv(Clinicalfile_name)
    set_label = np.array(set_clinical.iloc[:, 1])
    return set_data, set_label

def load_data(drug_data, drug_label, BATCHSIZE):
    # x=drug_data
    x = []
    y = drug_label
    for i in range(drug_data.shape[0]):
        # print(drug_data[i,:])
        x.append(np.array(drug_data[i]))
    x = torch.FloatTensor(np.array((x)))
    # print(x.size())
    y = torch.FloatTensor(np.array(y))
    # y = y.unsqueeze(1)
    # print(y.size())
    torch_dataset = Data.TensorDataset(x, y)
    data_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCHSIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    return data_loader

def main_model(train_dataloader, test_dataloader):
    LR = 0.0001
    EPOCH = 45
    Model = MOLI()
    # Model=DNN()
    # Model=Atten_CNN()
    # Model=Mask_CNN()
    optimizer = torch.optim.Adam(Model.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(EPOCH):
        train_loss = 0
        train_acc = 0
        for step, (x, train_label) in enumerate(train_dataloader):
            # b_x = Variable(x).cuda()
            b_x = Variable(x)
            # train_label=Variable(train_label.squeeze(1)).cuda()
            train_label = Variable(train_label)
            # print(train_label)
            out, x_atten = Model(b_x)
            # out=Model(b_x)
            loss = loss_func(out, train_label.long())  # 计算损失函数
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 梯度优化
            train_loss += loss.item()
            # 计算准确率
            _, pred = out.max(1)
            num_correct = (pred == train_label).sum().item()
            # num_correct=num_correct.numpy()
            acc = num_correct / b_x.shape[0]
            train_acc += acc
        print('Epoch: {}, Train Loss: {:.8f}, Train Acc: {:.6f}'
                .format(epoch, train_loss / len(train_dataloader), train_acc / len(train_dataloader)))
    Model.eval()
    print('开始测试')
    test_acc = 0
    test_label_arr = []
    prob_arr = []
    pred_arr = []
    for test_data, test_label in test_dataloader:
        test_data = Variable(test_data)
        out_label, x_atten = Model(test_data)
        # attentions.append(x_atten.detach().numpy())
        # out_label=Model(test_data)
        out_label = F.softmax(out_label, dim=1)
        prob_y = out_label[:, 1]  # 获取标签为1的类的概率
        pred_y = torch.argmax(out_label, dim=1)  # 获取预测的类别
        test_label_arr.append(test_label.item())
        prob_arr.append(prob_y.item())
        pred_arr.append(pred_y.item())
    x_atten = x_atten.detach().numpy()
    return test_label_arr, prob_arr, pred_arr

def MOLI_across(data_train, label_train,data_test, label_test):
    # data_train, label_train = Get_data('./Drug_data/CelllineFile/Docetaxel_exp_after3.csv',
    #                                    './Drug_data/CelllineFile/Docetaxel_label2.csv')
    # data_train = preprocessing.scale(data_train)
    # print(data_train.shape)
    # print(label_train.shape)
    # data_test, label_test = Get_data('./Drug_data/PatientFile/GSE6434_data_after2.csv',
    #                                  './Drug_data/PatientFile/GSE6434_label.csv')
    # data_test = preprocessing.scale(data_test)
    train_dataloader = load_data(data_train, label_train, 764)
    test_dataloader = load_data(data_test, label_test, 1)
    test_label, prob_y, pred_y = main_model(train_dataloader, test_dataloader)
    return test_label, prob_y
# auc = roc_auc_score(test_label, prob_y)
# auprc = average_precision_score(test_label, prob_y)
# fpr1, tpr1, _ = roc_curve(test_label, prob_y)
# plt.figure()
# plt.plot(fpr1, tpr1, label='PBAC (area = %0.2f)' % auc,color='red',linewidth=3,alpha=1)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()
# precision, recall, _ = precision_recall_curve(test_label, prob_y)
# plt.figure()
# plt.plot(recall, precision, label=f'AUPRC = {auprc:.2f}')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend(loc='lower left')
# plt.show()
# print(f'AUC: {auc}, AUPRC: {auprc}')
