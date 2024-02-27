import pandas as pd
import torch.nn as nn
import torch
from matplotlib import pyplot as plt

from MOLI import MOLI_across
from Model import MyModel
import numpy as np
import torch.utils.data as Data
from torch.autograd import Variable
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score, average_precision_score

from Model_D_across import D_across
from Model_withoutAtten import without_atten
from Model_withoutMask import without_mask
from PictureShow import AUC_show
from RF_across import RF_across
from SVM_across import SVM_across

torch.manual_seed(100)

def Get_data(Datafile_name,Clinicalfile_name):
    set_data=pd.read_csv(Datafile_name)
    set_data=set_data.iloc[:,1:]
    set_clinical=pd.read_csv(Clinicalfile_name)
    set_label = np.array(set_clinical.iloc[:,1])
    return set_data,set_label

def load_data(drug_data, drug_label, BATCHSIZE):
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

def main_model(train_dataloader,test_dataloader):
    # Bortezomib 94 6818
    # LR = 0.01
    # EPOCH=50
    # Paclitaxel 8018
    # LR = 0.001
    # EPOCH=50
    # cisplatin 8018
    # LR = 0.0001
    # EPOCH=50
    # Docetaxel 762 5418
    LR = 0.01
    EPOCH=20
    connection_matrix = pd.read_csv('Drug_data/PathwaymaskFile/pathway_mask_Docetaxel.csv', header=None).values
    # 将连接矩阵转换为浮点类型的张量
    connection_matrix = torch.tensor(connection_matrix, dtype=torch.float32)
    print(connection_matrix.shape)
    Model=MyModel(5418,1358,connection_matrix)
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
            out,x_atten = Model(b_x)
            # out=Model(b_x)
            # if torch.isnan(out).any():
            #     print("NaN value found in outputs before calculating loss")
            loss = loss_func(out, train_label.long())  # 计算损失函数
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 梯度优化
            train_loss += loss.item()
            # 计算准确率
            _,pred = out.max(1)
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
        out_label,x_atten = Model(test_data)
        # attentions.append(x_atten.detach().numpy())
        # out_label=Model(test_data)
        out_label = F.softmax(out_label, dim=1)
        prob_y = out_label[:, 1]  # 获取标签为1的类的概率
        pred_y = torch.argmax(out_label, dim=1)  # 获取预测的类别
        test_label_arr.append(test_label.item())
        prob_arr.append(prob_y.item())
        pred_arr.append(pred_y.item())
    x_atten = x_atten.detach().numpy()
    x_atten=np.ravel(x_atten)
    pathway_df = pd.read_csv('./Immune_data/PathwayMaskedFile/pathName_Geneset.csv')
    pathway_names = pathway_df['pathway_name'].values
    result_df = pd.DataFrame({
        'pathway_name': pathway_names,
        'mean_attention': x_atten
    })
    # result_df.to_csv('./Attention_score/Bortezomib_attentions.csv', index=False)
    return test_label_arr,prob_arr,pred_arr,x_atten


def PBAC_across(data_train, label_train, data_test, label_test):
    # data_train, label_train = Get_data('./Drug_data/CelllineFile/Paclitaxel_exp_after3.csv',
    #                            './Drug_data/CelllineFile/Paclitaxel_label2.csv')
    # data_train = preprocessing.scale(data_train)
    #
    # data_test, label_test=Get_data('./Drug_data/PatientFile/GSE22513_data_after2.csv',
    #                            './Drug_data/PatientFile/GSE22513_label.csv')
    # data_test = preprocessing.scale(data_test)

    train_dataloader = load_data(data_train, label_train, 762)
    test_dataloader = load_data(data_test, label_test, 1)
    test_label, prob_y, pred_y, x_atten = main_model(train_dataloader, test_dataloader)
    return test_label, prob_y

if __name__ == '__main__':
    data_train, label_train = Get_data('./Drug_data/CelllineFile/Docetaxel_exp_after3.csv',
                               './Drug_data/CelllineFile/Docetaxel_label2.csv')
    data_train = preprocessing.scale(data_train)

    data_test, label_test=Get_data('./Drug_data/PatientFile/GSE6434_data_after2.csv',
                               './Drug_data/PatientFile/GSE6434_label.csv')
    data_test = preprocessing.scale(data_test)

    PBAC_test_label,PBAC_prob=PBAC_across(data_train, label_train, data_test, label_test)
    MOLI_test_label,MOLI_prob=MOLI_across(data_train,label_train, data_test, label_test)
    D_test_label,D_prob=D_across(data_train, label_train, data_test, label_test)
    RF_test_label, RF_prob = RF_across(data_train, label_train, data_test, label_test)
    SVM_test_label, SVM_prob = SVM_across(data_train, label_train, data_test, label_test)
    # withoutatten_label,withoutatten_prob=without_atten(data_train, label_train, data_test, label_test)
    # withoutmask_label, withoutmask_prob = without_mask(data_train, label_train, data_test, label_test)

    AUC_show(PBAC_test_label,PBAC_prob,MOLI_test_label,MOLI_prob,D_test_label,D_prob,RF_test_label, RF_prob,SVM_test_label, SVM_prob)
    # auc = roc_auc_score(test_label, prob_y)
    # auprc = average_precision_score(test_label, prob_y)
    # print(test_label)
    # print(pred_y)
    # print(f'AUC: {auc}, AUPRC: {auprc}')
    #
    # auc = roc_auc_score(test_label, prob_y)
    # auprc = average_precision_score(test_label, prob_y)
    #
    # # 画出 ROC 曲线（即 AUC 图）
    # fpr, tpr, _ = roc_curve(test_label, prob_y)
    # plt.figure()
    # plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend(loc='lower right')
    # plt.show()
    #
    # # 画出精确度-召回率曲线（即 AUPRC 图）
    # precision, recall, _ = precision_recall_curve(test_label, prob_y)
    # plt.figure()
    # plt.plot(recall, precision, label=f'AUPRC = {auprc:.2f}')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve')
    # plt.legend(loc='lower left')
    # plt.show()