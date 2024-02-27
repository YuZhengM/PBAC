import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(100)
class DengModel(nn.Module):
    def __init__(self, gene_num, pathway_num, connection_matrix=None):
        super(DengModel, self).__init__()

        # 创建基因-通路连接权重矩阵，初始化为0
        self.connection_matrix = nn.Parameter(torch.zeros((pathway_num,gene_num)), requires_grad=False)
        if connection_matrix is not None:
            assert connection_matrix.shape == (
            pathway_num,gene_num), "The shape of the connection matrix is incorrect."
            self.connection_matrix.data.copy_(connection_matrix)

        # 全连接层1: 用于替代注意力机制层
        self.dense1 = nn.Linear(pathway_num, 256)

        # 全连接层2: 用于替代卷积层
        self.dense2 = nn.Linear(256, 128)

        # 分类层
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        # 使用预定义的连接矩阵对基因表达数据进行过滤
        x = torch.mm(x, self.connection_matrix.t())

        # 全连接层1
        x = F.relu(self.dense1(x))

        # 全连接层2
        x = F.relu(self.dense2(x))

        # 分类层
        x = self.classifier(x)

        return x

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
    LR = 0.001
    EPOCH=20
    connection_matrix = pd.read_csv('Drug_data/PathwaymaskFile/pathway_mask_Docetaxel.csv', header=None).values
    # 将连接矩阵转换为浮点类型的张量
    connection_matrix = torch.tensor(connection_matrix, dtype=torch.float32)
    print(connection_matrix.shape)
    Model=DengModel(5418,1358,connection_matrix)
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
            out = Model(b_x)
            # out=Model(b_x)
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
        out_label = Model(test_data)
        # attentions.append(x_atten.detach().numpy())
        # out_label=Model(test_data)
        out_label = F.softmax(out_label, dim=1)
        prob_y = out_label[:, 1]  # 获取标签为1的类的概率
        pred_y = torch.argmax(out_label, dim=1)  # 获取预测的类别
        test_label_arr.append(test_label.item())
        prob_arr.append(prob_y.item())
        pred_arr.append(pred_y.item())
    return test_label_arr,prob_arr,pred_arr


def D_across(data_train, label_train, data_test, label_test):
    # data_train, label_train = Get_data('./Drug_data/CelllineFile/Paclitaxel_exp_after3.csv',
    #                                    './Drug_data/CelllineFile/Paclitaxel_label2.csv')
    # data_train = preprocessing.scale(data_train)
    #
    # data_test, label_test = Get_data('./Drug_data/PatientFile/GSE22513_data_after2.csv',
    #                                  './Drug_data/PatientFile/GSE22513_label.csv')
    # data_test = preprocessing.scale(data_test)
    train_dataloader = load_data(data_train, label_train, 762)
    test_dataloader = load_data(data_test, label_test, 1)
    test_label, prob_y, pred_y = main_model(train_dataloader, test_dataloader)
    return test_label, prob_y
    # auc = roc_auc_score(test_label, prob_y)
    # auprc = average_precision_score(test_label, prob_y)
    # print(test_label)
    # print(pred_y)
    # print(f'AUC: {auc}, AUPRC: {auprc}')
