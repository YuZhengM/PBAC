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
from sklearn.metrics import roc_curve, auc, precision_recall_curve
class MyModelWithoutAttention(nn.Module):
    def __init__(self, gene_num, pathway_num, connection_matrix=None):
        super(MyModelWithoutAttention, self).__init__()

        # 创建基因-通路连接权重矩阵，初始化为0
        self.connection_matrix = nn.Parameter(torch.zeros((pathway_num,gene_num)), requires_grad=False)
        if connection_matrix is not None:
            assert connection_matrix.shape == (
            pathway_num,gene_num), "The shape of the connection matrix is incorrect."
            self.connection_matrix.data.copy_(connection_matrix)

        # 全连接层
        self.dense = nn.Linear(pathway_num, pathway_num)

        # 卷积层
        self.conv = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(1358, 100)

        # 分类层
        self.classifier = nn.Linear(100, 2)

    def forward(self, x):
        # 使用预定义的连接矩阵对基因表达数据进行过滤
        x = torch.mm(x, self.connection_matrix.t())

        # 全连接层
        x = self.dense(x)

        x = x.unsqueeze(1)
        x = self.conv(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.classifier(x)

        return x

torch.manual_seed(100)

def Get_data(Datafile_name,Clinicalfile_name):
    set_data=pd.read_csv(Datafile_name)
    set_data=set_data.iloc[:,1:]
    set_clinical=pd.read_csv(Clinicalfile_name)
    set_label = np.array(set_clinical.iloc[:,1])
    return set_data,set_label

def load_data(drug_data,drug_label,BATCHSIZE):
    # x=drug_data
    x=[]
    y=drug_label
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
    LR = 0.0001
    EPOCH=20
    connection_matrix = pd.read_csv('Drug_data/PathwaymaskFile/pathway_mask_Bortezomib.csv', header=None).values
    # 将连接矩阵转换为浮点类型的张量
    connection_matrix = torch.tensor(connection_matrix, dtype=torch.float32)
    print(connection_matrix.shape)
    Model=MyModelWithoutAttention(6818,1358,connection_matrix)
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
            out= Model(b_x)
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
        out_label= F.softmax(out_label, dim=1)
        prob_y = out_label[:, 1]  # 获取标签为1的类的概率
        pred_y = torch.argmax(out_label, dim=1)  # 获取预测的类别
        test_label_arr.append(test_label.item())
        prob_arr.append(prob_y.item())
        pred_arr.append(pred_y.item())
    return test_label_arr,prob_arr,pred_arr

def without_atten(data_train, label_train,data_test, label_test):
    # PBAC_auc = roc_auc_score(PBAC_test_label, PBAC_prob)
    # PBAC_auprc = average_precision_score(PBAC_test_label, PBAC_prob)
    train_dataloader = load_data(data_train, label_train, 753)
    test_dataloader = load_data(data_test, label_test, 1)
    test_label, prob_y, pred_y = main_model(train_dataloader, test_dataloader)
    return test_label, prob_y
    # print('run')
    # set_data, set_label = Get_data('./Immune_data/PatientFile/GSE78220_after.csv',
    #                                'Immune_data/CelllineFile/GSE78220_clinical.csv')
    # set_data = preprocessing.scale(set_data)
    #
    # # 创建StratifiedKFold对象
    # skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    #
    # tprs = []
    # aucs = []
    # attentions=[]
    # mean_fpr = np.linspace(0, 1, 100)
    #
    # auprcs = []
    # mean_recall = np.linspace(0, 1, 100)
    # precisions = []
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10)) # 创建两个子图
    # i = 0
    # for train_index, test_index in skf.split(set_data, set_label):
    #     print(f"正在进行第{i + 1}次训练...")  # 打印当前的迭代次数
    #     data_train, data_test = set_data[train_index], set_data[test_index]
    #     label_train, label_test = set_label[train_index], set_label[test_index]
    #
    #     train_dataloader = load_data(data_train, label_train, 1)
    #     test_dataloader = load_data(data_test, label_test, 1)
    #
    #     test_label, prob_y, pred_y = main_model(train_dataloader, test_dataloader)
    #     # 计算并画出每一折的ROC曲线
    #     fpr, tpr, _ = roc_curve(test_label, prob_y)
    #     roc_auc = auc(fpr, tpr)
    #     ax1.plot(fpr, tpr, lw=1, alpha=0.3,
    #              label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    #
    #     tprs.append(np.interp(mean_fpr, fpr, tpr))
    #     tprs[-1][0] = 0.0
    #     aucs.append(roc_auc)
    #     # 计算并画出每一折的PR曲线
    #     precision, recall, _ = precision_recall_curve(test_label, prob_y)
    #     auprc = auc(recall, precision)
    #     auprcs.append(auprc)
    #     ax2.plot(recall, precision, lw=1, alpha=0.3,
    #              label='PRC fold %d (AUPRC = %0.2f)' % (i, auprc))
    #     interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
    #     precisions.append(interp_precision)
    #
    #     i += 1
    # # 画出随机猜测的结果线
    # # plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
    # #          label='Chance', alpha=.8)
    # # 计算并画出平均ROC曲线
    # mean_tpr = np.mean(tprs, axis=0)
    # mean_tpr[-1] = 1.0
    # mean_auc = auc(mean_fpr, mean_tpr)
    # std_auc = np.std(aucs)
    # ax1.plot(mean_fpr, mean_tpr, color='b',
    #          label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
    #          lw=2, alpha=.8)
    #
    # # 计算并画出平均PR曲线
    # mean_precision = np.mean(precisions, axis=0)
    # mean_auprc = auc(mean_recall, mean_precision)
    # ax2.plot(mean_recall, mean_precision, color='b',
    #          label=r'Mean PRC (AUPRC = %0.2f)' % (mean_auprc),
    #          lw=2, alpha=.8)
    #
    # # 设置图像属性
    # ax1.set_xlim([0, 1])
    # ax1.set_ylim([0, 1])
    # ax1.set_xlabel('False Positive Rate')
    # ax1.set_ylabel('True Positive Rate')
    # ax1.set_title('Receiver operating characteristic example')
    # ax1.legend(loc="lower right")
    #
    # ax2.set_xlim([0, 1])
    # ax2.set_ylim([0, 1])
    # ax2.set_xlabel('Recall')
    # ax2.set_ylabel('Precision')
    # ax2.set_title('Precision-Recall curve')
    # ax2.legend(loc="lower right")
    #
    # plt.show()