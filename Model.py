import math

import torch

from maskedLinear import maskedLinear
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
# GSE106128
# num_gene=5046
# DavidLiu
# num_gene=6796
# GSE176307
# num_gene=7689
# GSE115821
# num_gene=8001
# GSE100797
# num_gene=8531
# PRJNA482620
# num_gene=8744
# GSE35640
# num_gene=8233
# GSE19293
# num_gene=8234
# GSE91061
# GSE78220
# num_gene=8743
# IMvigor210
# num_gene=8757
# phs000452=8744
# num_pathway=1358

class MyModel(nn.Module):
    def __init__(self, gene_num, pathway_num, connection_matrix=None):
        super(MyModel, self).__init__()

        # 创建基因-通路连接权重矩阵，初始化为0
        self.connection_matrix = nn.Parameter(torch.zeros((pathway_num,gene_num)), requires_grad=False)
        if connection_matrix is not None:
            assert connection_matrix.shape == (
            pathway_num,gene_num), "The shape of the connection matrix is incorrect."
            self.connection_matrix.data.copy_(connection_matrix)

        # 全连接层
        self.dense = nn.Linear(pathway_num, pathway_num)

        # 注意力机制层
        self.attention = nn.Sequential(
            nn.Linear(pathway_num, 128),
            nn.Tanh(),
            nn.Linear(128, pathway_num)  # 将此处修改为通路数量
        )

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

        # 注意力权重
        attn_weights = torch.softmax(self.attention(x), dim=1)

        # 加权求和
        weighted_pathway = attn_weights * x

        x = weighted_pathway.unsqueeze(1)
        x = self.conv(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.classifier(x)

        return x, attn_weights  # 返回分类结果以及每个通路的注意力分数



