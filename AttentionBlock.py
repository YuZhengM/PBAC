import torch
from torch import nn


class GenePathwayClassifier(nn.Module):
    def __init__(self, gene_num=8001, pathway_num=1358, connection_matrix=None):
        super(GenePathwayClassifier, self).__init__()

        # 创建基因-通路连接权重矩阵，初始化为0
        self.connection_matrix = nn.Parameter(torch.zeros((gene_num, pathway_num)), requires_grad=False)
        if connection_matrix is not None:
            assert connection_matrix.shape == (
            gene_num, pathway_num), "The shape of the connection matrix is incorrect."
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

        # 分类层
        self.classifier = nn.Linear(pathway_num, 2)

    def forward(self, x):
        # 使用预定义的连接矩阵对基因表达数据进行过滤
        x = torch.mm(x, self.connection_matrix)

        # 全连接层
        x = self.dense(x)

        # 注意力权重
        attn_weights = torch.softmax(self.attention(x), dim=1)

        # 加权求和
        weighted_pathway = attn_weights * x

        x = weighted_pathway.unsqueeze(1)
        x = self.conv(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x, attn_weights  # 返回分类结果以及每个通路的注意力分数
# 创建一个模型实例
model = GenePathwayClassifier()

# 创建一个连接矩阵，表示基因和通路的存在关系
connection_matrix = torch.rand(8001, 1358).float()
model.connection_matrix.data.copy_(connection_matrix)

# 创建一个简单的输入，大小为(batch_size, gene_num)，例如 (10, 8001)
inputs = torch.randint(0, 2, size=(10, 8001)).float()  # 使用随机二值作为示例输入

# 前向传播
outputs, pathway_scores = model(inputs)

print(outputs)
print(pathway_scores)