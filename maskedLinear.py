import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
class maskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, relation_file, bias=True):
        super(maskedLinear, self).__init__(in_features, out_features, bias)

        mask = self.readRelationFromFile(relation_file)
        self.register_buffer('mask', mask)

        self.iter = 0
    def forward(self, input):
        masked_weight = self.weight * self.mask
        return F.linear(input, masked_weight, self.bias)

    def readRelationFromFile(self, relation_file):
        mask = []
        with open(relation_file, 'r') as f:
            for line in f:
                l = [int(float(x)) for x in line.strip().split(',')]
                for item in l:
                    assert item == 1 or item == 0  # relation 只能为0或者1
                mask.append(l)
        return Variable(torch.Tensor(mask))
