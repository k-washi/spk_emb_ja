'''
AAMsoftmax loss function copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
@ reference
https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/loss.py
'''
from typing import List
import torch, math
import torch.nn as nn
import torch.nn.functional as F

class CELoss(nn.Module):
    def __init__(self, n_class, hidden_size=64,  cross_entropy_weight=None):
        """AAMsoftmax loss function

        Args:
            n_class (_type_): class num
            hidden_size (int): hidden size (model output). Defaults to 64.
            m (float, optional): loss margin. Defaults to 0.2.
            s (int, optional): loss scale. Defaults to 30.
        """
        
        super(CELoss, self).__init__()
        if cross_entropy_weight is not None:
            if isinstance(cross_entropy_weight, List):
                cross_entropy_weight = torch.FloatTensor(cross_entropy_weight)
            assert n_class == cross_entropy_weight.shape[0], f"Cross entropy weight should have {n_class} classes, but got {cross_entropy_weight.shape[0]}."
        
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, hidden_size), requires_grad=True) # (out_features, in_features)
        self.ce = nn.CrossEntropyLoss(weight=cross_entropy_weight)
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        x = F.log_softmax(x, dim=1)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        cosine = torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        
        loss = self.ce(output, label)

        return loss, output