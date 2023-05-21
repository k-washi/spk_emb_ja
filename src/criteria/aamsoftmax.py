'''
AAMsoftmax loss function copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
@ reference
https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/loss.py
'''

import torch, math
import torch.nn as nn
import torch.nn.functional as F

class AAMsoftmax(nn.Module):
    def __init__(self, n_class, hidden_size=64, m=0.2, s=30, cross_entropy_weight=None):
        """AAMsoftmax loss function

        Args:
            n_class (_type_): class num
            hidden_size (int): hidden size (model output). Defaults to 64.
            m (float, optional): loss margin. Defaults to 0.2.
            s (int, optional): loss scale. Defaults to 30.
        """
        
        super(AAMsoftmax, self).__init__()
        if cross_entropy_weight is not None:
            assert n_class == cross_entropy_weight.shape[0], f"Cross entropy weight should have {n_class} classes, but got {cross_entropy_weight.shape[0]}."
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, hidden_size), requires_grad=True) # (out_features, in_features)
        self.ce = nn.CrossEntropyLoss(weight=cross_entropy_weight)
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        
        loss = self.ce(output, label)

        return loss, output