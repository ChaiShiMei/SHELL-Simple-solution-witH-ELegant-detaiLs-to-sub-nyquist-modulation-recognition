import torch
import torch.nn as nn
from torch.nn import functional as F
import pdb

class OhemCrossEntropy(nn.Module):
    def __init__(self, thres=0.7,
                 min_kept=0.7, weight=None, label_smoothing=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(0.1, min_kept)
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            reduction='none',
            label_smoothing=label_smoothing
        )

    def _ce_forward(self, score, target):
        loss = self.criterion(score, target)
        return loss

    # def forward(self, score, target, **kwargs):
    #     pdb.set_trace()
    #     losses = self.criterion(score, target).contiguous().view(-1)
    #     losses_sorted, ind = losses.data.sort()
    #     min_value = losses_sorted[min(self.min_kept, losses_sorted.numel() - 1)]
    #     threshold = max(min_value, self.thresh)

    #     losses = losses[ind]
    #     losses = losses[losses_sorted < threshold]
    #     return losses.mean()
    def forward(self, score, target, **kwargs):
        min_kept_value = int(self.min_kept * score.shape[0])
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)

        tmp_target = target.clone()
        pred = pred.gather(1, tmp_target.unsqueeze(1).type(torch.int64))
        pred, ind = pred.contiguous().view(-1,).contiguous().sort()
        min_value = pred[min(min_kept_value, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

def weighted_bce(bd_pre, target):
    n, c, h, w = bd_pre.size()
    log_p = bd_pre.permute(0,2,3,1).contiguous().view(1, -1)
    target_t = target.view(1, -1)

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)

    weight = torch.zeros_like(log_p)
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')

    return loss


class BondaryLoss(nn.Module):
    def __init__(self, coeff_bce = 20.0):
        super(BondaryLoss, self).__init__()
        self.coeff_bce = coeff_bce
        
    def forward(self, bd_pre, bd_gt):

        bce_loss = self.coeff_bce * weighted_bce(bd_pre, bd_gt)
        loss = bce_loss
        
        return loss
    
if __name__ == '__main__':
    a = torch.zeros(2,64,64)
    a[:,5,:] = 1
    pre = torch.randn(2,1,16,16)
    
    Loss_fc = BondaryLoss()
    loss = Loss_fc(pre, a.to(torch.uint8))

        
        
        

