import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Loss layers ---------------- #

class BCELossLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, logits, targets):
        # logits, targets: same shape, e.g. (N, 1, H, W)
        return F.binary_cross_entropy_with_logits(logits, targets)

class CELossLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
    def forward(self, logits, targets):
        # logits: (N, C, H, W), targets: (N, H, W) with class indices in {0,...,C-1}
        return self.ce(logits, targets)

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, probs, targets):
        p = probs.view(-1)
        t = targets.view(-1)
        inter = (p * t).sum()
        dice = (2 * inter + self.eps) / (p.sum() + t.sum() + self.eps)
        return 1 - dice

class IoULoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, probs, targets):
        p = probs.view(-1)
        t = targets.view(-1)
        inter = (p * t).sum()
        union = p.sum() + t.sum() - inter
        return 1 - inter / (union + self.eps)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, eps=1e-8):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
    def forward(self, probs, targets):
        p = probs.view(-1)
        t = targets.view(-1)
        pt = p * t + (1 - p) * (1 - t)          # prob of correct class (binary)
        return -((1 - pt) ** self.gamma * torch.log(pt + self.eps)).mean()

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
    def forward(self, probs, targets):
        p = probs.view(-1)
        t = targets.view(-1)
        inter = (p * t).sum()
        fp = (p * (1 - t)).sum()
        fn = ((1 - p) * t).sum()
        tversky = inter / (inter + self.alpha * fp + self.beta * fn + self.eps)
        return 1 - tversky

# ---------------- Examples ---------------- #

if __name__ == "__main__":
    # 1) Binary example for BCE + custom losses
    # logits_bin, targets_bin: shape (N, 1, H, W)
    logits_bin = torch.tensor([[[[ 2.0, -1.0],
                                 [ 0.5,  3.0]]]])   # (1,1,2,2)
    targets_bin = torch.tensor([[[[1.0, 0.0],
                                  [1.0, 1.0]]]])   # (1,1,2,2)

    probs_bin = torch.sigmoid(logits_bin)          # for custom losses

    bce   = BCELossLayer()
    dice  = DiceLoss()
    iou   = IoULoss()
    focal = FocalLoss()
    tvers = TverskyLoss()

    print("BCE     :", bce(logits_bin, targets_bin).item())
    print("Dice    :", dice(probs_bin, targets_bin).item())
    print("IoU     :", iou(probs_bin, targets_bin).item())
    print("Focal   :", focal(probs_bin, targets_bin).item())
    print("Tversky :", tvers(probs_bin, targets_bin).item())

    # 2) Multi-class example for CrossEntropyLoss (C=2)
    # logits_ce: (N, C, H, W), targets_ce: (N, H, W) with class indices {0,1}
    logits_ce = torch.tensor(
        [[[[ 2.0, -1.0],
           [ 0.5,  3.0]],     # class 0 scores
          [[-0.5,  1.0],
           [ 2.0, -2.0]]]]    # class 1 scores
    )  # shape (1,2,2,2)

    targets_ce = torch.tensor([[[0, 1],
                                [1, 0]]])          # shape (1,2,2), long

    ce = CELossLayer()
    print("CE      :", ce(logits_ce, targets_ce).item())
