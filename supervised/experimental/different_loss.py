"""
Experimenting with different loss functions
"""

import torch as t
from torch.nn.functional import softmax


def cyclical_loss(logits, labels):
    probs = softmax(logits, dim=-1)
    l = logits.shape[-1]
    distr = t.arange(l).cuda()
    diffs = (distr.squeeze(0) - labels.unsqueeze(-1)) % l
    diffs2 = (l - (distr.squeeze(0) - labels.unsqueeze(-1))) % l
    d = t.minimum(diffs, diffs2)
    return t.sum(probs * d) / labels.shape[0]
