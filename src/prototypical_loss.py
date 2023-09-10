import torch
from torch.nn import functional as F
from torch.nn.modules import Module

from torch.autograd import Variable


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(out: torch.tensor, target: torch.tensor, n_support: int, n_classes: int):
    '''
    Compute the barycentres by averaging the features of n_support samples
    '''

    target_cpu = target.to('cpu')
    out_cpu = out.to('cpu')

    # input is made by (NQ + NS) for each NC.
    # batch = NC * (NS + NQ)
    # out should be [batch * 1600]

    batch_size = out_cpu.shape[0]
    n_query = int( (batch_size - (n_classes * n_support)) / n_classes )

    indexes_support = []
    indexes_query = []
    for i in range(n_classes):
        start = i * (n_support + n_query)
        stop = (i + 1) * (n_support + n_query)
        indexes_support.append(list(range(start, start + n_support, 1)))
        indexes_query += list(range(start + n_support, stop, 1))

    #indexes_support = torch.tensor(indexes_support)
    prototypes = torch.stack([out_cpu[idx_list].mean(0) for idx_list in indexes_support])

    #indexes_query = torch.tensor(indexes_query)
    query_samples = out_cpu[indexes_query]# torch.stack([out_cpu[idx_list] for idx_list in indexes_query])

    dists = euclidean_dist(query_samples, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()

    return loss_val,  acc_val
