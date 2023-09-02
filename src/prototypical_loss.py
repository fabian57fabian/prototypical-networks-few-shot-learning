import torch
from torch.nn import functional as F
from torch.nn.modules import Module


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


def prototypical_loss(model_out, y, number_support):
    '''
    Compute the barycentres by averaging the features of n_support samples
    Args:
    - input: the model output for a batch of samples (already on cpu)
    - target: ground truth for the above batch of samples (already on cpu)
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    classes = torch.unique(model_out)
    n_classes = len(classes)

    # n_query, n_target constants
    n_query = model_out.eq(classes[0].item()).sum().item() - number_support

    support_idxs = [model_out.eq(c).nonzero()[:number_support].squeeze(1) for c in classes]

    prototypes = torch.stack([y[idx_list].mean(0) for idx_list in support_idxs])
    query_idxs = torch.stack(list(map(lambda c: model_out.eq(c).nonzero()[number_support:], classes))).view(-1)

    query_samples = model_out.to('cpu')[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()

    return loss_val,  acc_val