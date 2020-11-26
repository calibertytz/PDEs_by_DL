import torch


def net_deriviate(net, x, order=1):
    x_ = x.clone().detach().view(-1)
    param = list(net.parameters())
    A1, b1 = param[0], param[1].view(-1, 1)
    A2 = param[2]
    if order == 1:
        I = A1 * (torch.sigmoid(A1 * x_ + b1) - (torch.sigmoid(A1 * x_ + b1)) ** 2)
        return torch.mm(A2, I)
    elif order == 2:
        I_1 = A1 * (torch.sigmoid(A1 * x_ + b1) - (torch.sigmoid(A1 * x_ + b1)) ** 2)
        I_2 = 2 * A1 * (torch.sigmoid(A1 * x_ + b1) - (torch.sigmoid(A1 * x_ + b1)) ** 2) * torch.sigmoid(A1 * x_ + b1)
        I_ = A1 * (I_1 - I_2)
        return torch.mm(A2, I_)
    else:
        raise NotImplementedError


def batch_get_deriviate(net, batch_x, order=1):
    return torch.stack([net_deriviate(net, x, order) for x in batch_x])


def get_difference(x, dt, order=1):
    global difference
    if order > 0:
        l = x.size()[0]
        difference = torch.tensor([(x[i] - x[i - 1]) / dt if i > 0 else x[i] / dt for i in range(0, l)])
        return get_difference(difference, dt, order - 1)
    else:
        return difference


def batch_get_difference(batch_x, dt, order=1):
    r = batch_x.size()[0]
    return torch.stack([get_difference(batch_x[i], dt, order) for i in range(r)]).unsqueeze(2)


