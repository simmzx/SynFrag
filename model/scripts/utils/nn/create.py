import torch
from torch.autograd import Variable


def create_var(tensor, requires_grad=None):
    device = torch.device("cuda:" + "0") if torch.cuda.is_available() else torch.device("cpu")
    if requires_grad is None:
        return Variable(tensor).to(device)
    else:
        return Variable(tensor, requires_grad=requires_grad).to(device)
