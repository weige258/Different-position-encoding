import torch
import math
def positional_encoding(tensor):
    pe=torch.zeros(tensor.size()[0],tensor.size()[1])
    pos_index=torch.arange(0,tensor.size(0))
    div_term = torch.exp(torch.arange(0, tensor.size()[1],2) * (-math.log(10000.0)/tensor.size()[1]))
    pe[:, 0::2] = torch.sin(pos_index.unsqueeze(-1) / div_term)
    pe[:, 1::2] = torch.cos(pos_index.unsqueeze(-1) / div_term)
    tensor=tensor.to(dtype=torch.float)+pe
    return tensor
