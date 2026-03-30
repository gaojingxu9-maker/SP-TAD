import scipy.stats as stats
import torch
import torch.nn as nn
from torch.nn import functional as F

keys = F.normalize(torch.rand((5,3), dtype=torch.float), dim=1)
key = F.normalize(torch.rand((6, 3), dtype=torch.float), dim=1)
counts = torch.tensor([6,5,4,9,8])

attn = torch.matmul(key, torch.t(keys.detach()))  # (TxC) x (CxM) -> TxM
_, max_indices = torch.max(attn, dim=1)
print(max_indices)
read_query = keys[max_indices]

increments = torch.ones_like(max_indices)
counts.scatter_add_(0, max_indices, increments)
print(counts)


#
# attn = torch.matmul(key.detach(), torch.t(keys))  # (TxC) x (CxM) -> TxM
# _, indices = torch.topk(attn, 1, dim=-1)  # T
# indices_items = attn.gather(dim=1, index=indices)
# rows, cols = torch.where(indices_items > 0.92)
# new_items =0.5 * key[rows] + 0.5 * keys[indices[rows]].squeeze(1)
# keys[indices[rows].squeeze(1)] = new_items
#
# ###################################
# k, _ = torch.where(indices_items <= 0.92)
# _, min_indices = torch.topk(-counts,k.size(0), dim=0)  # T
# keys[min_indices] = key[k]
# counts[min_indices] = 0
#
#
