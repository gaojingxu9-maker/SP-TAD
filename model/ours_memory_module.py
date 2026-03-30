from __future__ import absolute_import, print_function
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from utils.utils import EntropyLoss


class MemoryModule(nn.Module):#定义MemoryModule类的初始化函数，参数包括记忆单元数量、特征维度、设备和数据集名称
    def __init__(self, n_memory, fea_dim, device=None, dataset_name=None):
        super(MemoryModule, self).__init__()
        self.n_memory = n_memory
        self.fea_dim = fea_dim  # C(=d_model)
        self.device = device
        self.entropy_loss = EntropyLoss() # 初始化熵损失函数，用于计算注意力权重的熵
        self.temperature = 0.5  #设置温度参数，用于softmax计算中的温度缩放



        self.lsoftmax = torch.nn.LogSoftmax(dim=-1) #初始化对数softmax函数，用于对比损失计算


        print('loading memory item with random initilzation')#使用随机初始化加载记忆项
       # self.keys = F.normalize(torch.rand((self.n_memory, self.fea_dim), dtype=torch.float),dim=1).to(self.device)
    #    self.values = F.normalize(torch.rand((self.n_memory, self.fea_dim), dtype=torch.float), dim=1).to(self.device)
        self.keys = nn.Parameter(torch.rand((self.n_memory, self.fea_dim), dtype=torch.float))
        self.values = nn.Parameter(torch.rand((self.n_memory, self.fea_dim), dtype=torch.float) )
    #     self.keys = nn.Parameter(torch.empty(self.n_memory, self.fea_dim))
    #     self.values = nn.Parameter(torch.empty(self.n_memory, self.fea_dim))
    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     stdv = 1. / math.sqrt(self.keys.size(1))
    #     self.keys.data.uniform_(-stdv, stdv)
    #     if self.keys is not None:
    #         self.keys.data.uniform_(-stdv, stdv)
    #
    #     self.values.data.uniform_(-stdv, stdv)
    #     if self.values is not None:
    #         self.values.data.uniform_(-stdv, stdv)


    # 该函数计算查询和键的聚集损失。首先通过矩阵乘法和softmax计算相似度得分，选取最相似的记忆项；
    # 然后分别计算键和查询与记忆项的MSE损失；
    # 最后计算对比损失，通过einsum实现相似度计算并用logsoftmax处理，返回查询的聚集损失和对比损失
    def gatheringLoss(self,key, query):#定义聚集损失计算函数，输入为键(key)和查询(query)
        batch_size = query.size(0)#获取查询的批次大小和序列长度
        L = query.size(1)
        #创建均方误差损失函数实例
        loss_mse = torch.nn.MSELoss()
        #计算键与记忆键的相似度得分：键矩阵(TxC)与记忆键转置(CxM)相乘得到(TxM)的得分矩阵
        score = torch.matmul(key, torch.t(self.keys))  # Fea x Mem^T : (TXC) X (CXM) = TxM
        score = F.softmax(score, dim=-1)  # TxM    对得分进行softmax归一化，得到注意力权重分布

       #获取每个键最相似的记忆项索引（top-1）
        _, indices = torch.topk(score, 1, dim=1)

        #计算键与最相似记忆键的MSE和查询与最相似记忆值的MSE
        key_gathering_loss = loss_mse(key, self.keys[indices].squeeze(1))
        query_gathering_loss = loss_mse(query, self.values[indices].squeeze(1))

        ################## 聚集损失 ##################
        key_logits = torch.einsum("tc,mc->tm",key, self.keys)#使用爱因斯坦求和计算键与所有记忆键的相似度（与score计算等价）
        key_logits /= self.temperature #温度缩放，调整softmax的锐度
        key_nec = self.lsoftmax(key_logits)          #应用对数softmax函数

       #根据最相似索引收集对应的对数概率
        key_contrast_loss =key_nec.gather(dim=1, index=indices)         # T * M
        key_contrast_loss = torch.sum(key_contrast_loss)  # T   求和得到总的对比损失
        key_contrast_loss /= -1. * 128 * 100  #归一化损失值（硬编码的批次大小和序列长度）

        #对查询执行相同的对比损失计算过程
        query_logits = torch.einsum("tc,mc->tm",key, self.keys)
        query_logits /= self.temperature
        query_nec = self.lsoftmax(query_logits)
        query_contrast_loss =query_nec.gather(dim=1, index=indices)         # T * M
        query_contrast_loss = torch.sum(query_contrast_loss)  # T
        query_contrast_loss /= -1. * 128 * 100

        return query_gathering_loss, query_contrast_loss
        # return key_gathering_loss+query_gathering_loss, key_contrast_loss+query_contrast_loss


    def update(self, key,query):#更新记忆项（聚类中心），并分离编码器参数
        '''
        Update memory items(cluster centers)
        Fix Encoder parameters (detach)
        query (encoder output features) : (NxL) x C or N x C -> T x C
        '''
        #矩阵相乘的结果表示记忆项与输入键的相似度
        attn = torch.matmul(self.keys, torch.t(key.detach()))  # (MxC) x (CxT) -> M x T
        attn = F.softmax(attn, dim=-1)

#注意力权重与输入键矩阵相乘，得到新的记忆键
        keys = torch.matmul(attn, key.detach().to(self.device))   # M x C
        self.keys = F.normalize(keys.detach(), dim=1)

        queries = torch.matmul(attn, query.detach())   # M x C
        self.values = F.normalize(queries.detach(), dim=1)
# #整个更新过程通过注意力机制将输入的键（季节提取器和趋势提取器所提取到的特征表示）
#     # 和查询信息（编码器对整个序列的综合理）聚合到记忆模块中，实现记忆项的动态更新。
#     def update(self, key, query):
#         '''
#         Update memory items(cluster centers) with momentum
#         Allow gradients to flow back to keys and values
#         '''
#         attn = torch.matmul(self.keys, torch.t(key))  # 移除 .detach()
#         attn = F.softmax(attn, dim=-1)
#
#         # 计算新的keys和values
#         new_keys = torch.matmul(attn, key)  # 移除 .detach()
#         new_keys = F.normalize(new_keys, dim=1)
#
#         new_values = torch.matmul(attn, query)  # 移除 .detach()
#         new_values = F.normalize(new_values, dim=1)
#
#         # 使用动量更新
#         self.keys.data = self.momentum * self.keys.data + (1 - self.momentum) * new_keys
#         self.values.data = self.momentum * self.values.data + (1 - self.momentum) * new_values



    def read(self, key):#定义记忆读取函数，输入参数为键(key)
        '''
        query (initial features) : (NxL) x C or N x C -> T x C
        read memory items and get new robust features, 
        while memory items(cluster centers) being fixed 
        '''
        #表示输入键（所提取的趋势或季节特征表示）与记忆键的相似度得分
        attn = torch.matmul(key, torch.t(self.keys))  # (TxC) x (CxM) -> TxM
        weights = torch.softmax(attn, dim=-1)  #在最后一个维度(记忆单元维度)上进行softmax

#稀疏损失，即仅用少量相似度高的value便可重构出输入的特征表示，抑制异常被很好地重构
        entropy_loss = self.entropy_loss(weights)

#根据注意力权重读取记忆值（也就是自适应模块的value值）
        retrieved_query = torch.matmul(weights, self.values)
        # read_query = torch.cat((query, add_memory), dim=1)  # T x 2C  将原始查询和读取的记忆信息拼接
        return retrieved_query, entropy_loss

    def forward(self,key,query,mode):
        '''
        query (encoder output features) : B x L x C
        '''
        ##  query是编码器输出特征，形状为B×L×C（批次×序列长度×特征维度）
        s = key.data.shape
        key = key.contiguous().view(-1, s[-1])  # N x L x C or N x C -> T x C
        # Normalized encoder output features
        key = F.normalize(key, dim=-1)

        query = query.contiguous().view(-1, s[-1])  # N x L x C or N x C -> T x C
        # Normalized encoder output features
        query = F.normalize(query, dim=-1)


        #if mode == 'train':
            # self.update(key,query)
        gathering_loss, contrast_loss= self.gatheringLoss(key, query)


        read_query, entropy_loss = self.read(key)

        return read_query.view(s[0], s[1], s[2]), entropy_loss, gathering_loss, contrast_loss, self.keys, self.values