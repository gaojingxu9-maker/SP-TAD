import math
from .Disentangler import TrendFeatureDisentangler, SeasonalFeatureDisentangler
from .STEncoder import STEncoder
from .attn import AnomalyAttention
from .embed import DataEmbedding
from .encoder import *

from .attn import AttentionLayer
from .ours_memory_module import MemoryModule

class Encoder_ts(nn.Module):#时间序列编码器的初始化函数
    def __init__(self, in_dim):
        super(Encoder_ts, self).__init__()
        self.in_dim = in_dim
        self.main = nn.Sequential(

            nn.Conv1d(self.in_dim, 128 , 7, 1, 3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(128, 256, 7, 1, 3),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv1d(256, 512, 7, 1, 3),
            # nn.BatchNorm1d(512),
            # nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, input):
        output = self.main(input)

        return output

class Decoder(nn.Module):
    def __init__(self, d_model, c_out):
        super(Decoder, self).__init__()
        self.out_linear = nn.Linear(d_model, c_out)
    def forward(self, x):
        out = self.out_linear(x)
        return out      # N x L x c_out

class Model(nn.Module):
    def __init__(self, win_size,enc_in, c_out,d_model,device,
                 n_memory=5, mode = 'train', dataset = None,\
                 n_heads = 8, dropout = 0.0, activation = 'gelu', \
                 e_layers = 3, d_ff = 512):
        super(Model, self).__init__()
        # Encoder
        self.d_model = d_model
        self.device = device
        self.win_size = win_size
        self.mode = mode
        self.num_experts = math.floor(math.log2(win_size / 2)) + 1
        self.tfd = TrendFeatureDisentangler(enc_in, self.num_experts, 128)
        self.sfd = SeasonalFeatureDisentangler(enc_in,128, self.win_size)
        # self.encoder =STEncoder(input_dims=enc_in,device=self.device)

        self.embedding = DataEmbedding(enc_in, self.d_model, dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(self.win_size, False, attention_dropout=dropout),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.enc_t =nn.Linear(128, 256)
        self.enc_s =nn.Linear(128, 256)

        self.trendMemory = MemoryModule(n_memory, self.d_model, self.device, dataset_name=dataset)
        self.seasonalMemory =MemoryModule(n_memory, self.d_model, self.device, dataset_name=dataset)

        self.weak_decoder = Decoder(2* d_model, c_out)

        self.criterion = nn.MSELoss()


    def forward(self,x, mode='test'):
        '''
        x (input time window) : N x L x enc_in
        '''
        trend = self.tfd(x.permute(0,2,1))# 提取时间序列的趋势成分，输入需要转置为(N, enc_in, L)格式
        # 将趋势特征转回(N, L, enc_in)格式并进行线性变换得到趋势表示
        trend_representation = self.enc_t(trend.permute(0,2,1))

        # 提取时间序列的季节性成分
        # 对季节性特征进行线性变换得到季节性表示
        seasonal = self.sfd(x)
        seasonal_representation = self.enc_s(seasonal)

        # 对原始输入进行数据嵌入(值嵌入+位置嵌入)
        # 通过编码器提取高级特征表示
        enc_out = self.embedding(x)
        representation = self.encoder(enc_out)

        # 使用趋势记忆模块处理趋势表示和通用表示，返回重构输出和各种损失
        # 使用季节性记忆模块处理季节性表示和通用表示，返回重构输出和各种损失
        trend_outputs, t_entropy_loss, t_gathering_loss, t_contrast_loss, t_keys, t_values = self.trendMemory(trend_representation, representation,mode)
        seasonal_outputs, s_entropy_loss, s_gathering_loss, s_contrast_loss, s_keys,s_values= self.seasonalMemory(seasonal_representation, representation,mode)

        # 将趋势输出与通用表示拼接后通过弱解码器生成趋势重构输
        # 将季节性输出与通用表示拼接后通过弱解码器生成季节性重构输出
        out_t = self.weak_decoder(torch.cat((trend_outputs,representation), dim=-1))
        out_s = self.weak_decoder(torch.cat((seasonal_outputs,representation), dim=-1))

############################ regularization ##################################
        # 从季节性重构输出中再次提取趋势成分用于正则化
        # 从季节性重构输出中再次提取季节性成分用于正则化
        s_trend = self.tfd(out_s.permute(0,2,1))
        s_seasonal = self.sfd(out_s)

        # 从趋势重构输出中再次提取趋势成分用于正则化
        # 从趋势重构输出中再次提取季节性成分用于正则化
        t_trend = self.tfd(out_t.permute(0,2,1))
        t_seasonal = self.sfd(out_t)

        # 训练模式下的损失计算
        if mode != 'test':
            # 计算趋势重构的一致性损失和季节性重构的一致性损失
            t_loss = self.criterion(t_trend.permute(0,2,1), trend.permute(0,2,1)) + self.criterion(s_trend.permute(0,2,1), trend.permute(0,2,1))
            s_loss = self.criterion(t_seasonal, seasonal) + self.criterion(s_seasonal, seasonal)

            # 返回训练模式下的所有损失
            return {"out_t":out_t,
                    "out_s":out_s,
                     "gathering_loss": t_gathering_loss + s_gathering_loss,
                    "contrast_loss": t_contrast_loss + s_contrast_loss,
                    "entropy_loss": t_entropy_loss + s_entropy_loss,
                    'regular': t_loss + s_loss }

        # 测试模式下的损失计算
        criterion_test = nn.MSELoss(reduce=False)

        # 计算趋势重构误差和季节性重构误差
        t_loss =torch.mean(criterion_test(s_trend.permute(0,2,1), trend.permute(0,2,1)), dim=-1) + torch.mean(criterion_test(t_trend.permute(0,2,1), trend.permute(0,2,1)), dim=-1)
        s_loss = torch.mean(criterion_test(t_seasonal, seasonal), dim=-1) + torch.mean(criterion_test(s_seasonal, seasonal), dim=-1)
        # 返回测试模式下的结果和误差
        return {"out_t": out_t,
                "out_s": out_s,
                "t_loss": t_loss,
                "s_loss": s_loss,
                "representation": representation,
                "t_keys": t_keys, "t_values":t_values,
                "s_keys": s_keys, "s_values": s_values,
                "t_query": trend_representation,"s_query":seasonal_representation
                }
