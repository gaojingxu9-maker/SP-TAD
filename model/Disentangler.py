import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

class TrendFeatureDisentangler(nn.Module):
    def __init__(self, input_channels, num_experts, output_channels):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            self.experts.append(
                nn.Conv1d(in_channels=input_channels, out_channels = output_channels, kernel_size=2**i)
            )


    def forward(self, x):
        expert_outputs = [expert(F.pad(x,(2**i-1,0))) for i, expert in enumerate(self.experts)]
        # Average pooling across outputs from different experts
        output = torch.mean(torch.stack(expert_outputs), dim=0)
        return output

    class TrendFeatureDisentangler(nn.Module):
        def __init__(self, input_channels, num_experts, output_channels):
            super().__init__()
            self.num_experts = num_experts
            self.experts = nn.ModuleList()

            # 添加通道适配层，将输入通道数调整为卷积层期望的值
            self.channel_adapter = nn.Conv1d(
                in_channels=input_channels,
                out_channels=38,  # 调整为卷积层期望的通道数
                kernel_size=1
            )

            for i in range(num_experts):
                self.experts.append(
                    # 修改卷积层的输入通道数为适配后的通道数
                    nn.Conv1d(in_channels=38, out_channels=output_channels, kernel_size=2 ** i)
                )

        def forward(self, x):
            # 首先调整通道数
            x = self.channel_adapter(x)
            expert_outputs = [expert(F.pad(x, (2 ** i - 1, 0))) for i, expert in enumerate(self.experts)]
            # Average pooling across outputs from different experts
            output = torch.mean(torch.stack(expert_outputs), dim=0)
            return output

class SeasonalFeatureDisentangler(nn.Module):
    def __init__(self, input_channels, output_channels, win_size):
        super(SeasonalFeatureDisentangler, self).__init__()
        self.num_frequencies = win_size // 2 + 1  # Due to real-to-complex FFT
        # Adjust dimensions based on the provided formulas for A and B
        self.freq_weights = nn.Parameter(torch.randn(self.num_frequencies, input_channels, output_channels, dtype=torch.cfloat))
        self.freq_biases = nn.Parameter(torch.randn(self.num_frequencies, output_channels, dtype=torch.cfloat))

    def forward(self, x):
        # x shape: (batch_size, win_size, input_channels)
        # Applying DFT
        freq_domain = fft.rfft(x, dim=1)
        # Applying learnable Fourier transform for each frequency
        transformed = torch.einsum('bfi,fij->bfj', [freq_domain, self.freq_weights]) + self.freq_biases
        # Applying inverse DFT
        time_domain = fft.irfft(transformed, n=x.shape[1], dim=1)

        return time_domain