import torch
from torch import nn
from torch.nn import functional as F


class Quantize(nn.Module):
    """向量量化层"""

    def __init__(self, dim, n_embed, decay=0.99):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay

        # 初始化 codebook
        self.embed = nn.Parameter(torch.randn(dim, n_embed))
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', self.embed.clone())

    def forward(self, input):
        # 将输入展平为 (B*H*W, dim)
        flatten = input.reshape(-1, self.dim)

        # 计算与codebook的距离
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )

        # 获取最近邻编码索引
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])  # (B, H, W)

        # 量化操作
        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1))

        # 训练时更新codebook
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot.sum(0), alpha=1 - self.decay
            )
            embed_sum = flatten.T @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + 1e-5) / (n + self.n_embed * 1e-5) * n
            self.embed.data.copy_(self.embed_avg / cluster_size.unsqueeze(0))

        # 直通估计器
        quantize = input + (quantize - input).detach()
        diff = (quantize.detach() - input).pow(2).mean()
        return quantize, diff, embed_ind


class Encoder(nn.Module):
    """单层编码器 (原enc_b)"""

    def __init__(self, in_channel=3, channel=128, n_resblock=2):
        super().__init__()
        blocks = [
            nn.Conv2d(in_channel, channel // 2, 3, stride=1, padding=1),  # 128x128
            nn.ReLU(),
            nn.Conv2d(channel // 2, channel, 3, stride=1, padding=1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, padding=1)
        ]
        for _ in range(n_resblock):
            blocks.append(ResBlock(channel))
        blocks.append(nn.ReLU())
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class Decoder(nn.Module):
    """单层解码器"""

    def __init__(self, out_channel=3, channel=128, n_resblock=2):
        super().__init__()
        blocks = [nn.Conv2d(channel, channel, 3, padding=1)]
        for _ in range(n_resblock):
            blocks.append(ResBlock(channel))
        blocks += [
            nn.ReLU(),
            nn.ConvTranspose2d(channel, channel // 2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(channel // 2, out_channel, 3, stride=1, padding=1)
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class ResBlock(nn.Module):
    """残差块"""

    def __init__(self, channel):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 1)
        )

    def forward(self, x):
        return x + self.block(x)


class SingleLayerVQVAE(nn.Module):
    """单层VQ-VAE"""

    def __init__(self,
                 in_channel=3,
                 channel=128,
                 n_resblock=2,
                 embed_dim=64,
                 n_embed=512):
        super().__init__()

        # 编码器
        self.encoder = Encoder(in_channel, channel, n_resblock)

        # 量化前卷积
        self.quant_conv = nn.Conv2d(channel, embed_dim, 1)

        # 量化层
        self.quantize = Quantize(embed_dim, n_embed)

        # 解码器
        self.decoder = Decoder(in_channel, embed_dim, n_resblock)

    def forward(self, x):
        # 编码
        z = self.encoder(x)

        # 量化
        z = self.quant_conv(z).permute(0, 2, 3, 1)  # [B, D, H, W] -> [B, H, W, D]
        quant, diff, idx = self.quantize(z)
        quant = quant.permute(0, 3, 1, 2)  # [B, H, W, D] -> [B, D, H, W]

        # 解码
        recon = self.decoder(quant)
        return recon, diff, idx


# 示例用法
if __name__ == "__main__":
    model = SingleLayerVQVAE()
    x = torch.randn(2, 3, 256, 256)  # 输入图像
    recon, diff, idx = model(x)
    print("Input shape:", x.shape)  # [2, 3, 256, 256]
    print("Latent code shape:", idx.shape)  # [2, 64, 64] (压缩32倍)
    print("Recon shape:", recon.shape)  # [2, 3, 256, 256]
