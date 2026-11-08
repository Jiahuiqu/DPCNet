import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import numpy as np
from functools import partial


# -----------------------------
# VQVAE Components (Adjusted for hyperspectral)
# -----------------------------

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.8):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.embed = nn.Parameter(torch.randn(dim, n_embed))
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', self.embed.clone())

        # 添加步骤计数器
        self.register_buffer('step_counter', torch.tensor(0))
        # 设置监控间隔
        self.monitor_interval = 500  # 每100步监控一次

    def forward(self, input, use_st_estimator=True):
        # [B, C, H, W] -> [B, H, W, C]
        input = input.permute(0, 2, 3, 1).contiguous()
        flatten = input.reshape(-1, self.dim)

        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )

        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1))

        if self.training:  # 仅在训练阶段且需要时使用直通估计器
            # 更新步骤计数器
            self.step_counter += 1

            # EMA更新聚类中心：通过指数移动平均更新簇大小和嵌入向量的均值
            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot.sum(0), alpha=1 - self.decay
            )
            embed_sum = flatten.T @ embed_onehot  # 计算嵌入向量的加权和
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            # 对簇大小进行归一化处理，防止数值不稳定
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + 1e-5) / (n + self.n_embed * 1e-5) * n

            # 更新最终的嵌入向量
            self.embed.data.copy_(self.embed_avg / cluster_size.unsqueeze(0))

            # 每100步监控一次码本使用情况
            if self.step_counter % self.monitor_interval == 0:
                # 计算码本使用率
                usage = (self.cluster_size > 0.01).float().mean()
                print(f"📊 Codebook usage: {usage.item() * 100:.1f}%")

                # 计算码本分布
                min_val = self.cluster_size.min().item()
                max_val = self.cluster_size.max().item()

                # 确保有足够的范围来计算直方图
                if max_val - min_val > 1e-5:
                    hist = torch.histc(self.cluster_size, bins=10, min=min_val, max=max_val)
                    # print(f"📊 Code distribution: {hist.cpu().numpy()}")
                else:
                    print("⚠️ Code distribution: All values are similar")

                # 输出最常用和最不常用的码本索引
                top_5_indices = torch.topk(self.cluster_size, 5).indices
                bottom_5_indices = torch.topk(self.cluster_size, 5, largest=False).indices
                # print(f"🔝 Top 5 codes: {top_5_indices.cpu().numpy()}")
                # print(f"🔚 Bottom 5 codes: {bottom_5_indices.cpu().numpy()}")

            if use_st_estimator:
                quantize = input + (quantize - input).detach()
        else:
            # 非训练模式不使用直通估计器
            if use_st_estimator:
                quantize = input + (quantize - input).detach()

        # 使用直通估计器（Straight-Through Estimator）保持梯度传播
        # quantize = input + (quantize - input).detach()
        quantize = quantize.permute(0, 3, 1, 2)  # 将维度从 [B, H, W, C] 转换为 [B, C, H, W]

        # 计算量化损失（commitment loss），衡量量化前后差异
        diff = (quantize.detach() - input.permute(0, 3, 1, 2)).pow(2).mean()
        if diff > 1000:
            diff_tensor = quantize.detach() - input.permute(0, 3, 1, 2)
            diff = diff_tensor.pow(2).mean()

            max_diff = diff_tensor.abs().max().item()
            if max_diff > 1.0:  # 设置一个合理阈值
                print(f"⚠️ Large diff detected: {max_diff}")
                print("Input range:", input.min().item(), "-", input.max().item())
                print("Quantize range:", quantize.min().item(), "-", quantize.max().item())

        return quantize, diff, embed_ind

    def extend_codebook(self, n_new_embeddings):
        """扩展码本，添加新的嵌入向量"""
        device = self.embed.device
        dim = self.dim

        # 创建新的嵌入向量（随机初始化）
        new_embed = torch.randn(dim, n_new_embeddings, device=device) * 0.02

        # 扩展码本
        self.embed = nn.Parameter(torch.cat([self.embed, new_embed], dim=1))

        # 扩展缓冲区
        new_cluster_size = torch.zeros(n_new_embeddings, device=device)
        self.cluster_size = torch.cat([self.cluster_size, new_cluster_size])

        new_embed_avg = torch.zeros(dim, n_new_embeddings, device=device)
        self.embed_avg = torch.cat([self.embed_avg, new_embed_avg], dim=1)

        # 更新码本大小
        self.n_embed += n_new_embeddings

        print(f"✅ 码本已扩展: 新增 {n_new_embeddings} 个码向量, 总大小: {self.n_embed}")


class Encoder(nn.Module):
    """Hyperspectral Encoder (11x11 input)"""

    def __init__(self, in_channels, channels, n_resblock=2, embed_dim=64):
        super().__init__()
        blocks = [
            nn.Conv2d(in_channels, channels // 2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels, 3, stride=1, padding=1),
            nn.ReLU(),
        ]
        for _ in range(n_resblock):
            blocks.append(ResBlock(channels))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv2d(channels, embed_dim, 1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class Decoder(nn.Module):
    """Hyperspectral Decoder (11x11 output)"""

    def __init__(self, out_channels, embed_dim, channels, n_resblock=2):
        super().__init__()
        blocks = [nn.Conv2d(embed_dim, channels, 1)]
        for _ in range(n_resblock):
            blocks.append(ResBlock(channels))
        blocks += [
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1)
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class ResBlock(nn.Module):
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


class VQVAE(nn.Module):
    """VQVAE for Hyperspectral Images (11x11)"""

    def __init__(self, in_channels, channels=256, n_resblock=2, embed_dim=64, n_embed=512):
        super().__init__()
        self.encoder = Encoder(in_channels, channels, n_resblock, embed_dim)
        self.quantize = Quantize(embed_dim, n_embed)
        self.decoder = Decoder(in_channels, embed_dim, channels, n_resblock)
        self.embed_dim = embed_dim

    def forward(self, x, use_st_estimator=True):  # 添加参数
        z_e = self.encoder(x)
        z_q, diff, _ = self.quantize(z_e, use_st_estimator=use_st_estimator)  # 传递参数
        recon = self.decoder(z_q)
        return recon, diff, _

    def encode(self, x):
        z_e = self.encoder(x)
        _, _, indices = self.quantize(z_e)
        return indices

    def decode(self, indices, h=11, w=11):
        quantize = F.embedding(indices, self.quantize.embed.transpose(0, 1))
        quantize = quantize.permute(0, 3, 1, 2)  # [B, C, H, W]
        return self.decoder(quantize)

    def get_quantized(self, x):
        z_e = self.encoder(x)
        # z_q, _, _ = self.quantize(z_e)
        return z_e

    def encode_continuous(self, x):
        """返回连续潜在表示（不量化）"""
        return self.encoder(x)

    def quantize_continuous(self, z):
        """量化连续表示"""
        z_q, _, _ = self.quantize(z)
        return z_q

    def decode_continuous(self, z_continuous):
        """从连续表示解码（先量化再解码）"""
        z_q = self.quantize_continuous(z_continuous)
        return self.decoder(z_q)

    def extend_codebook(self, n_new_embeddings):
        """扩展码本"""
        self.quantize.extend_codebook(n_new_embeddings)

# -----------------------------
# DiT Components (Latent Space)
# -----------------------------

def modulate(x, scale, shift):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, nfreq=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nfreq, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.nfreq = nfreq

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.nfreq)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes + 1, dim)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding(labels)
        return embeddings


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):  # 增大eps防止除零
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        # 添加安全除法
        return x / torch.clamp(norm, min=self.eps) * self.scale * self.g


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # [B, E, grid, grid]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=True, attn_drop=0., proj_drop=0., norm_layer=RMSNorm):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = torch.matmul(attn, v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer() if act_layer is not None else nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=RMSNorm)
        self.norm2 = RMSNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_dim, act_layer=approx_gelu, drop=0
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), scale_msa, shift_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), scale_mlp, shift_mlp)
        )
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_dim):
        super().__init__()
        self.norm_final = RMSNorm(dim)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiTModel(nn.Module):
    def __init__(
            self,
            in_channels,  # 输入通道数 (潜在空间维度)
            img_size=11,  # 图像大小
            patch_size=1,  # 补丁大小
            dim=256,  # Transformer维度
            depth=4,  # DiT块数量
            num_heads=8,  # 注意力头数
            mlp_ratio=4.0,
            num_classes=10,  # 条件类别数
    ):
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.dim = dim  # 添加这行：存储dim参数

        # DiT组件
        self.x_embedder = PatchEmbed(img_size, patch_size, in_channels, dim)
        self.t_embedder = TimestepEmbedder(dim)
        self.r_embedder = TimestepEmbedder(dim)

        self.use_cond = num_classes is not None
        self.y_embedder = LabelEmbedder(num_classes, dim) if self.use_cond else None

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(dim, patch_size, in_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # 初始化transformer层
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # 位置编码
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # 初始化补丁嵌入
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # 标签嵌入
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding.weight, std=0.02)

        # 时间步嵌入MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # 零初始化调制层
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # 零初始化输出层
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.normal_(self.final_layer.linear.weight, std=0.02)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.in_channels
        p = self.patch_size
        h = w = int(math.sqrt(x.shape[1]))
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t, r, y=None):
        """前向传播
        Args:
            x: [B, C, H, W] 潜在空间特征
            t: [B,] 时间步
            r: [B,] 参考时间步
            y: [B,] 条件标签
        """
        # 处理DiT
        x = self.x_embedder(x) + self.pos_embed  # [B, num_patches, D]
        t = self.t_embedder(t)  # [B, D]
        r = self.r_embedder(r)  # [B, D]
        c = t + r  # 组合条件

        if self.use_cond and y is not None:
            y_emb = self.y_embedder(y)  # [B, D]
            c = c + y_emb  # 添加类别条件

        for block in self.blocks:
            x = block(x, c)  # [B, num_patches, D]

        x = self.final_layer(x, c)  # [B, num_patches, patch_size*patch_size*C]
        x = self.unpatchify(x)  # [B, C, H, W] (潜在空间)
        return x


class VQDiT(nn.Module):
    """VQDiT for Hyperspectral Image Generation"""

    def __init__(
            self,
            in_channels,  # 光谱波段数
            latent_dim=64,  # VQVAE潜在维度
            img_size=11,  # 固定大小11x11
            patch_size=1,  # 使用1x1补丁
            dim=256,  # Transformer维度
            depth=4,  # DiT块数量
            num_heads=8,  # 注意力头数
            mlp_ratio=4.0,
            num_classes=10,  # 条件类别数
            n_embed=512,  # VQVAE码本大小
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes

        # VQVAE组件
        self.vqvae = VQVAE(in_channels, embed_dim=latent_dim, n_embed=n_embed)

        # DiT组件 (封装为独立模块)
        self.dit_model = DiTModel(
            in_channels=latent_dim,  # DiT输入维度是潜在空间维度
            img_size=img_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes
        )

    def forward(self, x, t, r, y=None):
        """前向传播
        Args:
            x: [B, C, H, W] 高光谱图像
            t: [B,] 时间步
            r: [B,] 参考时间步
            y: [B,] 条件标签
        """
        # 通过DiT模型处理
        return self.dit_model(x, t, r, y)

    def encode_continuous(self, x):
        """编码到连续潜在空间"""
        return self.vqvae.encode_continuous(x)

    def quantize_continuous(self, z):
        """量化连续表示"""
        return self.vqvae.quantize_continuous(z)

    def decode_continuous(self, z_continuous):
        """从连续表示解码"""
        return self.vqvae.decode_continuous(z_continuous)

    def decode(self, indices, h=11, w=11):
        """从离散索引解码"""
        return self.vqvae.decode(indices, h, w)

    def train_vqvae(self, mode=True):
        """设置VQVAE训练模式"""
        self.vqvae.train(mode)

        # 确保所有VQVAE参数可训练
        for param in self.vqvae.parameters():
            param.requires_grad = True  # 确保梯度计算启用

        # 训练DiT时冻结VQVAE参数
        if not mode:
            for param in self.vqvae.parameters():
                param.requires_grad = False

    def train_dit(self, mode=True):
        """设置DiT训练模式"""
        # 只训练DiT部分
        self.dit_model.train(mode)

        # 确保VQVAE被冻结
        if mode:
            self.train_vqvae(False)


    def test_vqvae(self, mode=False):
        """设置VQVAE训练模式"""
        self.vqvae.train(mode)
        # 训练DiT时冻结VQVAE参数
        for param in self.vqvae.parameters():
            param.requires_grad = False

    def test_dit(self, mode=False):
        """设置DiT训练模式"""
        # 只训练DiT部分
        self.dit_model.train(mode)
        self.vqvae.train(mode)
        # 训练DiT时冻结VQVAE参数
        for param in self.vqvae.parameters():
            param.requires_grad = False

    def extend_codebook(self, n_new_embeddings):
        """扩展码本"""
        self.vqvae.extend_codebook(n_new_embeddings)


import copy  # 添加copy模块


class VQDiTIncremental_2(nn.Module):
    def __init__(
            self,
            base_model,  # 基础模型
            in_channels,
            img_size=11,
            patch_size=1,
            dim=256,
            depth=4,
            num_heads=8,
            mlp_ratio=4.0,
            num_classes=10,
            n_embed=512,  # 新码本大小
    ):
        super().__init__()
        # 关键修改：使用基础模型的潜在维度
        self.latent_dim = base_model.latent_dim

        # 使用基础模型的潜在维度创建新 VQVAE
        self.vqvae = VQVAE(in_channels, embed_dim=self.latent_dim, n_embed=n_embed)

        # 继承权重（现在维度匹配）
        self._inherit_vqvae_weights(base_model.vqvae)

        # 创建新 DiT，输入维度与基础模型一致
        self.dit_model = DiTModel(
            in_channels=self.latent_dim,  # 使用基础模型的维度
            img_size=img_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes
        )

    def _inherit_vqvae_weights(self, base_vqvae):
        """从基础模型继承VQVAE的编码器和解码器权重"""
        # 1. 继承编码器权重
        self.vqvae.encoder.load_state_dict(base_vqvae.encoder.state_dict())

        # 2. 继承解码器权重
        self.vqvae.decoder.load_state_dict(base_vqvae.decoder.state_dict())

        # 3. 冻结继承的权重（可选）
        for param in self.vqvae.encoder.parameters():
            param.requires_grad = False
        for param in self.vqvae.decoder.parameters():
            param.requires_grad = False

        # 4. 保持量化器可训练（全新初始化）
        print("✅ 继承VQVAE编码器和解码器参数，保持量化器可训练")

    def forward(self, x, t, r, y=None):
        return self.dit_model(x, t, r, y)

    def encode_continuous(self, x):
        return self.vqvae.encode_continuous(x)

    def quantize_continuous(self, z):
        return self.vqvae.quantize_continuous(z)

    def decode_continuous(self, z_continuous):
        return self.vqvae.decode_continuous(z_continuous)

    def decode(self, indices, h=11, w=11):
        return self.vqvae.decode(indices, h, w)

    def reconstruct(self, x):
        """VQVAE重建"""
        z_q, diff, _ = self.vqvae(x)
        return z_q, diff

    def train_vqvae(self, mode=True):
        """只训练量化器，冻结编码器和解码器"""
        # 量化器始终可训练
        self.vqvae.quantize.train(mode)

        # 编码器和解码器始终冻结
        self.vqvae.encoder.eval()
        self.vqvae.decoder.eval()
        for param in self.vqvae.encoder.parameters():
            param.requires_grad = True
        for param in self.vqvae.decoder.parameters():
            param.requires_grad = True

    def train_dit(self, mode=True):
        """训练DiT部分"""
        self.dit_model.train(mode)
        # 确保VQVAE被冻结（只允许量化器训练）
        self.train_vqvae(False)

    def test_vqvae(self, mode=False):
        self.vqvae.quantize.eval()
        self.vqvae.encoder.eval()
        self.vqvae.decoder.eval()

    def test_dit(self, mode=False):
        self.dit_model.eval()



class VQDiTIncremental(nn.Module):
    """增量学习专用模型，仅处理新增类别"""

    def __init__(self, base_model, n_new_embeddings, new_classes, args):
        super().__init__()
        # 存储配置参数
        self.in_channels = args.in_channels
        self.new_classes = new_classes
        self.num_new_classes = len(new_classes)
        self.device = next(base_model.parameters()).device

        # 1. 复制基础VQVAE的编码器和解码器（共享组件）
        self.vqvae_encoder = copy.deepcopy(base_model.vqvae.encoder)
        self.vqvae_decoder = copy.deepcopy(base_model.vqvae.decoder)

        # 2. 创建新的独立码本
        self.new_quantize = Quantize(
            dim=base_model.latent_dim,
            n_embed=n_new_embeddings
        )

        # 3. 创建全新的DiT模型（仅用于新增类别）
        self.dit_model = DiTModel(
            in_channels=base_model.latent_dim,
            img_size=base_model.img_size,
            patch_size=1,
            dim=args.dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            num_classes=self.num_new_classes  # 只识别新增类别
        )

        # 将模型移动到设备
        self.to(self.device)
        print(f"✅ 增量模型初始化完成: 新增 {self.num_new_classes} 个类别, {n_new_embeddings} 个码向量")

    def forward(self, x, t, r, y=None):
        """前向传播
        Args:
            x: [B, C, H, W] 高光谱图像
            t: [B,] 时间步
            r: [B,] 参考时间步
            y: [B,] 条件标签
        """
        # 通过DiT模型处理
        return self.dit_model(x, t, r, y)

    def reconstruct(self, x):
        """VQVAE重建"""
        z_e = self.encode_continuous(x)
        z_q, diff = self.quantize_continuous(z_e)
        return self.decode_continuous(z_q), diff

    def encode_continuous(self, x):
        """编码到连续潜在空间"""
        return self.vqvae_encoder(x)

    def quantize_continuous(self, z):
        """使用新码本量化"""
        z_q, diff, _ = self.new_quantize(z)
        return z_q, diff

    def decode_continuous(self, z_continuous):
        """从连续表示解码"""
        return self.vqvae_decoder(z_continuous)

    def train_vqvae(self, mode=True):
        """设置VQVAE训练模式"""
        self.vqvae_encoder.train(mode)
        self.vqvae_decoder.train(mode)
        self.new_quantize.train(mode)

    def train_dit(self, mode=True):
        """设置DiT训练模式"""
        self.dit_model.train(mode)
        if mode:
            self.vqvae_encoder.train(False)
            self.vqvae_decoder.train(False)
            self.new_quantize.train(False)

    def test_vqvae(self):
        """设置VQVAE训练模式"""
        self.vqvae_encoder.train(False)
        self.vqvae_decoder.train(False)
        self.new_quantize.train(False)

    def test_dit(self):
        """设置DiT训练模式"""
        self.dit_model.train(False)
        self.vqvae_encoder.train(False)
        self.vqvae_decoder.train(False)
        self.new_quantize.train(False)


# Positional embedding utilities
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb



# -----------------------------
# MeanFlow for Latent Space
# -----------------------------

class LatentNormalizer:
    def __init__(self, mode='mean_std', mean=None, std=None):
        assert mode in ['minmax', 'mean_std']
        self.mode = mode
        self.mean = torch.tensor(mean).view(-1, 1, 1) if mean is not None else None
        self.std = torch.tensor(std).view(-1, 1, 1) if std is not None else None

    @classmethod
    def from_list(cls, config):
        mode, mean, std = config
        return cls(mode, mean, std)

    def norm(self, x):
        if self.mode == 'minmax':
            return x * 2 - 1
        elif self.mode == 'mean_std':
            return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def unnorm(self, x):
        if self.mode == 'minmax':
            return (x + 1) * 0.5
        elif self.mode == 'mean_std':
            return x * self.std.to(x.device) + self.mean.to(x.device)


def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq
    return (stopgrad(w) * loss).mean()


class LatentMeanFlow:
    """
        初始化 VQVAE 模型的参数和配置。

        参数:
        - latent_dim: VQVAE 潜在空间的维度。
        - image_size: 高光谱图像的尺寸。
        - num_classes: 分类任务的类别数量。
        - normalizer: 一个列表，描述潜在空间的归一化方法，包括归一化类型和参数。
        - flow_ratio: 模型中使用的流的比例。
        - time_dist: 一个列表，描述时间分布，包括分布类型和参数。
        - cfg_ratio: 分类器引导的比例。
        - cfg_scale: 分类器引导的缩放因子。
        - cfg_uncond: 无条件引导的类型。
        - jvp_api: 用于计算雅可比向量积的 API，可以是 'funtorch' 或 'autograd'。
        """
    def __init__(
            self,
            latent_dim=64,
            image_size=11,
            num_classes=10,
            normalizer=['minmax', None, None],
            flow_ratio=0.50,
            time_dist=['lognorm', -0.4, 1.0],
            cfg_ratio=0.1,
            cfg_scale=2.0,
            cfg_uncond='u',
            jvp_api='autograd',
    ):
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.num_classes = num_classes
        self.normer = LatentNormalizer.from_list(normalizer)
        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        self.cfg_ratio = cfg_ratio
        self.w = cfg_scale
        self.cfg_uncond = cfg_uncond
        self.jvp_api = jvp_api

        assert jvp_api in ['funtorch', 'autograd'], "jvp_api must be 'funtorch' or 'autograd'"
        if jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        elif jvp_api == 'autograd':
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True


    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)
        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))

        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

    def loss(self, model, x, c=None):
        batch_size = x.shape[0]
        device = x.device
        t, r = self.sample_t_r(batch_size, device)

        t_ = t.view(-1, 1, 1, 1)
        r_ = r.view(-1, 1, 1, 1)

        e = torch.randn_like(x)
        # x_norm = self.normer.norm(x)  # 归一化后的特征
        x_norm = x  # 归一化后的特征

        z = (1 - t_) * x_norm + t_ * e  # 加噪特征
        v = e - x_norm  # 真实噪声方向

        if c is not None:
            assert self.cfg_ratio is not None
            uncond = torch.ones_like(c) * self.num_classes
            cfg_mask = torch.rand_like(c.float()) < self.cfg_ratio
            c = torch.where(cfg_mask, uncond, c)
            if self.w is not None:
                with torch.no_grad():
                    u_t = model(z, t, t, uncond)
                v_hat = self.w * v + (1 - self.w) * u_t
                if self.cfg_uncond == 'v':
                    cfg_mask = rearrange(cfg_mask, "b -> b 1 1 1").bool()
                    v_hat = torch.where(cfg_mask, v, v_hat)
            else:
                v_hat = v

        model_partial = partial(model, y=c)
        jvp_args = (
            lambda z, t, r: model_partial(z, t, r),
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
        )

        # 计算前向传播和导数
        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)

        # 计算目标值和误差
        delta_t = (t_ - r_).clamp(min=1e-3)  # 防止除零
        u_tgt = v_hat - delta_t * dudt
        error = u - stopgrad(u_tgt)
        loss = adaptive_l2_loss(error)

        # 计算均方误差
        mse_val = (stopgrad(error) ** 2).mean()

        if math.isnan(loss) or math.isnan(mse_val):
            print(f"NaN detected at step . Stopping training.")

        # 计算去噪后的特征
        with torch.no_grad():
            # 使用预测的方向向量进行去噪
            denoised_z_norm = z - t_ * u

            # 反归一化得到原始尺度的潜在特征
            # denoised_z = self.normer.unnorm(denoised_z_norm)

        return loss, mse_val, denoised_z_norm  # 返回去噪后的特征

    def loss_increment(self, model, x, c=None, new_classes=None):
        """
        增量训练阶段的专用损失函数
        参数:
            model: VQDiT模型
            x: 输入图像 [B, C, H, W]
            c: 类别标签 [B] (可选)
            new_classes: 新增类别的列表 (如 [6,7,8])
        """
        # 1. 验证新增类别
        if new_classes is None:
            raise ValueError("增量训练必须提供 new_classes 参数")
        if not isinstance(new_classes, list):
            new_classes = list(new_classes)

        # 2. 创建标签映射
        class_mapping = {cls: idx for idx, cls in enumerate(new_classes)}
        num_new_classes = len(new_classes)
        uncond_label = num_new_classes  # 无条件生成的标签

        # 3. 映射类别标签（如果提供）
        if c is not None:
            # 创建映射后的标签张量
            c_mapped = torch.zeros_like(c)
            for i in range(c.size(0)):
                original_label = c[i].item()
                # 映射到从0开始的索引
                if original_label in class_mapping:
                    c_mapped[i] = class_mapping[original_label]
                else:
                    # 如果不是新增类别，设为无条件标签
                    c_mapped[i] = uncond_label
            c = c_mapped
        else:
            # 如果没有提供标签，全部设为无条件生成
            c = torch.full((x.size(0),), uncond_label, device=x.device, dtype=torch.long)

        # 4. 临时调整无条件标签（仅用于损失计算）
        original_num_classes = self.num_classes
        self.num_classes = num_new_classes  # +1 用于无条件生成

        # 5. 执行与原始loss方法完全相同的计算过程
        batch_size = x.shape[0]
        device = x.device
        t, r = self.sample_t_r(batch_size, device)

        t_ = t.view(-1, 1, 1, 1)
        r_ = r.view(-1, 1, 1, 1)

        e = torch.randn_like(x)
        # x_norm = self.normer.norm(x)  # 归一化后的特征
        x_norm = x  # 归一化后的特征

        z = (1 - t_) * x_norm + t_ * e  # 加噪特征
        v = e - x_norm  # 真实噪声方向

        if c is not None:
            assert self.cfg_ratio is not None
            uncond = torch.ones_like(c) * self.num_classes
            cfg_mask = torch.rand_like(c.float()) < self.cfg_ratio
            c = torch.where(cfg_mask, uncond, c)
            if self.w is not None:
                with torch.no_grad():
                    u_t = model(z, t, t, uncond)
                v_hat = self.w * v + (1 - self.w) * u_t
                if self.cfg_uncond == 'v':
                    cfg_mask = rearrange(cfg_mask, "b -> b 1 1 1").bool()
                    v_hat = torch.where(cfg_mask, v, v_hat)
            else:
                v_hat = v

        model_partial = partial(model, y=c)
        jvp_args = (
            lambda z, t, r: model_partial(z, t, r),
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
        )

        # 计算前向传播和导数
        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)

        # 计算目标值和误差
        delta_t = (t_ - r_).clamp(min=1e-3)  # 防止除零
        u_tgt = v_hat - delta_t * dudt
        error = u - stopgrad(u_tgt)
        loss = adaptive_l2_loss(error)

        # 计算均方误差
        mse_val = (stopgrad(error) ** 2).mean()

        if math.isnan(loss) or math.isnan(mse_val):
            print(f"NaN detected at step . Stopping training.")

        # 计算去噪后的特征
        with torch.no_grad():
            # 使用预测的方向向量进行去噪
            denoised_z_norm = z - t_ * u

            # 反归一化得到原始尺度的潜在特征
            # denoised_z = self.normer.unnorm(denoised_z_norm)

        # 恢复原始的无条件标签
        self.num_classes = original_num_classes

        return loss, mse_val, denoised_z_norm

    @torch.no_grad()
    def sample_each_class(self, model, n_per_class, classes=None, device='cuda'):
        """
        高光谱图像采样函数（在潜在空间）

        参数:
            model: VQDiT模型
            n_per_class: 每个类别的采样数量
            classes: 要采样的类别列表，如果为None，则对所有类别采样
            device: 设备

        返回:
            采样得到的潜在特征 [B, latent_dim, 11, 11]
        """
        model.eval().to(device)

        if classes is None:
            classes = range(self.num_classes)
        c = torch.tensor(classes, device=device).repeat_interleave(n_per_class)
        batch_size = len(c)

        # 生成符合高光谱尺寸的噪声（潜在空间）
        z = torch.randn(
            batch_size, self.latent_dim,
            self.image_size, self.image_size,
            device=device
        )

        # 使用单步反演采样
        t = torch.ones((batch_size,), device=device)  # 最大时间步
        r = torch.zeros((batch_size,), device=device)  # 最小时间步

        # 模型预测并更新z
        z = z - model(z, t, r, c)

        return z

    @torch.no_grad()
    def increment_sample(self, model, n_per_class, classes=None, device='cuda'):
        """
        增量模型专用采样函数 - 将新类别映射到从0开始的索引

        参数:
            n_per_class: 每个类别的采样数量
            classes: 要采样的新类别列表（如[6,7,8]）
            device: 设备

        返回:
            采样得到的潜在特征 [B, latent_dim, 11, 11]
        """
        model.eval().to(device)

        # 映射新类别到从0开始的索引
        if classes is None:
            classes = self.new_classes

        # 创建映射表：原始类别 -> 新索引
        class_mapping = {cls: idx for idx, cls in enumerate(classes)}
        mapped_classes = [class_mapping[cls] for cls in classes]

        # 准备类别张量
        c = torch.tensor(mapped_classes, device=device).repeat_interleave(n_per_class)
        batch_size = len(c)


        # 生成符合高光谱尺寸的噪声（潜在空间）
        z = torch.randn(
            batch_size, self.latent_dim,
            self.image_size, self.image_size,
            device=device
        )

        # 使用单步反演采样
        t = torch.ones((batch_size,), device=device)  # 最大时间步
        r = torch.zeros((batch_size,), device=device)  # 最小时间步

        # 模型预测并更新z
        z = z - model(z, t, r, c)

        return z