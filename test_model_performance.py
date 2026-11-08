"""
DCPNet 性能测试脚本(训练时间 / 推理时间 / 参数量 / FLOPs)
=====================================================================
针对审稿意见:DCPNet 同时包含生成模型(VQVAE + DiT)和双流编码器,
本脚本为以下两大部分补充测量:

  [1] 生成模型:VQVAE(重建) + DiT(潜空间单步去噪,LatentMeanFlow/MeanFlow 训练与采样)
  [2] 双流编码器:DualPromptNetwork(共享分支 + 双私有分支 + 分类头)

测量口径:
  - 计时:先预热,再在 CUDA 上同步计时,报告单次迭代平均耗时;
  - FLOPs:优先使用 PyTorch >= 2.1 官方 FlopCounterMode(标准口径:乘加按 2 FLOPs),
    不可用时回退 thop,均不可用则记为 N/A;
  - 参数量:总参数量 + 可训练参数量;
  - DualPromptNetwork 前向自带的 figure_rand 随机增强在计时/计数时临时关闭,
    避免数据增强算子污染网络本身的耗时与 FLOPs;
  - 模型为随机初始化(指标与权重数值无关),batch 大小与输入形状打印在报告头部。

用法:
    python test_model_performance.py                    # 使用 args.py 默认参数
    python test_model_performance.py --device cuda:0 --batch_size 32
    python test_model_performance.py --batch_size 4     # 快速冒烟测试
"""

import sys
import time
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# Class_Increment.py 顶部 import 了 mamba_ssm(实际并未使用 Mamba),
# 若环境未安装该包则注入占位模块,保证本脚本可独立运行。
# ----------------------------------------------------------------------
try:
    import mamba_ssm  # noqa: F401
except Exception:
    class _MambaPlaceholder(nn.Module):
        """mamba_ssm.modules.mamba_simple.Mamba 的占位实现(未被使用)。"""

        def __init__(self, *args, **kwargs):
            super().__init__()

    _stub_ssm = types.ModuleType('mamba_ssm')
    _stub_modules = types.ModuleType('mamba_ssm.modules')
    _stub_mamba_simple = types.ModuleType('mamba_ssm.modules.mamba_simple')
    _stub_mamba_simple.Mamba = _MambaPlaceholder
    _stub_modules.mamba_simple = _stub_mamba_simple
    _stub_ssm.modules = _stub_modules
    sys.modules.setdefault('mamba_ssm', _stub_ssm)
    sys.modules.setdefault('mamba_ssm.modules', _stub_modules)
    sys.modules.setdefault('mamba_ssm.modules.mamba_simple', _stub_mamba_simple)

from args import parse_args
from image_generate import VQDiT, LatentMeanFlow
from Class_Increment import DualPromptNetwork

# ------------------------- 测量配置 -------------------------
TRAIN_WARMUP = 5      # 训练计时预热步数
TRAIN_ITERS = 30      # 训练计时步数
INFER_WARMUP = 10     # 推理计时预热步数
INFER_ITERS = 100     # 推理计时步数
REPORT_PATH = 'test_model_performance_report.txt'

_FLOP_BACKEND = None  # 缓存 FLOPs 统计后端:'flopcounter' / 'thop' / None


# ======================================================================
# 工具函数
# ======================================================================
def _sync(device):
    """CUDA 计时前同步,避免异步执行导致计时偏差。"""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _flop_backend():
    """探测可用的 FLOPs 统计后端。"""
    global _FLOP_BACKEND
    if _FLOP_BACKEND is None:
        try:
            from torch.utils.flop_counter import FlopCounterMode  # noqa: F401
            _FLOP_BACKEND = 'flopcounter'
        except Exception:
            try:
                import thop  # noqa: F401
                _FLOP_BACKEND = 'thop'
            except Exception:
                _FLOP_BACKEND = ''
    return _FLOP_BACKEND or None


def _fmt_count(n):
    return f'{n:,}'


def fmt_flops(flops, backend):
    if flops is None:
        return 'N/A(未安装 torch>=2.1 的 flop_counter,也未安装 thop)'
    return f'{flops / 1e9:.3f} GFLOPs [{backend}]'


def param_stats(model):
    """返回 (总参数量, 可训练参数量)。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def benchmark_step(step_fn, device, warmup, iters):
    """测量单步耗时,返回平均毫秒数与总秒数。"""
    for _ in range(warmup):
        step_fn()
    _sync(device)

    start = time.perf_counter()
    for _ in range(iters):
        step_fn()
    _sync(device)
    total = time.perf_counter() - start
    return total / iters * 1000.0, total


def count_flops(model, inputs):
    """
    统计一次前向的 FLOPs。inputs 为位置参数元组。
    返回 (flops, 后端名称),不可用时返回 (None, None)。
    """
    backend = _flop_backend()
    if backend is None:
        return None, None

    was_training = model.training
    model.eval()
    try:
        if backend == 'flopcounter':
            from torch.utils.flop_counter import FlopCounterMode
            counter = FlopCounterMode(display=False)
            with counter, torch.no_grad():
                model(*inputs)
            return counter.get_total_flops(), 'FlopCounterMode'
        # thop 口径为 MACs,乘 2 换算为 FLOPs
        import thop
        macs, _ = thop.profile(model, inputs=inputs, verbose=False)
        return macs * 2, 'thop (MACs x 2)'
    finally:
        if was_training:
            model.train()


# ======================================================================
# 主流程
# ======================================================================
def main():
    args = parse_args()
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    B = args.batch_size          # batch 大小
    C = args.in_channels         # 光谱波段数
    S = args.patch_size          # patch 尺寸(生成模型 img_size 与分类输入一致)
    L = args.latent_dim          # VQVAE 潜在维度
    K = args.base_classes        # 类别数(条件生成类别)

    print(f'性能测试开始: device={device}, batch_size={B}, patch={S}x{S}, '
          f'in_channels={C}, latent_dim={L}, num_classes={K}')

    # ---------------------- 公共输入张量 ----------------------
    x = torch.randn(B, C, S, S, device=device)                                # 图像空间输入
    z_shape = (B, L, S, S)
    z_img = torch.randn(*z_shape, device=device)                              # 潜在空间输入
    t = torch.ones(B, device=device)                                          # 时间步 t
    r = torch.zeros(B, device=device)                                         # 参考时间步 r
    labels_gen = torch.randint(0, K, (B,), device=device)                     # 生成条件标签
    x1 = torch.randn(B, C, S, S, device=device)                               # 双流输入 1
    x2 = torch.randn(B, C, S, S, device=device)                               # 双流输入 2
    labels_cls = torch.randint(0, K, (B,), device=device)                     # 分类标签

    lines = []
    lines.append('=' * 78)
    lines.append('DCPNet 性能测试报告')
    lines.append(f'设备: {device} | batch_size: {B} | patch: {S}x{S} | 光谱通道: {C}')
    lines.append(f'训练计时: warmup={TRAIN_WARMUP}, iters={TRAIN_ITERS} | '
                 f'推理计时: warmup={INFER_WARMUP}, iters={INFER_ITERS}')
    lines.append('-' * 78)

    # ==================================================================
    # [1] 生成模型 · VQVAE(重建)
    # ==================================================================
    vqdit = VQDiT(
        in_channels=C,
        latent_dim=L,
        img_size=S,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        num_classes=K,
        n_embed=args.n_embed,
    ).to(device)

    meanflow = LatentMeanFlow(
        latent_dim=L,
        image_size=S,
        num_classes=K,
        normalizer=['minmax', None, None],
        flow_ratio=0,
        time_dist=['lognorm', -0.4, 1.0],
        cfg_ratio=0,
        cfg_scale=2.0,
    )

    vqvae = vqdit.vqvae
    vq_total, vq_train = param_stats(vqvae)
    vq_flops, vq_backend = count_flops(vqvae, (x,))

    vqvae_opt = torch.optim.Adam(vqvae.parameters(), lr=args.lr)

    def vqvae_train_step():
        # 与 VQDiTTrainer.train_vqvae_step 一致
        vqvae.train()
        vqvae_opt.zero_grad()
        recon, diff, _ = vqvae(x)
        loss = F.mse_loss(recon, x) + 1e-4 * diff
        loss.backward()
        vqvae_opt.step()

    def vqvae_infer_step():
        vqvae.eval()
        with torch.no_grad():
            vqvae(x)

    vq_train_ms, _ = benchmark_step(vqvae_train_step, device, TRAIN_WARMUP, TRAIN_ITERS)
    vq_infer_ms, _ = benchmark_step(vqvae_infer_step, device, INFER_WARMUP, INFER_ITERS)

    lines.append('[1] 生成模型 · VQVAE(重建)')
    lines.append(f'    参数量          : {_fmt_count(vq_total)} (可训练 {_fmt_count(vq_train)})')
    lines.append(f'    FLOPs / 前向    : {fmt_flops(vq_flops, vq_backend)}')
    lines.append(f'    训练时间        : {vq_train_ms:.2f} ms/iter (batch={B})')
    lines.append(f'    推理时间        : {vq_infer_ms:.2f} ms/iter ({vq_infer_ms / B:.3f} ms/样本)')
    lines.append('')

    # ==================================================================
    # [2] 生成模型 · DiT + LatentMeanFlow(MeanFlow)
    # ==================================================================
    dit = vqdit.dit_model
    dit_total, dit_trainable = param_stats(dit)
    dit_flops, dit_backend = count_flops(dit, (z_img, t, r, labels_gen))

    dit_opt = torch.optim.Adam(dit.parameters(), lr=args.lr)

    def dit_train_step():
        # 与 VQDiTTrainer.train_dit_step 一致:编码(no_grad)+ MeanFlow 损失(含 JVP)
        dit.train()
        dit_opt.zero_grad()
        with torch.no_grad():
            z = vqdit.vqvae.encode_continuous(x)
        loss, _, _ = meanflow.loss(vqdit, z, labels_gen)
        loss.backward()
        dit_opt.step()

    def dit_sample_step():
        # 与 LatentMeanFlow.sample_each_class 一致的单步去噪采样
        vqdit.eval()
        with torch.no_grad():
            vqdit(z_img, t, r, labels_gen)

    def generate_step():
        # 端到端生成单张图:潜在采样 + VQVAE 解码(每类 1 张)
        vqdit.eval()
        with torch.no_grad():
            z = meanflow.sample_each_class(model=vqdit, n_per_class=1, classes=[0], device=device)
            vqdit.decode_continuous(z)

    dit_train_ms, _ = benchmark_step(dit_train_step, device, TRAIN_WARMUP, TRAIN_ITERS)
    dit_sample_ms, _ = benchmark_step(dit_sample_step, device, INFER_WARMUP, INFER_ITERS)
    gen_ms, _ = benchmark_step(generate_step, device, INFER_WARMUP, INFER_ITERS)

    lines.append('[2] 生成模型 · DiT + LatentMeanFlow(MeanFlow)')
    lines.append(f'    参数量(DiT)     : {_fmt_count(dit_total)} (可训练 {_fmt_count(dit_trainable)})')
    lines.append(f'    FLOPs / 前向    : {fmt_flops(dit_flops, dit_backend)} (单步去噪, batch={B})')
    lines.append(f'    训练时间        : {dit_train_ms:.2f} ms/iter (含 MeanFlow 损失与 JVP)')
    lines.append(f'    采样时间        : {dit_sample_ms:.2f} ms/iter (单步去噪, batch={B})')
    lines.append(f'    端到端生成      : {gen_ms:.2f} ms/张 (采样 + VQVAE 解码)')
    lines.append('')

    # ==================================================================
    # [3] 双流编码器 · DualPromptNetwork
    # ==================================================================
    dual = DualPromptNetwork(
        dim=C,
        size=S,
        num_classes=K,
        shared_dims=args.shared_dims,
        private_dims=args.private_dims,
        prompt_dim=args.prompt_dim,
        device=args.device,
        proto_warmup_epochs=args.proto_warmup_epochs,
        proto_quality_threshold=args.proto_quality_threshold,
    ).to(device)

    dual_total, dual_trainable = param_stats(dual)
    fe_total = sum(p.numel() for p in dual.feature_extractor.parameters())
    cls_total = sum(p.numel() for p in dual.classifier.parameters())
    prompt_total = sum(p.numel() for p in dual.prompt_module.parameters())

    # 临时关闭前向中的随机增强,避免数据增强算子污染网络耗时/FLOPs
    orig_p = dual.figure_rand1.p
    dual.figure_rand1.p = 0.0

    dual_flops, dual_backend = count_flops(dual, (x1, x2))

    dual_opt = torch.optim.Adam(dual.parameters(), lr=args.classifier_lr)

    def dual_train_step():
        dual.train()
        dual_opt.zero_grad()
        _, loss, _ = dual(x1, x2, labels_cls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dual.parameters(), max_norm=1.0)
        dual_opt.step()

    def dual_infer_step():
        dual.eval()
        with torch.no_grad():
            dual(x1, x2)

    dual_train_ms, _ = benchmark_step(dual_train_step, device, TRAIN_WARMUP, TRAIN_ITERS)
    dual_infer_ms, _ = benchmark_step(dual_infer_step, device, INFER_WARMUP, INFER_ITERS)

    dual.figure_rand1.p = orig_p  # 恢复原始增强概率

    lines.append('[3] 双流编码器 · DualPromptNetwork')
    lines.append(f'    参数量(总)      : {_fmt_count(dual_total)} (可训练 {_fmt_count(dual_trainable)})')
    lines.append(f'      特征提取器    : {_fmt_count(fe_total)}')
    lines.append(f'      分类头        : {_fmt_count(cls_total)}')
    lines.append(f'      提示模块      : {_fmt_count(prompt_total)} (当前未参与 forward,仅计入总量)')
    lines.append(f'    FLOPs / 前向    : {fmt_flops(dual_flops, dual_backend)} (双输入 batch={B})')
    lines.append(f'    训练时间        : {dual_train_ms:.2f} ms/iter (batch={B})')
    lines.append(f'    推理时间        : {dual_infer_ms:.2f} ms/iter ({dual_infer_ms / B:.3f} ms/样本)')
    lines.append('')

    # ==================================================================
    # [4] 参数量汇总(与实际训练流程核对后的口径)
    # ==================================================================
    # 依据训练文件核对:
    #   - VQVAE(含 EMA 码本)与 DiT:基础训练、增量微调中均实际参与训练;
    #   - 双流编码器中仅 feature_extractor 与 classifier 参与 forward/backward;
    #     prompt_module 在 forward 中的调用被注释(Class_Increment.py),参数收不到
    #     梯度,实际不会被训练,也不参与推理;
    #   - LatentMeanFlow 为无参数的纯 Python 类,不计入参数量。
    used_total = vq_total + dit_total + fe_total + cls_total
    defined_total = vq_total + dit_total + dual_total
    lines.append('[4] 参数量汇总')
    lines.append(f'    实际参与训练的参数量 : {_fmt_count(used_total)}')
    lines.append(f'      VQVAE              : {_fmt_count(vq_total)}')
    lines.append(f'      DiT                : {_fmt_count(dit_total)}')
    lines.append(f'      特征提取器          : {_fmt_count(fe_total)}')
    lines.append(f'      分类头              : {_fmt_count(cls_total)}')
    lines.append(f'    模型定义的总参数量   : {_fmt_count(defined_total)}')
    lines.append(f'      (其中提示模块 {_fmt_count(prompt_total)} 未参与 forward,实际不被训练)')
    lines.append('')

    # ---------------------- 口径说明 ----------------------
    lines.append('-' * 78)
    lines.append('注1: 训练/推理计时已做预热与 CUDA 同步;DualPromptNetwork 前向自带的')
    lines.append('     figure_rand 随机增强已临时关闭,计时口径为纯网络前向/反向耗时。')
    lines.append('注2: FLOPs 口径为乘加计 2 FLOPs,仅统计矩阵乘/卷积类算子;MeanFlow 训练')
    lines.append('     步中 JVP 的反向开销已计入耗时,但未计入 FLOPs(反向约为前向的 2 倍)。')
    lines.append('注3: 模型为随机初始化;参数量、FLOPs 与耗时均与权重数值无关。')
    lines.append('=' * 78)

    report = '\n'.join(lines)
    print('\n' + report)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report + '\n')
    print(f'\n报告已保存至: {REPORT_PATH}')


if __name__ == '__main__':
    main()
