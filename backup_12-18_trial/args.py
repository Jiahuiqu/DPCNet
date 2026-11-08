import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='高光谱图像生成与分类训练')

    # 通用参数
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='使用的设备（CPU/GPU）')
    parser.add_argument('--seed', type=int, default=10,
                        help='随机种子，用于保证实验可重复性')
    parser.add_argument('--data_name', type=str, default='houston13', choices=['paviaU', 'houston13', 'Salinas', 'Indian'],
                        help='数据集名称，支持 paviaU 和 houston13')
    parser.add_argument('--patch_size', type=int, default=9,
                        help='输入图像块大小（如：9x9 像素）')
    parser.add_argument('--in_channels', type=int, default=144, choices=[103, 144, 204, 220],
                        help='输入图像的通道数（即光谱波段数量）')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='训练时每个 batch 的样本数量')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载器使用的线程数')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='是否对输入图像进行归一化处理')
    parser.add_argument('--norm_method', type=str, default='minmax', choices=['l2', 'minmax'],
                        help='归一化方法：l2 或 minmax')

    # 数据集相关参数
    parser.add_argument('--base_classes', type=int, default=5, choices=[9, 15, 16, 16],
                        help='初始阶段训练的基础类别数量')
    parser.add_argument('--base_samples_per_class', type=int, default=5,
                        help='每个基础类别的训练样本数量')
    parser.add_argument('--new_classes_per_round', type=int, default=5,
                        help='每轮增量学习新增的类别数量')
    parser.add_argument('--new_samples_per_class', type=int, default=5,
                        help='增量学习阶段每个新类别的训练样本数量')
    parser.add_argument('--replay_samples_per_class', type=int, default=0,
                        help='每个类别保留的回放样本数量')
    parser.add_argument('--incremental_rounds', type=int, default=2,
                        help='增量学习的总轮次')

    parser.add_argument('--replace_ratio', type=float, default=0.3,
                        help='分类测试集中使用生成图像替换原始图像的比例')

    # 图像搜索参数（EnhancedHybridMatcher）
    parser.add_argument('--local_radius', type=int, default=40,
                        help='局部匹配的搜索半径')
    parser.add_argument('--num_samples_local_class_t2', type=int, default=5,
                        help='t2数据集中局部匹配的采样数量')
    parser.add_argument('--superpixel_topk_t2', type=int, default=3,
                        help='t2数据集中超像素区域选择的 top-k 候选')
    parser.add_argument('--num_samples_global_class_t2', type=int, default=20,
                        help='t2数据集中全局匹配的采样数量')

    parser.add_argument('--min_dist', type=int, default=15,
                        help='候选点之间的最小距离')
    parser.add_argument('--spectral_threshold', type=float, default=0.5,
                        help='谱相似性的阈值，用于判断两个点是否光谱相似')
    parser.add_argument('--context_threshold', type=float, default=0.6,
                        help='上下文相似性的阈值，用于判断两个点是否空间结构相似')

    parser.add_argument('--eval_step', type=int, default=100,
                        help='DiT学习率')

    # VQDiT 生成模型参数
    parser.add_argument('--vqvae_pretrained_path', type=str,
                        default='./checkp/best_vqvae_psnr43.30.pth',
                        choices=['./checkp/best_vqvae_psnr43.30.pth', 'None'],
                        help='预训练VQVAE模型的路径（可选）')
    parser.add_argument('--vqdit_pretrained_path', type=str,
                        default='./checkpoints/best_dit_psnr38.08.pth',
                        choices=['./checkpoints/best_dit_psnr38.08.pth', 'None'],
                        help='预训练VQVAE模型的路径（可选）')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='VQ-VAE 潜在空间维度')
    parser.add_argument('--n_embed', type=int, default=256,
                        help='码本大小（codebook size，Houston 权重为 256 训练）')

    parser.add_argument('--vqvae_steps', type=int, default=50000,
                        help='VQ-VAE 训练步数')
    parser.add_argument('--dit_steps', type=int, default=50000,
                        help='DiT（扩散 Transformer）训练步数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='优化器学习率')
    parser.add_argument('--log_step', type=int, default=5000,
                        help='训练日志输出间隔步数')
    parser.add_argument('--sample_step', type=int, default=5000,
                        help='模型采样保存间隔步数')

    # 分类模型参数（DualPromptNetwork）
    parser.add_argument('--classifier_pretrained_path', type=str,
                        default='',
                        choices=['./checkp/base_classifier_20251217_074451_0.9952.pth',
                                 ],
                        help='基础分类模型的预训练路径')
    parser.add_argument('--shared_dims', type=int, nargs='+', default=[64, 100, 100],
                        help='共享特征提取层的通道维度列表')
    parser.add_argument('--private_dims', type=int, nargs='+', default=[64, 100, 100],
                        help='私有提示向量的维度列表')
    parser.add_argument('--prompt_dim', type=int, default=128,
                        help='提示向量的维度')
    parser.add_argument('--classifier_lr', type=float, default=1e-4,
                        help='分类器的学习率')
    parser.add_argument('--classifier_epochs', type=int, default=300,
                        help='分类器训练的总轮数上限')

    parser.add_argument('--use_cached_t4', action='store_true', default=True,
                        help='如果设置此标志，则使用缓存的t4数据集')
    parser.add_argument('--t4_cache_path', type=str, default='None',
                        help='t4数据集缓存路径')
    parser.add_argument('--use_cached_t5', action='store_true', default=True,
                        help='如果设置此标志，则使用缓存的t5数据集')
    parser.add_argument('--t5_cache_path', type=str, default='None',
                        help='t5数据集缓存路径')

    parser.add_argument('--num_local_samples', type=int, default=3,
                        help='每个训练点从局部搜索中抽取的样本数量')
    parser.add_argument('--num_global_samples', type=int, default=5,
                        help='每个训练点从全局搜索中抽取的样本数量')
    parser.add_argument('--num_generated_samples', type=int, default=2,
                        help='每个训练点生成的样本数量')

    # 新增原型稳定性参数
    parser.add_argument('--proto_warmup_epochs', type=int, default=200,
                        help='原型更新的预热期数（前N个epoch不更新原型）')
    parser.add_argument('--proto_quality_threshold', type=float, default=0.92,
                        help='原型更新的特征质量阈值（0-1）')
    parser.add_argument('--proto_decay_start', type=float, default=0.3,
                        help='原型更新的初始衰减系数')
    parser.add_argument('--proto_decay_end', type=float, default=0.8,
                        help='原型更新的最终衰减系数')
    parser.add_argument('--proto_decay_steps', type=int, default=50,
                        help='从初始衰减到最终衰减的步数（epoch数）')
    parser.add_argument('--num_samples_local_class_t6', type=int, default=5,
                        help='t6数据集中局部匹配的采样数量')
    parser.add_argument('--superpixel_topk_t6', type=int, default=3,
                        help='t6数据集中超像素区域选择的 top-k 候选')
    parser.add_argument('--num_samples_global_class_t6', type=int, default=20,
                        help='t6数据集中全局匹配的采样数量')



    # VQVAE微调选项
    parser.add_argument('--freeze_vqvae_encoder', action='store_true', default=True,
                        help='冻结VQVAE编码器权重')
    parser.add_argument('--freeze_vqvae_decoder', action='store_true', default=True,
                        help='冻结VQVAE解码器权重')

    # 增量DiT微调选项
    parser.add_argument('--dit_train_mode', type=str, default='full',
                        choices=['full', 'class_only', 'freeze_base'],
                        help='DiT训练模式: full(全权重微调), class_only(仅训练新类别嵌入), freeze_base(冻结基础层)')
    parser.add_argument('--dit_lr_multiplier', type=float, default=1.0,
                        help='DiT学习率乘数（相对于基础学习率）')
    parser.add_argument('--dim', type=int, default=256, help='Transformer的维度')
    parser.add_argument('--depth', type=int, default=4, help='Transformer的层数')
    parser.add_argument('--num_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP的扩展比例')

    # 在parse_args()中添加以下增量学习参数
    parser.add_argument('--new_embed', type=int, default=128,
                        help='增量学习时的码本条目数量')

    # VQVAE微调参数
    parser.add_argument('--vqvae_ft_pretrained_path', type=str,
                        default=['./checkp/vqdit_ft_round_1_vqvae_psnr39.89.pth',
                                    './checkpoints/vqdit_ft_round_2_vqvae_psnr44.05.pth'],
                        help='VQVAE微调预训练模型路径（按轮次索引）')
    parser.add_argument('--vqvae_steps_ft', type=int, default=50000,
                        help='VQVAE微调步数')

    # DiT微调参数
    parser.add_argument('--dit_ft_pretrained_path', type=str,
                        default=['./checkpoints/dit_ft_round_1_psnr38.35.pth',
                                    './checkpoints/dit_ft_round_2_psnr41.50.pth'],
                        help='DiT微调预训练模型路径（按轮次索引）')
    parser.add_argument('--dit_steps_ft', type=int, default=50000,
                        help='DiT微调步数')

    # 通用微调参数
    parser.add_argument('--lr_ft', type=float, default=1e-4,
                        help='微调学习率')
    parser.add_argument('--log_step_ft', type=int, default=5000,
                        help='微调日志步数')
    parser.add_argument('--eval_step_ft', type=int, default=5000,
                        help='微调评估步数')
    parser.add_argument('--sample_step_ft', type=int, default=5000,
                        help='微调采样步数间隔')
    parser.add_argument('--incremental_num_local_samples', type=int, default=3,
                        help='每个训练点从局部搜索中抽取的样本数量')
    parser.add_argument('--incremental_num_global_samples', type=int, default=5,
                        help='每个训练点从全局搜索中抽取的样本数量')
    parser.add_argument('--incremental_num_generated_samples', type=int, default=2,
                        help='每个训练点生成的样本数量')
    parser.add_argument('--incremental_num_local_samples_t10', type=int, default=3,
                        help='每个训练点从局部搜索中抽取的样本数量')
    parser.add_argument('--incremental_num_global_samples_t10', type=int, default=5,
                        help='每个训练点从全局搜索中抽取的样本数量')
    parser.add_argument('--incremental_num_generated_samples_t10', type=int, default=2,
                        help='每个训练点生成的样本数量')

    parser.add_argument('--incremental_replace_ratio', type=float, default=0.3,
                        help='分类测试集中使用生成图像替换原始图像的比例')
    parser.add_argument('--use_cached_t9', action='store_true', default=True,
                        help='如果设置此标志，则使用缓存的t9数据集')
    parser.add_argument('--use_cached_t10', action='store_true', default=True,
                        help='如果设置此标志，则使用缓存的t10数据集')

    parser.add_argument('--incremental_classifier_path', type=str,
                        default=['None',
                                    'None'],
                        choices=['./checkp/base_classifier_20251212_063530_0.9482.pth'],
                        help='增量分类模型的路径（可选）')

    return parser.parse_args()

