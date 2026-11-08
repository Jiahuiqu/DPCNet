import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import time
import os
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
from image_generate import VQDiT, LatentMeanFlow, VQDiTIncremental, VQDiTIncremental_2
from torchvision.utils import make_grid, save_image

import random
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import os
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF


def save_samples_by_class(samples, n_per_class, classes, step, root_dir="images",
                          class_names=None, size=(72, 72), test_loader=None, bands=[100, 60, 30], round_idx=0):
    """
    按类别分行保存采样图像，每类占两行（真实图像和生成图像），并放大单个样本尺寸
    真实样本从测试集中随机选择

    参数:
        samples: 采样张量 [batch_size, channels, height, width]
        n_per_class: 每类样本数量
        classes: 类别列表
        step: 当前步骤编号
        root_dir: 保存根目录
        class_names: 类别名称列表 (可选)
        size: 每个样本的目标尺寸 (h, w)
        test_loader: 测试数据加载器，用于获取真实图像
        bands: 要使用的波段索引列表（用于可视化）
    """
    # 获取批次信息
    num_classes = len(classes)
    batch_size, channels, orig_h, orig_w = samples.shape
    assert batch_size == num_classes * n_per_class

    # 创建真实图像字典 {class_id: [image1, image2, ...]}
    real_images_dict = {}
    for class_id in classes:
        real_images_dict[class_id] = []

    # 从测试集中收集真实图像 - 随机选择
    if test_loader is not None:
        # 首先收集每个类别的所有图像
        all_class_images = {class_id: [] for class_id in classes}

        for data in test_loader:
            # 根据不同的数据集结构处理数据
            if len(data) == 4:  # 包含 image, patch, label, y, x
                x, labels, _, _ = data
            elif len(data) >= 3:  # 包含 patches, labels, ...
                x, labels = data[:2]
            else:
                continue

            for i in range(x.size(0)):
                label = labels[i].item()
                if label in all_class_images:
                    img = x[i]
                    # 确保波段索引在有效范围内
                    valid_bands = [b for b in bands if b < img.shape[0]]
                    if not valid_bands:
                        valid_bands = list(range(min(3, img.shape[0])))
                    all_class_images[label].append(img[valid_bands])

        # 从每个类别的所有图像中随机选择n_per_class个
        for class_id, images in all_class_images.items():
            if len(images) >= n_per_class:
                # 随机选择n_per_class个图像
                real_images_dict[class_id] = random.sample(images, n_per_class)
            else:
                # 如果图像不足，使用所有可用图像
                real_images_dict[class_id] = images.copy()

    # 放大单个样本
    if orig_h != size[0] or orig_w != size[1]:
        samples = F.interpolate(samples, size=size, mode="bilinear")

    all_rows = []

    for class_idx in range(num_classes):
        class_id = classes[class_idx]

        # 确保class_label总是有值
        class_label = str(class_id)  # 默认使用类别ID作为标签
        if class_names:
            class_label = class_names[class_id]  # 如果有提供类别名称，则使用

        # 获取当前类别的真实图像
        real_imgs = real_images_dict.get(class_id, [])
        real_imgs = real_imgs[:n_per_class]  # 确保不超过n_per_class

        # 如果真实图像不足，用黑色图像填充
        while len(real_imgs) < n_per_class:
            real_imgs.append(torch.zeros(len(bands), orig_h, orig_w))

        # 调整真实图像尺寸
        real_imgs_tensor = torch.stack(real_imgs).to(samples.device)
        if real_imgs_tensor.shape[-2:] != size:
            real_imgs_tensor = F.interpolate(real_imgs_tensor, size=size, mode="bilinear")

        # 获取当前类别的生成图像
        start_idx = class_idx * n_per_class
        end_idx = start_idx + n_per_class
        gen_imgs = samples[start_idx:end_idx]

        # 确保生成图像也只使用指定波段
        valid_bands = [b for b in bands if b < gen_imgs.shape[1]]
        if not valid_bands:
            valid_bands = list(range(min(3, gen_imgs.shape[1])))
        gen_imgs = gen_imgs[:, valid_bands]

        # 创建真实图像行
        real_row = make_grid(real_imgs_tensor, nrow=n_per_class, padding=0, normalize=True)

        # 创建生成图像行
        gen_row = make_grid(gen_imgs, nrow=n_per_class, padding=0, normalize=True)

        # 添加文本标签
        pil_real_row = TF.to_pil_image(real_row.cpu())
        draw = ImageDraw.Draw(pil_real_row)
        font = ImageFont.load_default()
        draw.text((5, 5), f"Real {class_label}", fill="white", font=font)
        real_row = TF.to_tensor(pil_real_row)

        pil_gen_row = TF.to_pil_image(gen_row.cpu())
        draw = ImageDraw.Draw(pil_gen_row)
        draw.text((5, 5), f"Gen {class_label}", fill="white", font=font)
        gen_row = TF.to_tensor(pil_gen_row)

        # 添加到所有行
        all_rows.extend([real_row, gen_row])

    # 将所有行垂直堆叠
    combined = torch.cat(all_rows, dim=1)

    # 保存图像
    target_dir = f"{root_dir}_{round_idx}"
    os.makedirs(target_dir, exist_ok=True)
    save_path = f"{target_dir}/sample_step_{step}.png"
    save_image(combined, save_path)

    return save_path
def calculate_psnr(recon, target, max_val=1.0):
    """
    计算峰值信噪比(PSNR)

    参数:
    - recon: 重建图像
    - target: 原始图像
    - max_val: 像素值范围的最大值

    返回:
    - psnr: 峰值信噪比
    """

    mse = torch.mean((recon - target) ** 2).item()

    if mse < 1.0e-10:
        return 100

    # 使用 PyTorch 计算 PSNR
    psnr = 20 * math.log10(max_val / math.sqrt(mse))
    return psnr

class VQDiTTrainer:
    def __init__(self, vqdit, vqvae_optimizer, dit_optimizer,
                 latent_meanflow, device, test_loader=None):
        """
        Args:
            vqdit: VQDiT model instance
            vqvae_optimizer: Optimizer for VQVAE part
            dit_optimizer: Optimizer for DiT part
            latent_meanflow: LatentMeanFlow instance
            device: Training device
            test_loader: Test data loader for evaluation
        """
        self.vqdit = vqdit
        self.vqvae_optimizer = vqvae_optimizer
        self.dit_optimizer = dit_optimizer
        self.latent_meanflow = latent_meanflow
        self.device = device
        self.test_loader = test_loader
        self.best_vqvae_psnr = 0.0
        self.best_dit_psnr = 0.0

    def train_vqvae_step(self, x):
        """Train VQVAE component"""
        self.vqdit.train_vqvae(True)
        self.vqdit.vqvae.train()
        recon, diff, commitment_weight = self.vqdit.vqvae(x)
        recon_loss = F.mse_loss(recon, x)
        # l1_loss = F.l1_loss(recon, x)
        # recon_loss = F.l1_loss(recon, x)
        loss = recon_loss + 0.0001*diff

        self.vqvae_optimizer.zero_grad()
        loss.backward()
        self.vqvae_optimizer.step()

        return loss.item(), recon_loss.item(), diff.item()

    def train_dit_step(self, x, c):
        """Train DiT component in latent space"""
        self.vqdit.train_vqvae(False)
        self.vqdit.train_dit(True)

        # 编码图像到潜在空间（连续表示）
        with torch.no_grad():
            z = self.vqdit.vqvae.encode_continuous(x)

        # print("z:", z.size(), "x:", x.size())
        # 训练DiT
        loss, mse_val, u = self.latent_meanflow.loss(self.vqdit, z, c)

        # with torch.no_grad():
        #     rex = self.vqdit.vqvae.decode_continuous(u)
        #     l1 = F.l1_loss(rex, x)

        # loss += l1
        self.dit_optimizer.zero_grad()
        # if self.accelerator:
        #     self.accelerator.backward(loss)
        # else:
        loss.backward()
        self.dit_optimizer.step()

        return loss.item(), mse_val.item()

    def evaluate_vqvae(self):
        """Evaluate VQVAE reconstruction performance"""
        if not self.test_loader:
            return 0.0

        self.vqdit.test_vqvae(False)
        self.vqdit.test_dit(False)

        self.vqdit.eval()
        total_psnr = 0.0
        total_samples = 0

        with torch.no_grad():
            for data in self.test_loader:
                x = data[0].to(self.device)
                recon, _, _ = self.vqdit.vqvae(x)
                total_psnr += calculate_psnr(recon, x) * x.size(0)
                total_samples += x.size(0)

        avg_psnr = total_psnr / total_samples if total_samples > 0 else 0.0
        return avg_psnr

    def evaluate_dit(self):
        if not self.test_loader:
            return 0.0

        self.vqdit.test_vqvae(False)
        self.vqdit.test_dit(False)

        self.vqdit.eval()
        total_psnr = 0.0
        total_samples = 0

        with torch.no_grad():
            for data in self.test_loader:
                x = data[0].to(self.device)
                labels = data[1].to(self.device) if len(data) > 3 else None

                # 1. 编码到连续空间
                clean_z = self.vqdit.vqvae.encode_continuous(x)

                # 2. 直接调用loss函数获取去噪特征
                _, _, denoised_z = self.latent_meanflow.loss(
                    self.vqdit, clean_z, labels
                )
                # print(clean_z.size(), denoised_z.size())

                # 3. 解码
                recon = self.vqdit.vqvae.decode_continuous(denoised_z)

                # 4. 计算PSNR
                total_psnr += calculate_psnr(recon, x) * x.size(0)
                total_samples += x.size(0)

        return total_psnr / total_samples if total_samples > 0 else 0.0

    @torch.no_grad()
    def sample(self, n_per_class, classes=None):
        """Generate hyperspectral images"""
        self.vqdit.eval()
        device = next(self.vqdit.parameters()).device

        # 生成潜在表示
        z_latent = self.latent_meanflow.sample_each_class(
            self.vqdit,
            n_per_class,
            classes,
            device=device
        )

        # 解码为图像空间
        generated = self.vqdit.vqvae.decode_continuous(z_latent)
        return generated

class VQDiTTrainer_increment:
    def __init__(self, dit_model, vqvae_optimizer, dit_optimizer,
                 latent_meanflow, device, new_classes, test_loader=None):
        """
        Args:
            vqdit: VQDiT model instance
            vqvae_optimizer: Optimizer for VQVAE part
            dit_optimizer: Optimizer for DiT part
            latent_meanflow: LatentMeanFlow instance
            device: Training device
            test_loader: Test data loader for evaluation
        """
        self.vqvae_optimizer = vqvae_optimizer
        self.dit_optimizer = dit_optimizer
        self.latent_meanflow = latent_meanflow
        self.device = device
        self.test_loader = test_loader
        self.best_vqvae_psnr = 0.0
        self.best_dit_psnr = 0.0
        self.dit_model = dit_model
        self.new_classes = new_classes

    def train_vqvae_step(self, x):
        """Train VQVAE component"""
        self.dit_model.train_vqvae(True)
        self.dit_model.train_dit(False)
        recon, diff, commitment_weight = self.dit_model.vqvae(x, use_st_estimator=False)
        recon_loss = F.mse_loss(recon, x)
        # l1_loss = F.l1_loss(recon, x)
        # recon_loss = F.l1_loss(recon, x)
        loss = recon_loss + 0.0001*diff

        self.vqvae_optimizer.zero_grad()
        loss.backward()
        self.vqvae_optimizer.step()

        return loss.item(), recon_loss.item(), diff.item()

    def train_dit_step(self, x, c):
        """Train DiT component in latent space"""
        self.dit_model.test_vqvae()
        self.dit_model.test_dit()

        # 编码图像到潜在空间（连续表示）
        with torch.no_grad():
            z = self.dit_model.encode_continuous(x)

        # print("z:", z.size(), "x:", x.size())
        # 训练DiT
        loss, mse_val, u = self.latent_meanflow.loss_increment(self.dit_model, z, c, self.new_classes)

        # with torch.no_grad():
        #     rex = self.vqdit.vqvae.decode_continuous(u)
        #     l1 = F.l1_loss(rex, x)

        # loss += l1
        self.dit_optimizer.zero_grad()
        # if self.accelerator:
        #     self.accelerator.backward(loss)
        # else:
        loss.backward()
        self.dit_optimizer.step()

        return loss.item(), mse_val.item()

    def evaluate_vqvae(self):
        """Evaluate VQVAE reconstruction performance"""
        if not self.test_loader:
            return 0.0

        self.dit_model.test_vqvae()
        self.dit_model.test_dit()

        self.dit_model.eval()
        total_psnr = 0.0
        total_samples = 0

        with torch.no_grad():
            for data in self.test_loader:
                x = data[0].to(self.device)
                recon, _, _ = self.dit_model.vqvae(x)
                total_psnr += calculate_psnr(recon, x) * x.size(0)
                total_samples += x.size(0)

        avg_psnr = total_psnr / total_samples if total_samples > 0 else 0.0
        return avg_psnr

    def evaluate_dit(self):
        if not self.test_loader:
            return 0.0

        self.dit_model.train_vqvae(False)
        self.dit_model.train_dit(False)

        self.dit_model.eval()
        total_psnr = 0.0
        total_samples = 0

        with torch.no_grad():
            for data in self.test_loader:
                x = data[0].to(self.device)
                labels = data[1].to(self.device) if len(data) > 3 else None

                # 1. 编码到连续空间
                clean_z = self.dit_model.encode_continuous(x)

                # 2. 直接调用loss函数获取去噪特征
                _, _, denoised_z = self.latent_meanflow.loss_increment(
                    self.dit_model, clean_z, labels, self.new_classes
                )
                # print(clean_z.size(), denoised_z.size())

                # 3. 解码
                recon = self.dit_model.decode_continuous(denoised_z)

                # 4. 计算PSNR
                total_psnr += calculate_psnr(recon, x) * x.size(0)
                total_samples += x.size(0)

        return total_psnr / total_samples if total_samples > 0 else 0.0

    @torch.no_grad()
    def sample(self, n_per_class, classes=None):
        """Generate hyperspectral images"""
        self.dit_model.eval()
        device = next(self.dit_model.parameters()).device

        # 生成潜在表示
        z_latent = self.latent_meanflow.increment_sample(
            self.dit_model,
            n_per_class,
            classes,
            device=device
        )

        # 解码为图像空间
        generated = self.dit_model.decode_continuous(z_latent)
        return generated



def train_vqdit_hyperspectral(train_loader, test_loader, args, base_classes):
    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """训练VQDiT模型的主函数（包含测试评估）"""
    # 从args中获取参数
    in_channels = args.in_channels
    img_size = args.patch_size
    num_classes = base_classes
    latent_dim = args.latent_dim
    n_embed = args.n_embed
    vqvae_steps = args.vqvae_steps
    dit_steps = args.dit_steps
    lr = args.lr
    # sample_step = args.sample_step
    # log_step = args.log_step
    # eval_step = args.eval_step
    sample_step = 5000
    log_step = 5000
    eval_step = 5000

    # 设置设备
    device = torch.device(args.device)

    # 创建目录
    os.makedirs('images', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    # 创建VQDiT模型
    vqdit = VQDiT(
        in_channels=in_channels,
        latent_dim=latent_dim,
        img_size=img_size,
        num_classes=num_classes,
        n_embed=n_embed
    ).to(device)

    # 创建优化器
    vqvae_optimizer = Adam(vqdit.vqvae.parameters(), lr=lr)

    # 只优化DiT部分参数
    dit_optimizer = Adam(vqdit.dit_model.parameters(), lr=lr)

    # 创建潜在空间MeanFlow
    latent_meanflow = LatentMeanFlow(
        latent_dim=latent_dim,
        image_size=img_size,
        num_classes=num_classes,
        normalizer=['minmax', None, None],
        flow_ratio=0,
        time_dist=['lognorm', -0.4, 1.0],
        cfg_ratio=0,
        cfg_scale=2.0
    )

    # 创建训练器（包含测试数据）
    trainer = VQDiTTrainer(
        vqdit=vqdit,
        vqvae_optimizer=vqvae_optimizer,
        dit_optimizer=dit_optimizer,
        latent_meanflow=latent_meanflow,
        device=device,
        test_loader=test_loader
    )
    yqvae_image_show = True
    # 设备验证
    print(f"✅ 模型设备: {device}")

    # 初始化跳过标志
    skip_vqvae_training = False
    skip_dit_training = False

    # 处理 VQVAE 预训练模型
    vqvae_pretrained_path = getattr(args, 'vqvae_pretrained_path', None)
    if vqvae_pretrained_path and vqvae_pretrained_path.lower() != 'none' and os.path.exists(vqvae_pretrained_path):
        print(f"🔍 尝试加载 VQVAE 预训练模型: {vqvae_pretrained_path}")

        try:
            state_dict = torch.load(vqvae_pretrained_path, map_location=device)
            vqdit.vqvae.load_state_dict(state_dict)
            skip_vqvae_training = True
            print("✅ VQVAE 模型加载成功")
            print("⏭️ 将跳过 VQVAE 训练")
        except Exception as e:
            print(f"⚠️ 加载 VQVAE 预训练模型失败: {str(e)}")
            print("🆕 将从头训练 VQVAE")
    else:
        if vqvae_pretrained_path and vqvae_pretrained_path.lower() != 'none':
            print(f"⚠️ VQVAE 预训练模型路径无效: {vqvae_pretrained_path}")
        print("🆕 未提供有效 VQVAE 预训练模型，将从头训练 VQVAE")

    # 处理 VQDiT 完整模型预训练
    vqdit_pretrained_path = getattr(args, 'vqdit_pretrained_path', None)
    if vqdit_pretrained_path and vqdit_pretrained_path.lower() != 'none' and os.path.exists(vqdit_pretrained_path):
        print(f"🔍 尝试加载完整 VQDiT 预训练模型: {vqdit_pretrained_path}")

        try:
            state_dict = torch.load(vqdit_pretrained_path, map_location=device)
            vqdit.load_state_dict(state_dict)
            skip_dit_training = True
            print("✅ 完整 VQDiT 模型加载成功")
            print("⏭️ 将跳过 DiT 训练")
        except Exception as e:
            print(f"⚠️ 加载完整 VQDiT 模型失败: {str(e)}")
            print("🆕 将从头训练 DiT")
    else:
        if vqdit_pretrained_path and vqdit_pretrained_path.lower() != 'none':
            print(f"⚠️ VQDiT 预训练模型路径无效: {vqdit_pretrained_path}")
        print("🆕 未提供有效 VQDiT 预训练模型，将从头训练 DiT")

    # 调整训练步数
    if skip_vqvae_training:
        vqvae_steps = 0
        print("🆕 跳过 VQVAE 训练")

    if skip_dit_training:
        dit_steps = 0
        print("🆕 跳过 DiT 训练")


    # ===========================================================
    # 第一阶段：训练VQVAE（重建能力）
    # ===========================================================
    vqdit.vqvae.train()
    print(f"🚀 开始训练VQVAE，共{vqvae_steps}步...")
    if not skip_vqvae_training and vqvae_steps > 0:
        vqdit.train_vqvae(True)  # 添加这一行确保模型处于训练模式

    train_iter = iter(train_loader)
    best_vqvae_psnr = 0.0
    best_vqvae_model_path = None  # 添加变量存储最优模型路径

    for step in range(vqvae_steps):

        try:
            data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            data = next(train_iter)

        # 获取输入数据
        x = data[0].to(device)  # 图像块

        # 训练VQVAE
        vqvae_loss, recon_loss, diff_loss = trainer.train_vqvae_step(x)


        # 记录日志
        if step % log_step == 0:
            print(f"VQVAE Step {step}/{vqvae_steps}: "
                  f"Loss={vqvae_loss:.4f}, Recon={recon_loss:.4f}, Diff={diff_loss:.4f}")

        # 评估和保存模型
        if step % eval_step == 0 or step == vqvae_steps - 1:
            psnr = trainer.evaluate_vqvae()
            print(f"VQVAE Step {step}/{vqvae_steps}: PSNR={psnr:.2f} dB")

            if psnr > best_vqvae_psnr:
                best_vqvae_psnr = psnr
                best_vqvae_model_path = f"checkpoints/best_vqvae_psnr_{args.data_name}_{psnr:.2f}.pth"  # 更新最优路径
                torch.save(vqdit.vqvae.state_dict(), best_vqvae_model_path)
                print(f"✅ 保存最佳VQVAE模型 (PSNR={psnr:.2f} dB)")

                if yqvae_image_show == True:
                    vqdit.vqvae.eval()
                    # 获取整个测试集
                    test_set = test_loader.dataset
                    num_samples = len(test_set)
                    num_show = min(100, num_samples)

                    # 随机选择num_show个索引
                    indices = torch.randperm(num_samples)[:num_show]

                    # 收集原始图像和重建图像
                    originals = []
                    reconstructions = []

                    # 选择波段的索引，这里使用前3个波段（或者指定波段）
                    band_indices = [100, 60, 30]  # 示例波段，可以根据实际情况调整
                    # 确保波段索引在通道范围内
                    band_indices = [b for b in band_indices if b < in_channels]
                    if len(band_indices) < 3:
                        # 如果通道数不够，则用前min(3, in_channels)个波段
                        band_indices = list(range(min(3, in_channels)))

                    for idx in indices:
                        # 获取一个样本
                        sample = test_set[idx]
                        # 假设测试集返回的是 (image, label, ...) 我们只需要图像
                        x_sample = sample[0].unsqueeze(0).to(device)  # 增加batch维度

                        with torch.no_grad():
                            recon, _,_ = vqdit.vqvae(x_sample)

                        # 只取指定波段
                        original_img = x_sample[0, band_indices, ...]
                        recon_img = recon[0, band_indices, ...]

                        originals.append(original_img)
                        reconstructions.append(recon_img)

                    # 拼接所有图像
                    originals_tensor = torch.stack(originals)
                    reconstructions_tensor = torch.stack(reconstructions)

                    # 合并原始和重建图像
                    comparison = torch.cat([originals_tensor, reconstructions_tensor], dim=0)

                    # 保存图像
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    save_path = f"images/reconstruction_comparison_{args.data_name}_psnr{psnr:.2f}_{timestamp}.png"
                    save_image(comparison, save_path, nrow=num_show, normalize=True)
                    print(f"✅ 保存重建对比图: {save_path}")
                    print(f"  原始图像 (上排) 和重建图像 (下排) 对比")

                    # 恢复训练模式
                    vqdit.vqvae.train()

    # 训练结束后加载最优模型
    if best_vqvae_model_path:
        vqdit.vqvae.load_state_dict(torch.load(best_vqvae_model_path, map_location=device))
        print(f"✅✅ 加载最优VQVAE模型: {best_vqvae_model_path} (PSNR={best_vqvae_psnr:.2f} dB)")

    print(f"✅ VQVAE训练完成，最佳PSNR: {best_vqvae_psnr:.2f} dB")

    # 冻结VQVAE参数
    vqdit.train_vqvae(False)

    # ===========================================================
    # 第二阶段：训练DiT（生成能力）
    # ===========================================================
    vqdit.vqvae.eval()
    vqdit.train_dit(True)

    print(f"🚀 开始训练DiT，共{dit_steps}步...")

    train_iter = iter(train_loader)  # 重置迭代器
    best_dit_psnr = 0.0
    best_dit_model_path = None  # 添加变量存储最优模型路径

    for step in range(dit_steps):
        try:
            data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            data = next(train_iter)

        # 获取输入数据和标签
        # print(data)
        x = data[0].to(device)
        c = data[1].to(device) if len(data) > 3 else None

        # 训练DiT
        dit_loss, mse_val = trainer.train_dit_step(x, c)

        # 记录日志
        if step % log_step == 0:
            print(f"DiT Step {step}/{dit_steps}: Loss={dit_loss:.4f}, MSE={mse_val:.4f}")

        # 处理NaN值
        if math.isnan(dit_loss) or math.isnan(mse_val):
            print(f"⚠️ 警告: 在步骤 {step} 检测到NaN值，停止训练")
            break

        # 采样和保存样本
        if step % sample_step == 0:
            # 限制生成的最大类别数
            # print(num_classes)
            num_class_count = num_classes  # 假设 num_classes 是整数表示的类别数量

            # 创建类别索引列表 [0, 1, ..., num_class_count]
            class_indices = list(range(0, num_class_count))

            # 生成样本
            samples = trainer.sample(
                n_per_class=50,
                classes=class_indices  # 使用类别索引列表
            )
            # 保存样本图像（使用前3个波段进行可视化）
            save_path = save_samples_by_class(
                samples.cpu(),  # 注意这里不再预先选择波段
                n_per_class=50,
                classes=class_indices,
                step=step,
                size=(72, 72),
                test_loader=test_loader,
                bands=[100, 60, 30]  # 明确指定波段
            )
            print(f"📸 保存采样图像: {save_path}")

            # 评估和保存模型（修复位置和逻辑）
            if step % eval_step == 0 or step == dit_steps - 1:
                psnr = trainer.evaluate_dit()
                print(f"DiT Step {step}/{dit_steps}: PSNR={psnr:.2f} dB")

                if psnr > best_dit_psnr:
                    best_dit_psnr = psnr
                    best_dit_model_path = f"checkpoints/best_dit_psnr_{args.data_name}_{psnr:.2f}.pth"  # 更新最优路径
                    torch.save(vqdit.state_dict(), best_dit_model_path)
                    print(f"✅ 保存最佳DiT模型 (PSNR={psnr:.2f} dB)")

            # 仅在最后一步保存最终模型
            # if step == dit_steps - 1:
            #     timestamp = time.strftime('%Y%m%d_%H%M%S')
            #     ckpt_path = f"checkpoints/vqdit_final_psnr{psnr:.2f}_{timestamp}.pth"
            #     torch.save(vqdit.state_dict(), ckpt_path)
            # print(f"✅ 最终模型已保存至: {ckpt_path}")
            print(f"✅ DiT训练完成，最佳PSNR: {best_dit_psnr:.2f} dB")

    if best_dit_model_path:
        vqdit.load_state_dict(torch.load(best_dit_model_path, map_location=device))
        print(f"✅✅ 加载最优DiT模型: {best_dit_model_path} (PSNR={best_dit_psnr:.2f} dB)")

    return vqdit, latent_meanflow


def fine_tune_vqdit_hyperspectral(model, latent_meanflow,
                                  train_loader_vqvae, test_loader_vqvae,
                                  train_loader_dit, test_loader_dit,
                                  args, new_classes, round_idx):
    """
    增量微调VQDiT模型，支持分别加载VQVAE和DiT预训练模型
    """
    device = torch.device(args.device)
    print(f"🔧🔧🔧🔧 准备增量微调 - 轮次 {round_idx}")
    print(f"🆕🆕🆕🆕🆕🆕 新增类别: {new_classes} (共 {len(new_classes)} 个)")
    new_classes_number = len(new_classes)

    # 创建增量模型
    incremental_model = VQDiTIncremental_2(
        model,
        in_channels=args.in_channels,
        img_size=args.patch_size,
        num_classes=new_classes_number,
        n_embed=args.new_embed
    ).to(device)

    latent_meanflow = LatentMeanFlow(
        latent_dim=args.latent_dim,
        image_size=args.patch_size,
        num_classes=new_classes,
        normalizer=['minmax', None, None],
        flow_ratio=0,
        time_dist=['lognorm', -0.4, 1.0],
        cfg_ratio=0,
        cfg_scale=2.0
    )

    # ============ 统一在此处设置参数训练状态 ============
    # 1. 设置编码器训练状态
    if args.freeze_vqvae_encoder:
        for param in incremental_model.vqvae.encoder.parameters():
            param.requires_grad = False
        print("🔒🔒 冻结VQVAE编码器参数")
    else:
        print("🔓🔓 解冻VQVAE编码器参数")

    # 2. 设置解码器训练状态
    if args.freeze_vqvae_decoder:
        for param in incremental_model.vqvae.decoder.parameters():
            param.requires_grad = False
        print("🔒🔒 冻结VQVAE解码器参数")
    else:
        print("🔓🔓 解冻VQVAE解码器参数")

    # 3. 新量化器始终需要训练
    for param in incremental_model.vqvae.quantize.parameters():
        param.requires_grad = True
    print("🔓🔓 解冻新量化器参数")

    # 4. DiT模型始终需要训练
    for param in incremental_model.dit_model.parameters():
        param.requires_grad = True
    print("🔓🔓 解冻DiT模型参数")

    # ============ 创建优化器 ============
    vqvae_optimizer = Adam(incremental_model.vqvae.parameters(), lr=args.lr_ft)
    dit_optimizer = Adam(incremental_model.dit_model.parameters(), lr=args.lr_ft)

    # ============ 1. 根据round_idx选择VQVAE预训练路径 ============
    vqvae_pretrained_path = getattr(args, 'vqvae_ft_pretrained_path', None)

    # 如果路径是列表，根据round_idx-1选择对应的元素
    if isinstance(vqvae_pretrained_path, list):
        path_index = round_idx - 1
        if 0 <= path_index < len(vqvae_pretrained_path):
            vqvae_pretrained_path = vqvae_pretrained_path[path_index]
            print(f"🔄🔄🔄🔄 第{round_idx}轮使用VQVAE预训练路径索引: {path_index}")
        else:
            vqvae_pretrained_path = None
            print(f"⚠️ VQVAE预训练路径列表索引 {path_index} 超出范围")
    else:
        print("ℹℹ️ 使用单个VQVAE预训练路径")

    # 尝试加载VQVAE预训练模型
    skip_vqvae = False
    if vqvae_pretrained_path and vqvae_pretrained_path != 'None' and os.path.exists(vqvae_pretrained_path):
        print(f"🔍🔍🔍🔍 尝试加载VQVAE预训练模型: {vqvae_pretrained_path}")
        try:
            state_dict = torch.load(vqvae_pretrained_path, map_location=device)

            # 提取VQVAE相关组件的状态
            vqvae_state = {}
            for k in state_dict.keys():
                if 'vqvae_encoder' in k or 'vqvae_decoder' in k or 'new_quantize' in k:
                    vqvae_state[k] = state_dict[k]

            # 加载状态并设置跳过标志
            incremental_model.load_state_dict(vqvae_state, strict=False)
            skip_vqvae = True
            print("✅✅ VQVAE预训练模型加载成功，将跳过VQVAE微调")
        except Exception as e:
            print(f"⚠️⚠️ VQVAE预训练加载失败: {str(e)}")
            print("🆕🆕🆕🆕🆕🆕 将从头训练VQVAE部分")
    else:
        if vqvae_pretrained_path and vqvae_pretrained_path != 'None':
            print(f"⚠️ VQVAE预训练模型路径无效或不存在: {vqvae_pretrained_path}")
        else:
            print("ℹℹ️ 未提供VQVAE预训练模型路径")
        print("🆕🆕🆕🆕🆕🆕 将从头训练VQVAE部分")

    # ============ 2. 根据round_idx选择DiT预训练路径 ============
    dit_pretrained_path = getattr(args, 'dit_ft_pretrained_path', None)

    # 如果路径是列表，根据round_idx-1选择对应的元素
    if isinstance(dit_pretrained_path, list):
        path_index = round_idx - 1
        if 0 <= path_index < len(dit_pretrained_path):
            dit_pretrained_path = dit_pretrained_path[path_index]
            print(f"🔄🔄🔄🔄 第{round_idx}轮使用DiT预训练路径索引: {path_index}")
        else:
            dit_pretrained_path = None
            print(f"⚠️ DiT预训练路径列表索引 {path_index} 超出范围")
    else:
        print("ℹℹ️ 使用单个DiT预训练路径")

    # 尝试加载DiT预训练模型
    skip_dit = False
    if dit_pretrained_path and dit_pretrained_path != 'None' and os.path.exists(dit_pretrained_path):
        print(f"🔍🔍🔍🔍 尝试加载DiT预训练模型: {dit_pretrained_path}")
        try:
            state_dict = torch.load(dit_pretrained_path, map_location=device)
            incremental_model.load_state_dict(state_dict, strict=False)
            skip_dit = True
            print("✅✅ DiT预训练模型加载成功，将跳过DiT微调")
        except Exception as e:
            print(f"⚠️⚠️ DiT预训练加载失败: {str(e)}")
            print("🆕🆕🆕🆕🆕🆕 将从头训练DiT部分")
    else:
        if dit_pretrained_path and dit_pretrained_path != 'None':
            print(f"⚠️ DiT预训练模型路径无效或不存在: {dit_pretrained_path}")
        else:
            print("ℹℹ️ 未提供DiT预训练模型路径")
        print("🆕🆕🆕🆕🆕🆕 将从头训练DiT部分")

    # 打印最终选择的路径
    print(f"📊📊📊📊 第{round_idx}轮最终使用的模型路径:")
    print(f"  VQVAE: {vqvae_pretrained_path if vqvae_pretrained_path else '无'}")
    print(f"  DiT: {dit_pretrained_path if dit_pretrained_path else '无'}")

    # ========================= VQVAE微调部分 =========================
    if not skip_vqvae and args.vqvae_steps_ft > 0:
        print(f"🚀🚀🚀🚀 开始微调新码本 ({args.vqvae_steps_ft} 步)...")

        best_vqvae_psnr = 0.0
        best_vqvae_model_path = None
        train_iter = iter(train_loader_vqvae)

        # 确保模型处于正确的训练模式
        incremental_model.train_vqvae(True)

        for step in range(args.vqvae_steps_ft):
            try:
                data = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader_vqvae)
                data = next(train_iter)

            x = data[0].to(device)

            # 确保模型在训练模式
            incremental_model.train_vqvae(True)

            # 前向传播
            recon, quant_loss = incremental_model.reconstruct(x)

            # 计算损失
            recon_loss = F.mse_loss(recon, x)
            loss = recon_loss + 0.0001 * quant_loss

            # 反向传播
            vqvae_optimizer.zero_grad()
            loss.backward()
            vqvae_optimizer.step()

            # 日志记录
            if step % args.log_step_ft == 0:
                print(f"VQVAE-FT 步骤 {step}/{args.vqvae_steps_ft}: "
                      f"总损失={loss.item():.4f}, 重建损失={recon_loss.item():.4f}, 量化损失={quant_loss.item():.4f}")

            # 评估和保存
            if step % args.eval_step_ft == 0 or step == args.vqvae_steps_ft - 1:
                incremental_model.eval()
                total_psnr = 0.0
                total_samples = 0

                with torch.no_grad():
                    for data in test_loader_vqvae:
                        x = data[0].to(device)
                        recon, _ = incremental_model.reconstruct(x)

                        # 计算PSNR
                        mse = torch.mean((recon - x) ** 2).item()
                        if mse < 1.0e-10:
                            psnr = 100
                        else:
                            psnr = 20 * math.log10(1.0 / math.sqrt(mse))

                        total_psnr += psnr * x.size(0)
                        total_samples += x.size(0)

                avg_psnr = total_psnr / total_samples if total_samples > 0 else 0.0
                print(f"VQVAE-FT 步骤 {step}/{args.vqvae_steps_ft}: PSNR={avg_psnr:.2f} dB")

                # 保存最佳模型
                if avg_psnr > best_vqvae_psnr:
                    best_vqvae_psnr = avg_psnr
                    best_vqvae_model_path = f"checkpoints/vqdit_ft_round_{args.data_name}_{round_idx}_vqvae_psnr{avg_psnr:.2f}.pth"
                    torch.save(incremental_model.state_dict(), best_vqvae_model_path)
                    print(f"💾💾💾💾 保存最佳VQVAE微调模型 (PSNR={avg_psnr:.2f} dB)")

                incremental_model.train()

        # VQVAE微调结束：加载最优模型
        if best_vqvae_model_path:
            incremental_model.load_state_dict(torch.load(best_vqvae_model_path, map_location=device))
            print(f"✅✅ 加载VQVAE微调最优模型: {best_vqvae_model_path}")

        print(f"✅✅ VQVAE微调完成，最佳PSNR: {best_vqvae_psnr:.2f} dB")
    else:
        print("⏭⏭⏭️⏭⏭⏭️ 跳过VQVAE微调")

    # ========================= DiT微调部分 =========================
    if not skip_dit and args.dit_steps_ft > 0:
        print(f"🚀🚀🚀🚀 开始微调DiT ({args.dit_steps_ft} 步)...")

        # 创建训练器
        trainer_dit = VQDiTTrainer_increment(
            dit_model=incremental_model,
            vqvae_optimizer=None,
            dit_optimizer=dit_optimizer,
            latent_meanflow=latent_meanflow,
            device=device,
            test_loader=test_loader_dit,
            new_classes=new_classes
        )

        best_dit_psnr = 0.0
        best_dit_model_path = None
        train_iter = iter(train_loader_dit)

        for step in range(args.dit_steps_ft):
            try:
                data = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader_dit)
                data = next(train_iter)

            # 获取数据
            x = data[0].to(device)
            c = data[1].to(device)  # 新类别标签

            # 训练DiT
            dit_loss, mse_val = trainer_dit.train_dit_step(x, c)

            # 日志记录
            if step % args.log_step_ft == 0:
                print(f"DiT-FT 步骤 {step}/{args.dit_steps_ft}: 损失={dit_loss:.4f}, MSE={mse_val:.4f}")

            # 采样和保存样本
            if step % args.sample_step_ft == 0:
                incremental_model.eval()
                samples = trainer_dit.sample(
                    n_per_class=50,
                    classes=new_classes
                )

                # 保存样本图像
                save_path = save_samples_by_class(
                    samples.cpu(),
                    n_per_class=50,
                    classes=new_classes,
                    step=step,
                    size=(72, 72),
                    test_loader=test_loader_dit,
                    bands=[100, 60, 30],
                    round_idx=round_idx
                )
                print(f"📸📸📸📸 保存采样图像: {save_path}")

                # 评估
                if step % args.eval_step_ft == 0:
                    psnr = trainer_dit.evaluate_dit()
                    print(f"DiT-FT 步骤 {step}/{args.dit_steps_ft}: PSNR={psnr:.2f} dB")

                    if psnr > best_dit_psnr:
                        best_dit_psnr = psnr
                        best_dit_model_path = f"checkpoints/dit_ft_round_{args.data_name}_{round_idx}_psnr{psnr:.2f}.pth"
                        torch.save(incremental_model.state_dict(), best_dit_model_path)
                        print(f"💾💾💾💾 保存最佳DiT微调模型 (PSNR={psnr:.2f} dB)")

                incremental_model.train()

        # DiT微调结束：加载最优模型
        if best_dit_model_path:
            incremental_model.load_state_dict(torch.load(best_dit_model_path, map_location=device))
            print(f"✅✅ 加载DiT微调最优模型: {best_dit_model_path}")

        print(f"✅✅ DiT微调完成，最佳PSNR: {best_dit_psnr:.2f} dB")
    else:
        print("⏭⏭⏭️⏭⏭⏭️ 跳过DiT微调")

    return incremental_model, latent_meanflow