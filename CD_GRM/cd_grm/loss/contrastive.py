import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    计算对比损失
    """
    def __init__(self, temperature: float = 0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        #z_a,z_b:[bs,emb_size]
        batch_size = z_a.shape[0]
        device = z_a.device

        #强制将数据类型提升到 float32 (FP32)，抵抗混合精度的下溢出
        z_a = z_a.float()
        z_b = z_b.float()
        # 裁剪极端值，防止 L2 Norm 求平方时数值爆炸 (FP32上限)
        z_a = torch.clamp(z_a, min=-1e4, max=1e4)
        z_b = torch.clamp(z_b, min=-1e4, max=1e4)
        if torch.isnan(z_a).any() or torch.isnan(z_b).any():
            print("⚠️ 警告: 检测到 NaN!")
            # 使用 nan_to_num 将 nan 转为 0，然后再乘 0.0，确保返回真实的 0.0 而非 NaN
            z_a_safe = torch.nan_to_num(z_a)
            z_b_safe = torch.nan_to_num(z_b)
            dummy_loss = (z_a_safe.sum() * 0.0) + (z_b_safe.sum() * 0.0)
            return dummy_loss

        z_a_norm = F.normalize(z_a, p=2, dim=-1, eps=1e-5)
        z_b_norm = F.normalize(z_b, p=2, dim=-1, eps=1e-5)
        logits = torch.matmul(z_a_norm, z_b_norm.T)
        scaled_logits = logits / self.temperature
        #正样本标签
        labels = torch.arange(
            batch_size,
            dtype=torch.long,
            device=device
        )
        #双向损失
        loss_a2b = F.cross_entropy(scaled_logits, labels)
        loss_b2a = F.cross_entropy(scaled_logits.T, labels)

        loss = (loss_a2b + loss_b2a) / 2.0
        return loss