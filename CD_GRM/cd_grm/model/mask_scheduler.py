import math
import torch
import torch.nn as nn


class CosineMaskScheduler(nn.Module):
    # 根据公式算出每一步的保留率；
    # 给每个批次的样本生成随机扩散步数；
    # 生成随机数，根据保留率决定哪些token被遮盖，返回遮盖后数据和遮盖位置；
    def __init__(self, num_steps: int = 10, mask_token_id: int = 0, s: float = 0.008):
        """
        Args:
            num_steps (int): 扩散总步数 T
            mask_token_id (int): MASK token 的 ID
            s (float): 余弦调度平滑参数
        """
        super(CosineMaskScheduler, self).__init__()  # 调用父类 nn.Module 初始化

        self.num_steps = num_steps
        self.mask_token_id = mask_token_id
        # steps = [0,1,2,...,T]
        steps = torch.arange(num_steps + 1, dtype=torch.float32)
        #根据公式计算每一步的 保留率；
        f_t = torch.cos(
            ((steps / num_steps) + s) / (1.0 + s) * (math.pi / 2.0)
        ) ** 2
        alpha_bar_t = f_t / f_t[0] #[T+1]
        self.register_buffer(
            'alpha_bar_t',
            alpha_bar_t
        )

    def get_random_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        #为批次中每个数据随机构造 扩散步
        t = torch.randint(
            1,                       # 最小值 闭
            self.num_steps + 1,      # 最大值 (T+1) 开
            (batch_size,),           # 输出 shape
            device=device            # 放到 GPU / CPU
        )
        return t

    def add_noise(self, x_start: torch.Tensor, t: torch.Tensor) -> tuple:
        #根据随机步 给批次sid序列加上257
        """
        Args:
            x_start:原始 Semantic IDshape:[batch_size, m_layer]
            t:扩散时间步 shape:[batch_size]
        """
        batch_size, seq_len = x_start.shape
        extract_alpha = self.alpha_bar_t[t].unsqueeze(-1).expand(batch_size, seq_len)#[batch_size, m_layer]

        rand_tensor = torch.rand_like(x_start, dtype=torch.float32)# [batch_size, m_layer]
        mask_bool = rand_tensor < (1.0 - extract_alpha)# [batch_size, m_layer]

        # 强制每个样本至少 mask 一个位置
        no_mask_rows = ~mask_bool.any(dim=1)#[bs]按行找是否有true，有true的行经取反后变为faslse
        #整行无掩码的行取反后为 true；
        if no_mask_rows.any():#若有整行无掩码的
            row_idx = no_mask_rows.nonzero(as_tuple=False).squeeze(-1)#找出这些行 索引；[num_nomask]
            col_idx = torch.randint(0, seq_len, (row_idx.numel(),), device=x_start.device)#[num_nomask]
            mask_bool[row_idx, col_idx] = True #强制转为mask

        x_masked = x_start.clone()
        x_masked[mask_bool] = self.mask_token_id
        return x_masked, mask_bool


    def get_inference_mask_ratio(self, step: int) -> float:
        #step:当前步数
        # 当前掩码比例
        ratio = 1.0 - self.alpha_bar_t[step].item()
        ratio = max(
            0.0,
            min(1.0, ratio)
        )
        return ratio