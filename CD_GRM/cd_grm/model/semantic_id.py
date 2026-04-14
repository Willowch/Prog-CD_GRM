import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualQuantizer(nn.Module):

    def __init__(self,
                 num_layers: int = 4,        # 量化层数
                 codebook_size: int = 256,   # 每层 codebook 大小
                 embed_dim: int = 128,       # 向量维度
                 commitment_cost: float = 0.25,#损失衰减率
                 decay: float = 0.99,        # EMA衰减率
                 eps: float = 1e-5,          # 平滑率
                 restart_thres: float = 1.0,  # 死码阈值
                 dead_code_warmup_steps: int = 100,
                 dead_code_patience: int = 100,
                 dead_code_min_usage_ratio: float = 0.1
                 ):

        super(ResidualQuantizer, self).__init__()

        # 保存参数
        self.num_layers = num_layers
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.commitment_cost = commitment_cost

        # EMA参数
        self.decay = decay
        self.eps = eps
        self.restart_thres = restart_thres

        self.dead_code_warmup_steps = dead_code_warmup_steps #死码开启步数
        self.dead_code_patience = dead_code_patience #死码开启的不用步数
        self.dead_code_min_usage_ratio = dead_code_min_usage_ratio #最小使用率

        self.freeze_codebook = False #codebook更新控制
        #残差量化步数：每批次调用一次+1
        self.register_buffer('ema_step', torch.zeros(1, dtype=torch.long))


        # 均匀初始化范围[-1/sqrt(D), 1/sqrt(D)]
        bound = 1.0 / math.sqrt(self.embed_dim)
        # 创建多个 codebook
        for m in range(num_layers):
            codebook = torch.empty(codebook_size, embed_dim)
            # 均匀初始化
            nn.init.uniform_(codebook, -bound, bound)
            self.register_buffer(f'codebook_{m}', codebook)

            # 每个 code 被选中的次数
            self.register_buffer(
                f'cluster_size_{m}',
                torch.zeros(codebook_size)
            )
            #每个 code 被选中的量化 embedding 的平均值
            self.register_buffer(
                f'embed_avg_{m}',
                codebook.clone()
            )
            # 每个code的持续未被调用步数
            self.register_buffer(
                f'inactive_steps_{m}',
                torch.zeros(codebook_size, dtype=torch.long)
            )


    def forward(self, x: torch.Tensor):
        '''
         逐层量化：计算每个输入向量到每个code向量的距离[bs,cs]；
         每个输入向量选出最小距离下标即为此层sid；
         根据被选的code向量，更新引用次数，和引用累加embd；
         计算使用率 持续未使用次数 指标，选出死码位置，对死码位置进行初始化；
         更新此层codebook；
         计算累加向量 和 损失的均方误差；
        '''
        #x:[batch_size, embed_dim]经过embedding的批次目标物品向量

        residual = x #当前量化向量
        quantized_out = torch.zeros_like(x)#量化累积向量
        semantic_ids = []
        # ema步数+1
        if self.training and not self.freeze_codebook:
            self.ema_step += 1

        for m in range(self.num_layers):
            codebook = getattr(self, f'codebook_{m}')
            residual_sq = torch.sum(# ||x||²
                residual ** 2,
                dim=-1,
                keepdim=True
            )#[bs]
            codebook_sq = torch.sum(# ||e||²
                codebook ** 2,
                dim=-1
            ).unsqueeze(0)#[cs]
            cross_term = torch.matmul(# x·e
                residual,
                codebook.t()
            )#[bs,bs]
            #广播机制：residual_sq + codebook_sq：[batchsize]->[bs,cs]
            distances = residual_sq + codebook_sq - 2 * cross_term
            min_indices = torch.argmin(#[bs]
                distances,
                dim=-1
            )
            semantic_ids.append(min_indices)
            q_m = F.embedding( #去codebook中取出对应的向量 组合
                min_indices,
                codebook
            )# [bs, embedsize]

            #EMA更新codebook
            if self.training and not self.freeze_codebook:

                cluster_size = getattr(self, f'cluster_size_{m}')
                embed_avg = getattr(self, f'embed_avg_{m}')
                inactive_steps = getattr(self, f'inactive_steps_{m}')
                # one-hot
                encodings = F.one_hot(
                    min_indices,#[bs]要独热编码的 整数
                    self.codebook_size#独热编码维度
                ).float()#[bs,cs]
                #每个code此次被用次数
                batch_hits = encodings.sum(0)

                #cluster_size = cluster_size * self.decay + batch_hits * (1 - self.decay)
                #batch_hits即为每个code的此批次 此codebook层调用次数
                #为新老加和处理
                cluster_size.data.mul_(self.decay).add_(
                    batch_hits,
                    alpha=1 - self.decay
                )
                #[codebook_size,embed_size]计算每个code被选中的 向量之和
                embed_sum = torch.matmul(
                    encodings.t(),
                    residual.detach()
                )
                #embed_avg = embed_avg * self.decay + embed_sum * (1 - self.decay)
                embed_avg.data.mul_(self.decay).add_(
                    embed_sum,
                    alpha=1 - self.decay
                )
                # 更新此codebook未被调用次数
                inactive_steps.data.add_(1)
                inactive_steps.data[batch_hits > 0] = 0
                # cluster_size修正
                current_ema_step = max(int(self.ema_step.item()), 1)

                bias_correction = 1.0 - (self.decay ** current_ema_step)#修正率
                effective_cluster_size = cluster_size / max(bias_correction, self.eps)

                expected_hits = residual.size(0) / float(self.codebook_size)
                min_usage = self.dead_code_min_usage_ratio * expected_hits #最小使用率；
                #死码处理
                dead_mask = torch.zeros_like(cluster_size, dtype=torch.bool)
                if current_ema_step >= self.dead_code_warmup_steps:
                    low_usage_mask = effective_cluster_size < min_usage #使用频率小于最小预期的位置
                    patience_mask = inactive_steps >= self.dead_code_patience#持续未更新位置小于最小预期的位置
                    #【可改为|】
                    dead_mask = low_usage_mask & patience_mask#[codebook_size]死码位置：两者都符合的位置
                if dead_mask.any():#如果有死码
                    num_dead = dead_mask.sum().item()#死码的数量
                    rand_indices = torch.randint(
                        0,
                        residual.size(0),
                        (num_dead,),
                        device=residual.device
                    )
                    revived_vectors = residual.detach()[rand_indices]#从量化向量中随机取 作为死码的起点
                    revived_vectors = revived_vectors + torch.randn_like(
                        revived_vectors
                    ) * 1e-5 #加上噪声
                    # [FIX-DEAD-CODE] Write back raw EMA-scale counts after bias-corrected thresholding.
                    raw_min_usage = min_usage * max(bias_correction, self.eps)
                    cluster_size.data[dead_mask] = raw_min_usage
                    embed_avg.data[dead_mask] = revived_vectors * raw_min_usage
                    inactive_steps.data[dead_mask] = 0

                n = cluster_size.sum()
                cluster_size_smoothed = ( #平滑处理调用次数
                    (cluster_size + self.eps) /
                    (n + self.codebook_size * self.eps) * n
                )
                #更新codebook
                codebook.data.copy_(
                    embed_avg /
                    cluster_size_smoothed.unsqueeze(-1)
                )

            quantized_out = quantized_out + q_m#累加 模拟向量
            residual = residual - q_m #更新残差

        semantic_ids_tensor = torch.stack(
            semantic_ids,
            dim=-1
        )#[bs,m_layer]

        loss_encoder_commit = F.mse_loss(
            quantized_out.detach(),
            x
        )#输入和 量化结果 的均方损失
        commit_loss = self.commitment_cost * loss_encoder_commit
        #直通评估器
        quantized_out = x + (quantized_out - x).detach()

        return quantized_out, semantic_ids_tensor, commit_loss
