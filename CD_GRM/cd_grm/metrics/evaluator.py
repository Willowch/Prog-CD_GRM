import torch
import torch.nn as nn
import math


class TopKEvaluator(nn.Module):
    """
    """
    def __init__(self, k_list: list = [5, 10, 20], codebook_size: int = 256, m_layers: int = 4):
        #初始化ndcg权重
        super(TopKEvaluator, self).__init__()
        self.k_list = k_list
        self.codebook_size = codebook_size
        self.m_layers = m_layers
        self.max_k = max(k_list)

        ranks = torch.arange(1, self.max_k + 1, dtype=torch.float32)
        # discounts = 1 / log2(rank+1)
        discounts = 1.0 / torch.log2(ranks + 1.0)#[max_k]
        self.register_buffer('discounts', discounts)

    def evaluate_ranking(self, pred_topk_items: torch.Tensor, target_items: torch.Tensor) -> dict:
        """
        Args:
            pred_topk_items:[batch_size, max_k]预测数据
            target_items:[batch_size]目标数据
        目标物品扩充，与pred对比；
        求sum的均值为平均；乘以ndcg权重求平均；
        返回每个k的recall和ndcg；
        """
        target_items = target_items.to(pred_topk_items.device)
        target_expanded = target_items.unsqueeze(1)# [batch_size,1]
        target_expanded = target_expanded.expand(-1, self.max_k)# [batch_size,max_k]
        hits = (pred_topk_items == target_expanded).float()#hits: [batch_size, max_k]

        metrics_dict = {}
        for k in self.k_list:
            hits_k = hits[:, :k]# shape: [batch_size, k]
            hr_k_tensor = hits_k.sum(dim=1)# shape: [batch_size]
            hr_score = hr_k_tensor.mean().item()

            discount_k = self.discounts[:k].to(hits_k.device)# shape: [k]
            ndcg_k_tensor = (hits_k * discount_k).sum(dim=1)# shape: [batch_size, k]
            ndcg_score = ndcg_k_tensor.mean().item()

            metrics_dict[f'HR@{k}'] = hr_score
            metrics_dict[f'NDCG@{k}'] = ndcg_score
            metrics_dict[f'Recall@{k}'] = hr_score
        return metrics_dict

    #无效生成率
    def evaluate_igr(self, generated_sids: torch.Tensor, valid_sid_pool: torch.Tensor) -> float:
        #1-生成的物品id在物品id池概率；

        device = generated_sids.device
        powers = torch.arange(#[3,2,1,0]
            self.m_layers - 1,
            -1,
            -1,
            dtype=torch.long,
            device=device
        )
        base = torch.tensor(self.codebook_size, dtype=torch.long, device=device)
        weights = (base ** powers).unsqueeze(1) #[4]

        generated_1d = torch.matmul(
            generated_sids.long(),  # 直接使用整型
            weights
        ).squeeze(-1)  #[gen_num]

        valid_1d = torch.matmul(
            valid_sid_pool.to(device).long(),  # 直接使用整型
            weights
        ).squeeze(-1) #[sid_size]
        is_valid_mask = torch.isin(generated_1d, valid_1d)#[gen_num]
        valid_ratio = is_valid_mask.float().mean().item()
        igr_score = 1.0 - valid_ratio
        # 返回 IGR 指标
        return igr_score

    #计算 codebook使用率 生成sid冲突率
    def evaluate_codebook_health(self, all_item_sids: torch.Tensor) -> dict:
        """
        all_item_sids: 生成的整个物品池量化后的 Semantic ID: [num_items, m_layers]
        根据生成的所有物品id池计算 codebook使用率 和 生成sid冲突率；
        codebook使用率：每层不同sid/256
        生成sid冲突率：1-不同sid/所有物品；
        """

        num_items, m_layers = all_item_sids.shape#num_items:item总数
        metrics = {}
        device = all_item_sids.device

        active_rates = []
        for m in range(m_layers):
            # 这一层codebook被使用过的唯一Code数量
            unique_codes = torch.unique(all_item_sids[:, m])
            active_rate = len(unique_codes) / self.codebook_size#这一层的code使用率
            active_rates.append(active_rate)
            metrics[f'Active_Rate_L{m}'] = active_rate

        metrics['Active_Rate_Mean'] = sum(active_rates) / m_layers# 计算所有层的平均利用率

        # 2. 整体编码冲突率 (Collision Rate)
        # 1 - (不同SID数量 / item数量)
        powers = torch.arange(
            m_layers - 1,
            -1,
            -1,
            dtype=torch.long,  # 修复精度丢失
            device=device
        )
        base = torch.tensor(self.codebook_size, dtype=torch.long, device=device)
        weights = (base ** powers).unsqueeze(1)
        sids_1d = torch.matmul(
            all_item_sids.long(),  # 保持长整型运算
            weights
        ).squeeze(-1)


        unique_sids = torch.unique(sids_1d)
        # 1 - (不同SID数量 / item数量)
        collision_rate = 1.0 - (len(unique_sids) / float(max(1, num_items)))
        metrics['Collision_Rate'] = collision_rate

        return metrics
