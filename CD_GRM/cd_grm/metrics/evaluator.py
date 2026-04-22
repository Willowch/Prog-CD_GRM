import torch
import torch.nn as nn


class TopKEvaluator(nn.Module):
    def __init__(self, k_list: list = [5, 10, 20], codebook_size: int = 256, m_layers: int = 4):
        super(TopKEvaluator, self).__init__()
        self.k_list = k_list
        self.codebook_size = codebook_size
        self.m_layers = m_layers
        self.max_k = max(k_list)

        # 计算 NDCG 的折扣权重
        ranks = torch.arange(1, self.max_k + 1, dtype=torch.float32)

        discounts = 1.0 / torch.log2(ranks + 1.0)
        self.register_buffer("discounts", discounts)
    #计算每个k的recall和ndcg分数
    def evaluate_ranking(self, pred_topk_items: torch.Tensor, target_items: torch.Tensor) -> dict:
        """
            pred_topk_items: [bs, max_k] 每个用户预测出的 top-k item id列表
            target_items: [bs] 每个用户真实目标 item id
        输出:
            metrics_dict: dict:包含 HR@k / NDCG@k / Recall@k

        通过比较得到hits的bool矩阵。再计算ndcg和recall；
        返回每个k的recall和ndcg 平均分数；
        """
        target_items = target_items.to(pred_topk_items.device)#[bs]

        target_expanded = target_items.unsqueeze(1).expand(-1, self.max_k)# target_expanded: [bs, max_k]
        hits = (pred_topk_items == target_expanded).float()# hits:[bs, max_k]
        #命中为 1.0，否则为 0.0

        metrics_dict = {}
        for k in self.k_list:
            hits_k = hits[:, :k]# [bs, k]

            hr_k_tensor = hits_k.sum(dim=1)# hr_k_tensor: [bs]单目标最大为1
            hr_score = hr_k_tensor.mean().item()# float所有用户上的平均命中率

            discount_k = self.discounts[:k].to(hits_k.device)#[k]
            ndcg_k_tensor = (hits_k * discount_k).sum(dim=1)#[bs]
            ndcg_score = ndcg_k_tensor.mean().item()#float所有用户上的平均 NDCG@k

            metrics_dict[f"HR@{k}"] = hr_score
            metrics_dict[f"NDCG@{k}"] = ndcg_score
            metrics_dict[f"Recall@{k}"] = hr_score
        return metrics_dict
    #将sid序列映射为整数
    def _hash_sid_sequences(self, sid_tensor: torch.Tensor, base: int) -> torch.Tensor:
        """
            sid_tensor: [num, m_layer]传入的sid序列
            base: int 进制基数，用于把多位 SID 序列编码成一个整数：codebooksize+1

            hashed: [N]每条 SID 序列经过hash后的整数

        将传入的sid序列都映射为一个整数
        """
        sid_cpu = sid_tensor.detach().to("cpu", dtype=torch.int64)# sid_cpu:[num, m_layer]
        if sid_cpu.ndim != 2:
            raise ValueError(f"Expected a 2D SID tensor, got shape={tuple(sid_cpu.shape)}")
        if sid_cpu.numel() == 0:
            return torch.empty(0, dtype=torch.int64)

        max_token = int(sid_cpu.max().item())# sid_cpu 中最大的 token id
        hash_base = max(base, max_token + 1)# 实际使用的进制，确保大于所有 token 值
        powers = torch.arange(sid_cpu.size(1) - 1, -1, -1, dtype=torch.int64)
        # sid_cpu.size(1) = M
        # powers: [M]
        # 例如 M=4 时，powers = [3, 2, 1, 0]
        weights = (torch.tensor(hash_base, dtype=torch.int64) ** powers).unsqueeze(0)
        # torch.tensor(hash_base) ** powers: [M]
        # weights: [1, M]
        # 每一位的权重，例如 [base^(M-1), ..., base^0]

        return (sid_cpu * weights).sum(dim=1)
        # sid_cpu: [N, M]
        # weights: [1, M]，广播后变成 [N, M]
        # sid_cpu * weights: [N, M]
        # sum(dim=1): [N]
        # 返回每条 SID 序列对应的整数 hash
    #计算无效生成率
    def evaluate_igr(self, generated_sids: torch.Tensor, valid_sid_pool: torch.Tensor) -> float:
        """
        输入:
            generated_sids: [bs, m_layer]
            valid_sid_pool: [num_size, m_layer]
        输出:
            igr: float
        根据生成的物品sid是否存在所有物品sid池中，1-在的/所有
        """

        generated_1d = self._hash_sid_sequences(generated_sids, self.codebook_size + 1)#[bs]
        valid_1d = self._hash_sid_sequences(valid_sid_pool, self.codebook_size + 1)#[num_size]

        is_valid_mask = torch.isin(generated_1d, valid_1d)#[bs]
        valid_ratio = is_valid_mask.float().mean().item()#float

        return 1.0 - valid_ratio
    #计算codebook使用率 sid冲突率
    def evaluate_codebook_health(self, all_item_sids: torch.Tensor) -> dict:
        """
        输入:
            all_item_sids: [num_items, m_layer]生成的所有物品sid
        输出:
            metrics: dict包括每层活跃率、平均活跃率、冲突率

        活跃率：每层unique的code/256；冲突率：1-不同sid/总物品数
        """
        all_item_sids_cpu = all_item_sids.detach().to("cpu", dtype=torch.int64)#[num_items, m_layer]
        # 排除 PAD item
        if all_item_sids_cpu.size(0) > 1:
            all_item_sids_cpu = all_item_sids_cpu[1:]
        num_items, m_layers = all_item_sids_cpu.shape

        metrics = {}
        active_rates = []
        for m in range(m_layers):
            unique_codes = torch.unique(all_item_sids_cpu[:, m])#[num_items]
            active_rate = len(unique_codes) / self.codebook_size #m层利用率
            active_rates.append(active_rate)
            metrics[f"Active_Rate_L{m}"] = active_rate

        metrics["Active_Rate_Mean"] = sum(active_rates) / m_layers#float 4层平均利用率

        unique_sids = torch.unique(all_item_sids_cpu, dim=0)#不同 SID 序列的数量
        metrics["Unique_SID_Count"] = len(unique_sids)
        collision_rate = 1.0 - (len(unique_sids) / float(max(1, num_items)))
        metrics["Collision_Rate"] = collision_rate

        return metrics
