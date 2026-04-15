import torch
import torch.nn as nn


class TopKEvaluator(nn.Module):
    def __init__(self, k_list: list = [5, 10, 20], codebook_size: int = 256, m_layers: int = 4):
        super(TopKEvaluator, self).__init__()
        self.k_list = k_list
        self.codebook_size = codebook_size
        self.m_layers = m_layers
        self.max_k = max(k_list)

        ranks = torch.arange(1, self.max_k + 1, dtype=torch.float32)
        discounts = 1.0 / torch.log2(ranks + 1.0)
        self.register_buffer("discounts", discounts)

    def evaluate_ranking(self, pred_topk_items: torch.Tensor, target_items: torch.Tensor) -> dict:
        target_items = target_items.to(pred_topk_items.device)
        target_expanded = target_items.unsqueeze(1).expand(-1, self.max_k)
        hits = (pred_topk_items == target_expanded).float()

        metrics_dict = {}
        for k in self.k_list:
            hits_k = hits[:, :k]
            hr_k_tensor = hits_k.sum(dim=1)
            hr_score = hr_k_tensor.mean().item()

            discount_k = self.discounts[:k].to(hits_k.device)
            ndcg_k_tensor = (hits_k * discount_k).sum(dim=1)
            ndcg_score = ndcg_k_tensor.mean().item()

            metrics_dict[f"HR@{k}"] = hr_score
            metrics_dict[f"NDCG@{k}"] = ndcg_score
            metrics_dict[f"Recall@{k}"] = hr_score
        return metrics_dict

    def _hash_sid_sequences(self, sid_tensor: torch.Tensor, base: int) -> torch.Tensor:
        sid_cpu = sid_tensor.detach().to("cpu", dtype=torch.int64)
        if sid_cpu.ndim != 2:
            raise ValueError(f"Expected a 2D SID tensor, got shape={tuple(sid_cpu.shape)}")

        if sid_cpu.numel() == 0:
            return torch.empty(0, dtype=torch.int64)

        max_token = int(sid_cpu.max().item())
        hash_base = max(base, max_token + 1)
        powers = torch.arange(sid_cpu.size(1) - 1, -1, -1, dtype=torch.int64)
        weights = (torch.tensor(hash_base, dtype=torch.int64) ** powers).unsqueeze(0)
        return (sid_cpu * weights).sum(dim=1)

    def evaluate_igr(self, generated_sids: torch.Tensor, valid_sid_pool: torch.Tensor) -> float:
        # In inference, valid SIDs are shifted by +1, so the hash base must be
        # larger than codebook_size to avoid collisions.
        generated_1d = self._hash_sid_sequences(generated_sids, self.codebook_size + 1)
        valid_1d = self._hash_sid_sequences(valid_sid_pool, self.codebook_size + 1)

        is_valid_mask = torch.isin(generated_1d, valid_1d)
        valid_ratio = is_valid_mask.float().mean().item()
        return 1.0 - valid_ratio

    def evaluate_codebook_health(self, all_item_sids: torch.Tensor) -> dict:
        all_item_sids_cpu = all_item_sids.detach().to("cpu", dtype=torch.int64)
        num_items, m_layers = all_item_sids_cpu.shape
        metrics = {}

        active_rates = []
        for m in range(m_layers):
            unique_codes = torch.unique(all_item_sids_cpu[:, m])
            active_rate = len(unique_codes) / self.codebook_size
            active_rates.append(active_rate)
            metrics[f"Active_Rate_L{m}"] = active_rate

        metrics["Active_Rate_Mean"] = sum(active_rates) / m_layers

        unique_sids = torch.unique(all_item_sids_cpu, dim=0)
        collision_rate = 1.0 - (len(unique_sids) / float(max(1, num_items)))
        metrics["Collision_Rate"] = collision_rate

        return metrics
