import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrieNode:
    def __init__(self):
        self.children = {}
        self.item_ids = []


class ValidItemTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, sid_seq: list, item_id: int):
        node = self.root
        for sid in sid_seq:
            if sid not in node.children:
                node.children[sid] = TrieNode()
            node = node.children[sid]

        if item_id not in node.item_ids:
            node.item_ids.append(item_id)


class ParallelDenoisingEngine(nn.Module):
    def __init__(self, model: nn.Module, total_steps: int = 10, m_layers: int = 4):
        super(ParallelDenoisingEngine, self).__init__()
        self.model = model
        self.total_steps = total_steps
        self.m_layers = m_layers
        self.item_trie = None

    def invalidate_trie_cache(self):
        self.item_trie = None

    @torch.no_grad()
    def build_item_trie(self):
        # 生成1-item_size的物品原始id；
        # 调用量化层得到所有物品的sid；
        # 根据所有物品sid建树；
        self.item_trie = ValidItemTrie()
        device = self.model.item_embedding.weight.device

        valid_item_ids = torch.arange(
            1,
            self.model.item_vocab_size,
            device=device,
        )

        all_item_embs = self.model.item_embedding(valid_item_ids)
        _, all_true_sids, _ = self.model.quantizer(all_item_embs)
        all_true_sids = all_true_sids + 1

        for sid_seq, item_id in zip(all_true_sids.cpu().tolist(), valid_item_ids.cpu().tolist()):
            self.item_trie.insert(sid_seq, item_id)

        print(
            f"[CD-GRM Inference Engine] Built inference trie with "
            f"{len(valid_item_ids)} items."
        )

    @torch.no_grad()
    def predict_item(self, history_seq: torch.Tensor, top_k: int = 10, full_history_list: list = None):
        # 先建树；对整个批次从全掩码预测，得到final_probs；
        # 对每个样本，用概率为分数，在树上做束搜索得到候选集，并排序；
        # 根据全量历史，对候选集进行筛选，计入结果（top_k,和最大分数的sid）；
        # 返回结果；
        if self.item_trie is None:
            self.build_item_trie()

        batch_size = history_seq.shape[0]
        device = history_seq.device
        history_pad_mask = history_seq == self.model.PAD_TOKEN

        x_t = torch.full(
            (batch_size, self.m_layers),
            self.model.MASK_TOKEN,
            dtype=torch.long,
            device=device,
        )
        final_probs = None

        for t in reversed(range(1, self.total_steps + 1)):
            logits, _ = self.model.transformer(
                history_seq,
                x_t,
                history_pad_mask,
            )
            logits[:, :, self.model.PAD_TOKEN] = -float("inf")
            logits[:, :, self.model.MASK_TOKEN] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            if t == 1:
                final_probs = probs#[bs,m_layer,sid_vocab_size]
                break

            max_probs, pred_ids = torch.max(probs, dim=-1)#[bs,m_layer]
            ratio = self.model.scheduler.get_inference_mask_ratio(t - 1)
            num_mask = int(ratio * self.m_layers)

            if num_mask == 0:
                x_t = pred_ids
                continue

            _, mask_indices = torch.topk(
                max_probs,
                k=num_mask,
                dim=-1,
                largest=False,
            )#[bs,num_mask]

            x_t = pred_ids.clone()
            x_t.scatter_(
                dim=-1,
                index=mask_indices,
                value=self.model.MASK_TOKEN,
            )

        final_log_probs = torch.log(final_probs + 1e-9).cpu().numpy()#[bs,m_layer,sid_vocab_size]

        topk_item_ids_batch = []
        gen_sids_batch = []

        #束搜索
        for batch_idx in range(batch_size):
            user_log_probs = final_log_probs[batch_idx]
            queue = [(self.item_trie.root, 0.0, [])]

            for layer_idx in range(self.m_layers):
                next_queue = []
                for node, current_score, path in queue:
                    for sid, child_node in node.children.items():
                        step_score = user_log_probs[layer_idx, sid]
                        next_queue.append(
                            (
                                child_node,
                                current_score + step_score,
                                path + [sid],
                            )
                        )

                next_queue.sort(key=lambda x: x[1], reverse=True)
                beam_width = max(100, top_k * 5)
                queue = next_queue[:beam_width]

            valid_candidates = []
            for node, score, path in queue:
                for item_id in node.item_ids:
                    valid_candidates.append(
                        (item_id, score, path)
                    )

            valid_candidates.sort(key=lambda x: x[1], reverse=True)

            user_top_items = []
            user_top_sids = None

            if full_history_list is not None:
                seen_items = set(full_history_list[batch_idx])
            else:
                user_history = history_seq[batch_idx].cpu().tolist()
                seen_items = {
                    item for item in user_history if item != self.model.PAD_TOKEN
                }

            for item_id, score, path in valid_candidates:
                if item_id in seen_items:
                    continue

                seen_items.add(item_id)
                user_top_items.append(item_id)
                if user_top_sids is None:
                    user_top_sids = path

                if len(user_top_items) >= top_k:
                    break

            while len(user_top_items) < top_k:
                user_top_items.append(self.model.PAD_TOKEN)

            topk_item_ids_batch.append(user_top_items)
            gen_sids_batch.append(
                user_top_sids if user_top_sids is not None else [0] * self.m_layers
            )

        topk_item_ids = torch.tensor(topk_item_ids_batch, device=device)#[bs,top-k]
        gen_sids = torch.tensor(gen_sids_batch, device=device)#[bs,m_layer]
        return topk_item_ids, gen_sids
