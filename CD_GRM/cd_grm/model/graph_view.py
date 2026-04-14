import os
import torch
import torch.nn as nn
from collections import defaultdict


class GraphViewBuilder(nn.Module):
    def __init__(self, item_vocab_size: int, dataset=None, max_degree: int = 50, window_size: int = 3,
                 walk_temp: float = 0.8):
        """
        初始化图构建器
        参数说明：
        item_vocab_size: item 词表大小
        dataset: RecBole Dataset 对象
        max_degree: 截断的最大邻居数量
        window_size: [新增] 序列共现的滑动窗口大小，用于捕获短程时序
        walk_temp: [新增] 随机游走的温度系数。小于1.0会使得游走更偏向于高频时序邻居，减少时序漂移：用于放大概率差距
        """
        super(GraphViewBuilder, self).__init__()

        self.item_vocab_size = item_vocab_size
        self.max_degree = max_degree
        self.window_size = window_size
        self.walk_temp = walk_temp

        # 构建包含位置衰减的时序有向图
        #adj为物品id prob为转移概率
        adj_matrix, prob_matrix = self._build_graph(
            dataset,
            item_vocab_size,
            max_degree,
            window_size
        )

        self.register_buffer('adj_matrix', adj_matrix)
        self.register_buffer('prob_matrix', prob_matrix)

    def _build_graph(self, dataset, item_vocab_size: int, max_degree: int, window_size: int):
        """
        离线构建 Item 共现图 (加入时序位置衰减)

        将dataset中的每个用户的最长交互序列找出（拼接了targetid）；
        以每个历史序列的（除了最后一个）为起始，向后看window_size个物品；
        根据距离系数(1/dis)加上到每个 后看物品的分数；
        每个物品选出分数最高的max_degree个物品；
        将max_degree个物品分数归一化为概率；

        """
        dataset_name = getattr(dataset, 'dataset_name', "unknown") if dataset is not None else "unknown"
        cache_dir = "CD_GRM/dataset/cache"
        os.makedirs(cache_dir, exist_ok=True)

        cache_file = os.path.join(
            cache_dir,
            f"seq_graph_{dataset_name}_vsize{item_vocab_size}_mdeg{max_degree}_win{window_size}_trainsplit.pt"
        )

        if os.path.exists(cache_file):
            print(f"[Graph Builder] Loading temporal-decayed graph from {cache_file}...")
            return torch.load(cache_file)

        print("[Graph Builder] Compiling temporal item co-occurrence graph with position decay...")

        # 记录加权共现分数，改为 float 以支持小数衰减权重
        transition_counts = defaultdict(lambda: defaultdict(float))

        if dataset is not None:
            all_seqs = dataset['item_id_list'].numpy()
            all_targets = dataset['item_id'].numpy()
            all_users = dataset['user_id'].numpy()

            # 利用字典按用户去重，提取每个用户的全量无重复序列:dataset中可能有同一用户的多个历史序列
            user_full_seqs = {}
            for i in range(len(dataset)): #逐个取出样本
                uid = all_users[i]
                seq = all_seqs[i]
                valid_seq = seq[seq > 0].tolist()  # 去除 padding 并转为 list
                target = all_targets[i]

                # 将历史序列与目标物品拼接
                full_seq = valid_seq + [target]

                # 保留该用户最长的那条序列（即全量训练历史）
                if uid not in user_full_seqs or len(full_seq) > len(user_full_seqs[uid]):
                    #若没有此用户序列/当前用户序列较长 则覆盖
                    user_full_seqs[uid] = full_seq

            for uid, valid_seq in user_full_seqs.items():
                seq_len = len(valid_seq)

                # 引入滑动窗口与位置衰减
                for j in range(seq_len - 1):
                    src = valid_seq[j]

                    # 在滑动窗口内向前看，捕获短程连续性
                    max_step = min(window_size + 1, seq_len - j)
                    for k in range(1, max_step):
                        dst = valid_seq[j + k]

                        # 位置衰减逻辑
                        decay_weight = 1.0 / k
                        transition_counts[src][dst] += decay_weight

        adj_matrix = torch.zeros((item_vocab_size, max_degree), dtype=torch.long)
        prob_matrix = torch.zeros((item_vocab_size, max_degree), dtype=torch.float32)

        for src, neighbors in transition_counts.items():
            if src == 0 or src >= item_vocab_size:
                continue

            # 按累积时序衰减分数排序
            sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)
            top_k_neighbors = sorted_neighbors[:max_degree]

            if not top_k_neighbors:
                continue

            dst_nodes = [x[0] for x in top_k_neighbors] #物品id
            counts = [x[1] for x in top_k_neighbors]#加权分数

            dst_tensor = torch.tensor(dst_nodes, dtype=torch.long)
            count_tensor = torch.tensor(counts, dtype=torch.float32)

            num_edges = len(dst_tensor)
            adj_matrix[src, :num_edges] = dst_tensor
            prob_matrix[src, :num_edges] = count_tensor / count_tensor.sum()

        print(f"[Graph Builder] Temporal graph cached. Window={window_size}, MaxDegree={max_degree}.")
        torch.save((adj_matrix, prob_matrix), cache_file)

        return adj_matrix, prob_matrix

    def sample_view_B(self, target_items: torch.Tensor, walk_length: int) -> torch.Tensor:
        """
        带时序偏置的随机游走
        target_items:[batch_size]原始的物品id

        根据步长，构造batchsize个物品每步的 物品id
        根据当前物品id，和图中的概率转移矩阵，multinomial随机出节点 加入当前步
        返回转置矩阵即为每个样本的 随机序列
        """
        batch_size = target_items.shape[0]
        device = target_items.device

        walk_sequences = torch.zeros((walk_length, batch_size), dtype=torch.long, device=device)
        current_nodes = target_items

        for step in range(walk_length):
            current_probs = self.prob_matrix[current_nodes]#[batch_size,max_degree]
            sum_probs = current_probs.sum(dim=-1)
            isolated_nodes_mask = sum_probs == 0

            if isolated_nodes_mask.any():
                current_probs[isolated_nodes_mask, 0] = 1.0

            #通过温度系数 锐化概率
            if self.walk_temp != 1.0:
                current_probs = (current_probs + 1e-9) ** (1.0 / self.walk_temp)
                # 重新归一化
                current_probs = current_probs / current_probs.sum(dim=-1, keepdim=True)

            sampled_indices = torch.multinomial(current_probs, num_samples=1) #返回选取概率的 下标
            sampled_indices_flat = sampled_indices.squeeze(-1)

            next_nodes = self.adj_matrix[current_nodes, sampled_indices_flat] #通过选取的下标，取出物品id
            next_nodes = torch.where(isolated_nodes_mask, current_nodes, next_nodes) #isolated_nodes_mask为true时，仍然用current_nodes

            walk_sequences[step] = next_nodes
            current_nodes = next_nodes

        return walk_sequences.t()#[batch_size,walk_length]
