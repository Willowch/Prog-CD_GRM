from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CD_GRM_Loss_Engine(nn.Module):
    # 定义 CD-GRM 的联合损失引擎，继承 nn.Module。
    # 该类负责：
    # 1. 管理 item embedding；
    # 2. 调用 quantizer / scheduler / transformer / seq_ranker / graph_builder；
    # 3. 计算 stage1 与 stage2 的训练损失；
    # 4. 计算 item-level 的排序 logits。

    def __init__(self, item_vocab_size: int, embed_dim: int = 128, m_layers: int = 4,
                codebook_size: int = 256,
                 quantizer: nn.Module = None, scheduler: nn.Module = None,
                 transformer: nn.Module = None, seq_ranker: nn.Module = None, graph_builder=None,
                 cl_loss_fn: nn.Module = None,
                 item_loss_weight: float = 0.0,
                  hybrid_semantic_weight: float = 0.0,
                  stage2_aux_loss_weight: float = 1.0,
                  finetune_item_embedding_stage2: bool = False,
                  item_embedding_stage2_lr_scale: float = 0.5,
                  graph_context_decay: float = 0.8,
                  graph_prior_weight: float = 0.0,
                  use_target_aware_graph_prior: bool = False,
                  graph_prior_sim_scale: float = 1.0,
                  graph_prior_recency_strength: float = 1.0,
                  graph_prior_second_hop_weight: float = 0.0,
                  history_repeat_prior_weight: float = 0.0,
                  last_item_prior_weight: float = 0.0):
        # 初始化函数。
        # 主要输入参数：
        # item_vocab_size: int
        #   item 总数（通常含 PAD）
        # embed_dim: int
        #   item embedding 维度 D
        # m_layers: int
        #   semantic id 的层数 M
        # codebook_size: int
        #   每层 codebook 大小 K
        # quantizer:
        #   残差量化器，输入 [B, D]，输出量化向量、语义 ID、量化损失
        # scheduler:
        #   mask/noise 调度器
        # transformer:
        #   去噪网络
        # seq_ranker:
        #   item 级排序器/融合器
        # graph_builder:
        #   图视图构建器，提供邻接信息与随机游走
        # cl_loss_fn:
        #   对比学习损失函数

        super(CD_GRM_Loss_Engine, self).__init__()
        # 调用父类 nn.Module 的初始化逻辑。

        self.item_vocab_size = item_vocab_size
        # 保存 item 词表大小。
        # 标量 int

        self.embed_dim = embed_dim
        # 保存 embedding 维度 D。
        # 标量 int

        self.m_layers = m_layers
        # 保存 semantic id 的层数 M。
        # 标量 int

        self.item_loss_weight = item_loss_weight
        # item-level 分类损失权重。
        # 标量 float

        self.use_item_ranker = item_loss_weight > 0
        # 是否启用 item ranker 分支。
        # 若 item_loss_weight > 0，则为 True，否则为 False。

        self.hybrid_semantic_weight = hybrid_semantic_weight
        # 混合语义损失权重。
        # 当前这段代码中没有实际使用到。

        self.stage2_aux_loss_weight = stage2_aux_loss_weight
        # stage2 辅助损失总权重。
        # 会乘在 CE + CL + commit 的加权和上。

        self.finetune_item_embedding_stage2 = finetune_item_embedding_stage2
        # stage2 是否微调 item_embedding。
        # 当前该变量主要给外部 trainer/optimizer 配置使用。

        self.item_embedding_stage2_lr_scale = item_embedding_stage2_lr_scale
        # stage2 中 item embedding 的学习率缩放比例。
        # 当前此类内部未直接用到，通常供训练器使用。

        self.graph_context_decay = graph_context_decay #有用
        # 图上下文中的时间衰减系数。
        # 越靠近序列末尾的 item，权重通常越高。

        self.graph_prior_weight = graph_prior_weight #有用
        # 图先验分数的融合权重。
        # >0 时会加到 item logits 上。

        self.use_target_aware_graph_prior = use_target_aware_graph_prior #有用
        # 是否启用 target-aware 图先验。
        # 启用后，历史权重会同时参考 recency 与 target 相似度。

        self.graph_prior_sim_scale = graph_prior_sim_scale #有用
        # target-aware graph prior 中，相似度 logits 的缩放系数。

        self.graph_prior_recency_strength = graph_prior_recency_strength #有用
        # target-aware graph prior 中，时间先验的强度。

        self.graph_prior_second_hop_weight = graph_prior_second_hop_weight
        self.history_repeat_prior_weight = history_repeat_prior_weight
        self.last_item_prior_weight = last_item_prior_weight
        # 二跳邻居先验的权重。
        # >0 时，会把 second-hop 邻居也加入 prior。

        self.log_vars = nn.Parameter(torch.zeros(3))
        # 可学习的不确定性参数，形状: [3]
        # 含义：
        # log_vars[0] -> CE loss 的 log variance
        # log_vars[1] -> CL loss 的 log variance
        # log_vars[2] -> commit loss 的 log variance

        self.item_embedding = nn.Embedding(
            item_vocab_size,    # 所有 item 数量 = V
            embed_dim,          # embedding 维度 = D
            padding_idx=0       # 0 作为 PAD token，对应向量通常不参与有效训练
        )
        # 定义 item embedding 表。
        # 权重张量形状: [V, D]

        nn.init.uniform_(self.item_embedding.weight, -0.05, 0.05)
        # 使用均匀分布初始化 item embedding 权重。
        # self.item_embedding.weight 形状: [V, D]
        # 初始化范围: [-0.05, 0.05]

        self.MASK_TOKEN = codebook_size + 1
        # 定义 MASK token 的 ID。
        # 因为语义 token 实际 code 范围通常是 [1, codebook_size]，
        # 这里把 MASK 设为 codebook_size + 1。

        self.PAD_TOKEN = 0
        # 定义 PAD token 的 ID 为 0。

        self.SID_VOCAB_SIZE = codebook_size + 2
        # Semantic ID 的词表大小。
        # 组成：
        # 0 -> PAD
        # 1~codebook_size -> 有效 code
        # codebook_size+1 -> MASK
        # 因此总大小 = codebook_size + 2

        self.quantizer = quantizer
        # 残差量化器模块。
        # 典型输入: [B, D]
        # 典型输出:
        #   quantized_out: [B, D]
        #   true_sids: [B, M]
        #   commit_loss: [B] 或 [B, ...]（视实现而定）

        self.scheduler = scheduler
        # mask/noise 调度器模块。
        # 负责随机采样时间步 t、给 semantic ids 加噪/加 mask。

        self.transformer = transformer
        # 去噪 Transformer 模块。
        # 典型输入:
        #   history_seq: [B, L]
        #   masked_sids: [B, M]
        #   history_pad_mask: [B, L]
        # 典型输出:
        #   logits: [B, M, SID_VOCAB_SIZE]
        #   target_latent: [B, M, D]

        self.seq_ranker = seq_ranker
        # 序列排序/融合模块。
        # 用于把历史序列语义、target latent、graph context 融合成最终上下文表示。

        self.graph_builder = graph_builder
        # 图构建器模块。
        # 通常提供：
        # adj_matrix: 邻接 item id 表
        # prob_matrix: 邻接概率表
        # sample_view_B(): 采样图视图序列

        self.cl_loss_fn = cl_loss_fn
        # 对比学习损失函数。
        # 通常输入两个 [B, D] 表示，输出标量损失。

        self._tie_embeddings()
        # 调用 embedding 绑定函数，
        # 让 transformer / seq_ranker 中也共享当前 item_embedding。

    def _tie_embeddings(self):
        # 把本类的 item_embedding 绑定给下游模块，确保共享同一套参数。

        if self.transformer is not None:  # 如果 transformer 已经初始化
            self.transformer.item_embedding = self.item_embedding
            # 将 transformer 内部的 item_embedding 指向当前共享 embedding。
            # embedding 权重形状: [V, D]

        if self.seq_ranker is not None:
            self.seq_ranker.item_embedding = self.item_embedding
            # 将 seq_ranker 内部的 item_embedding 也绑定到当前 embedding。
    #根据历史序列 构建图历史 表示； [B,L]->[B,D]
    #每个样本的张量 是根据历史+图邻居 时间权重+概率加权 得到的综合embed表示
    def _build_graph_context(self, history_seq: torch.Tensor) -> torch.Tensor:
        # 根据历史序列构建图上下文表示。
        #
        # 输入：
        # history_seq: [B, L]
        #   B = batch size
        #   L = 历史序列长度
        #
        # 输出：
        # graph_context: [B, D]
        #   每个样本一个图上下文向量

        if self.seq_ranker is None or self.graph_builder is None:
            # 如果没有 seq_ranker 或 graph_builder，就无法构建有效图上下文，
            # 直接返回全 0 向量。

            return torch.zeros(
                history_seq.size(0),   # B
                self.embed_dim,        # D
                device=history_seq.device
            )
            # 返回张量形状: [B, D]

        history_mask = history_seq != self.PAD_TOKEN
        # 构造历史有效位置 mask。
        # history_seq: [B, L]
        # history_mask: [B, L]，bool
        # True 表示该位置不是 PAD。

        seq_len = history_seq.size(1)
        # 获取序列长度 L。
        # 标量 int

        neighbor_ids = self.graph_builder.adj_matrix[history_seq]
        # 根据 history_seq 中每个 item，查图中的邻居 id。
        # 若 adj_matrix 形状为 [V, K]，
        # 则 history_seq: [B, L]
        # 索引后 neighbor_ids: [B, L, K]
        # 其中 K = 每个 item 保留的邻居数

        neighbor_probs = self.graph_builder.prob_matrix[history_seq]
        # 根据 history_seq 中每个 item，查图中的邻居概率。
        # 若 prob_matrix 形状为 [V, K]，
        # 则 neighbor_probs: [B, L, K]

        neighbor_emb = self.item_embedding(neighbor_ids)
        # 取邻居 item 的 embedding。
        # neighbor_ids: [B, L, K]
        # neighbor_emb: [B, L, K, D]

        local_graph_context = (neighbor_emb * neighbor_probs.unsqueeze(-1)).sum(dim=2)
        # 用邻居概率对邻居 embedding 做加权求和，得到每个历史位置的局部图表示。
        # neighbor_probs.unsqueeze(-1): [B, L, K, 1]
        # neighbor_emb * neighbor_probs.unsqueeze(-1): [B, L, K, D]
        # sum(dim=2) 后:
        # local_graph_context: [B, L, D]

        positions = torch.arange(seq_len, device=history_seq.device).unsqueeze(0)
        # 构造位置索引。
        # torch.arange(seq_len): [L]
        # unsqueeze(0) 后:
        # positions: [1, L]

        valid_lengths = history_mask.sum(dim=1, keepdim=True).clamp(min=1)
        # 统计每个样本的有效历史长度。
        # history_mask: [B, L]
        # sum(dim=1, keepdim=True): [B, 1]
        # clamp(min=1) 防止全 PAD 时分母为 0
        # valid_lengths: [B, 1]

        distance_to_last = (valid_lengths - 1 - positions).clamp(min=0).float()
        # 计算每个位置距离“最后一个有效 item”的距离。
        # valid_lengths - 1为最后一个有效历史item的位置
        # 再减去positions，得到每个历史item到最后一个有效历史item的 距离；
        # 例如：若最后一个有效历史item索引为3，seql=5,则可为[3,2,1,0,1]
        # clamp(min=0) 避免负值
        # distance_to_last: [B, L]

        recency_weights = (self.graph_context_decay ** distance_to_last) * history_mask.float()
        # 根据距离构造时间衰减权重。
        # self.graph_context_decay ** distance_to_last: [B, L]
        # history_mask.float(): [B, L]
        # recency_weights: [B, L]

        recency_weights = recency_weights / recency_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        # 对每个样本的时间权重做归一化。
        # recency_weights.sum(dim=1, keepdim=True): [B, 1]
        # 最终 recency_weights: [B, L]

        return (local_graph_context * recency_weights.unsqueeze(-1)).sum(dim=1)
        # 用时间权重对每个位置的局部图上下文加权求和。
        # recency_weights.unsqueeze(-1): [B, L, 1]
        # local_graph_context * recency_weights.unsqueeze(-1): [B, L, D]
        # sum(dim=1) 后输出:
        # graph_context: [B, D]

    #调用seq_ranker返回 融合了 图上下文表示+target_latent的融合表示[B,D]
    def _build_ranker_context(self, history_seq: torch.Tensor, target_latent: torch.Tensor,
                              graph_context: torch.Tensor | None = None) -> torch.Tensor:
        # 构建供 item ranker 使用的上下文表示。
        #
        # 输入：
        # history_seq: [B, L]
        # target_latent: [B, M, D]
        # graph_context: [B, D] 或 None
        #
        # 输出：
        # fused_context: [B, D]

        semantic_context = target_latent.mean(dim=1)
        # 对 target_latent 在 semantic layer 维度做平均池化。
        # target_latent: [B, M, D]
        # semantic_context: [B, D]

        if self.seq_ranker is None:
            # 如果没有 seq_ranker，就直接返回 semantic_context。
            return semantic_context
            # 返回形状: [B, D]

        history_pad_mask = history_seq == self.PAD_TOKEN
        # 构造历史序列的 PAD mask。
        # history_seq: [B, L]
        # history_pad_mask: [B, L]，bool

        if graph_context is None:
            # 如果外部没传 graph_context，就现算一个。
            graph_context = self._build_graph_context(history_seq)
            # graph_context: [B, D]

        fused_context, _ = self.seq_ranker(
            history_seq,       # [B, L]
            semantic_context,  # [B, D]
            graph_context,     # [B, D]
            history_pad_mask   # [B, L]
        )
        # seq_ranker 输出：
        # fused_context: [B, D]
        # 第二个返回值通常是额外中间量/注意力权重/日志项，这里不使用

        return fused_context
        # 返回最终融合后的上下文表示，形状: [B, D]
    #根据历史+图邻居；将neighbor_probs经过 时间权重（target aware)/二跳加分，加入图先验分数中；
    #理解：根据历史+图邻居，每个样本先对物品的交互可能分数；
    def _build_graph_prior_scores(
        self,
        history_seq: torch.Tensor,
        pooled_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 构建 item-level 图先验分数
        #
        # 输入：
        # history_seq: [B, L]
        # pooled_target: [B, D] 或 None
        #
        # 输出：
        # prior_scores: [B, V]
        #   V = item_vocab_size
        #   每个用户对所有 item 的图先验得分

        if self.graph_builder is None or self.graph_prior_weight <= 0:
            # 如果没有图构建器，或图先验权重不大于 0，
            # 则直接返回全 0 分数。

            return torch.zeros(
                history_seq.size(0),   # B
                self.item_vocab_size,  # V
                device=history_seq.device
            )
            # 返回形状: [B, V]

        history_mask = history_seq != self.PAD_TOKEN
        # 有效历史 mask。
        # [B, L]

        batch_size, seq_len = history_seq.shape
        # batch_size = B
        # seq_len = L

        neighbor_ids = self.graph_builder.adj_matrix[history_seq]
        # 取每个历史 item 的一跳邻居 id。
        # 若每个 item 有 K 个邻居：
        # neighbor_ids: [B, L, K]

        neighbor_probs = self.graph_builder.prob_matrix[history_seq]
        # 取每个历史 item 的一跳邻居概率。
        # neighbor_probs: [B, L, K]

        positions = torch.arange(seq_len, device=history_seq.device).unsqueeze(0)
        # 位置索引。
        # positions: [1, L]

        valid_lengths = history_mask.sum(dim=1, keepdim=True).clamp(min=1)
        # 每个样本有效历史长度。
        # valid_lengths: [B, 1]

        distance_to_last = (valid_lengths - 1 - positions).clamp(min=0).float()
        # 位置到最后一个有效 item 的距离。
        # distance_to_last: [B, L]

        recency_weights = (self.graph_context_decay ** distance_to_last) * history_mask.float()
        # 时间衰减权重。
        # recency_weights: [B, L]

        if self.use_target_aware_graph_prior and pooled_target is not None:
            # 若启用 target-aware graph prior，且给定 pooled_target，
            # 则历史位置权重由“相似度 + 时间先验”联合决定。

            recency_logits = self.graph_prior_recency_strength * torch.log(recency_weights.clamp_min(1e-9))
            # 对 recency_weights 取 log，变成 logits 形式。
            # recency_weights: [B, L]
            # recency_logits: [B, L]

            history_emb = self.item_embedding(history_seq)
            # 取历史 item 的 embedding。
            # history_seq: [B, L]
            # history_emb: [B, L, D]

            query = F.normalize(pooled_target, dim=-1)
            # 对 pooled_target 做 L2 归一化。
            # pooled_target: [B, D]
            # query: [B, D]

            history_keys = F.normalize(history_emb, dim=-1)
            # 对历史 item embedding 做 L2 归一化。
            # history_emb: [B, L, D]
            # history_keys: [B, L, D]

            sim_logits = torch.sum(history_keys * query.unsqueeze(1), dim=-1) * self.graph_prior_sim_scale
            # 计算 pooled_target 与每个历史 item 的点积相似度。
            # query.unsqueeze(1): [B, 1, D]
            # history_keys * query.unsqueeze(1): [B, L, D]
            # sum(dim=-1): [B, L]
            # sim_logits: [B, L]

            attn_logits = sim_logits + recency_logits
            # 相似度 logits + 时间先验 logits
            # attn_logits: [B, L]

            attn_logits = attn_logits.masked_fill(~history_mask, -1e9)
            # 将 PAD 位置填成极小值，避免 softmax 后占权重。
            # attn_logits: [B, L]

            history_weights = torch.softmax(attn_logits, dim=-1)
            # 在历史长度维度做 softmax，得到每个历史位置的归一化权重。
            # history_weights: [B, L]
        else:
            # 若不使用 target-aware 版本，则只使用 recency 权重。
            history_weights = recency_weights / recency_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            # history_weights: [B, L]

        weighted_probs = neighbor_probs * history_weights.unsqueeze(-1)
        # 将每个历史位置的邻居概率乘以该历史位置的权重。
        # neighbor_probs: [B, L, K]
        # history_weights.unsqueeze(-1): [B, L, 1]
        # weighted_probs: [B, L, K]

        prior_scores = torch.zeros(
            batch_size,            # B
            self.item_vocab_size,  # V
            device=history_seq.device
        )
        # 初始化全 item 的先验分数表。
        # prior_scores: [B, V]

        prior_scores.scatter_add_(
            1,
            neighbor_ids.reshape(batch_size, -1),
            weighted_probs.reshape(batch_size, -1)
        )
        # 把一跳邻居的加权概率累加到对应 item 的槽位中。
        # 每行（即每个样本）按neighborids指定的物品id，加上对应weighted_probs的分数；
        # neighbor_ids.reshape(batch_size, -1): [B, L*K]
        # weighted_probs.reshape(batch_size, -1): [B, L*K]
        # 沿 dim=1（item 维）进行 scatter_add_
        # 输出 prior_scores 仍是 [B, V]

        if self.graph_prior_second_hop_weight > 0:
            # 若启用二跳图先验，则额外加入二跳邻居贡献。

            second_neighbor_ids = self.graph_builder.adj_matrix[neighbor_ids]
            # 对一跳邻居再查一次邻接表，得到二跳邻居。
            # neighbor_ids: [B, L, K]
            # second_neighbor_ids: [B, L, K, K2]
            # 若一跳和二跳都固定 K，则通常是 [B, L, K, K]

            second_neighbor_probs = self.graph_builder.prob_matrix[neighbor_ids]
            # 一跳邻居到二跳邻居的转移概率。
            # second_neighbor_probs: [B, L, K, K2]

            weighted_second_probs = (
                neighbor_probs.unsqueeze(-1)
                * second_neighbor_probs
                * history_weights.unsqueeze(-1).unsqueeze(-1)
            )
            # 二跳路径概率 = 历史位置权重 × 一跳概率 × 二跳概率
            #
            # neighbor_probs.unsqueeze(-1): [B, L, K, 1]
            # second_neighbor_probs:        [B, L, K, K2]
            # history_weights.unsqueeze(-1).unsqueeze(-1): [B, L, 1, 1]
            # weighted_second_probs: [B, L, K, K2]

            prior_scores.scatter_add_(
                1,
                second_neighbor_ids.reshape(batch_size, -1),
                (self.graph_prior_second_hop_weight * weighted_second_probs).reshape(batch_size, -1)
            )
            # 把二跳邻居的加权概率累加到 prior_scores。
            #
            # second_neighbor_ids.reshape(batch_size, -1): [B, L*K*K2]
            # weighted_second_probs.reshape(batch_size, -1): [B, L*K*K2]
            # 输出 prior_scores 仍是 [B, V]

        if self.history_repeat_prior_weight > 0:
            repeat_scores = history_weights * history_mask.float()
            prior_scores.scatter_add_(
                1,
                history_seq,
                self.history_repeat_prior_weight * repeat_scores
            )

        if self.last_item_prior_weight > 0:
            last_indices = valid_lengths.squeeze(1) - 1
            last_items = history_seq[
                torch.arange(batch_size, device=history_seq.device),
                last_indices
            ]
            last_item_boost = torch.zeros_like(prior_scores)
            last_item_boost.scatter_(
                1,
                last_items.unsqueeze(1),
                self.last_item_prior_weight
            )
            prior_scores = prior_scores + last_item_boost

        prior_scores[:, self.PAD_TOKEN] = 0.0
        # 把 PAD item 的先验分数强制设为 0。
        # prior_scores: [B, V]

        return torch.log1p(prior_scores * float(self.item_vocab_size))
        # 对先验分数进行 log(1+x) 压缩，防止值过大。
        # prior_scores * V: [B, V]
        # 返回形状: [B, V]

    #根据融合的历史表示 与 item_embed做点积，得到分数[B, V]
    #分数也可加上之前的图先验分数
    def predict_item_logits(self, history_seq: torch.Tensor, target_latent: torch.Tensor) -> torch.Tensor:
        # 预测 item-level logits。
        #
        # 输入：
        # history_seq: [B, L]
        # target_latent: [B, M, D]
        #
        # 输出：
        # item_logits: [B, V]

        graph_context = self._build_graph_context(history_seq)
        # 构建图上下文。
        # graph_context: [B, D]

        pooled_target = self._build_ranker_context(history_seq, target_latent, graph_context)  # [B, D]
        # 构建融合后的 target/context 表示。
        # pooled_target: [B, D]

        item_logits = torch.matmul(pooled_target, self.item_embedding.weight.t())  # [B, V]
        # 用 pooled_target 与所有 item embedding 做点积，得到所有 item 的分数。
        #
        # pooled_target: [B, D]
        # self.item_embedding.weight.t(): [D, V]
        # item_logits: [B, V]

        if self.graph_prior_weight > 0:
            # 若启用图先验，则把图先验加到 item logits 上。
            item_logits = item_logits + self.graph_prior_weight * self._build_graph_prior_scores(
                history_seq,
                pooled_target
            )
            # self._build_graph_prior_scores(...) -> [B, V]
            # item_logits 仍为 [B, V]

        item_logits[:, self.PAD_TOKEN] = -float("inf")
        # PAD item 不允许被预测，直接赋为负无穷。
        # item_logits: [B, V]

        return item_logits
        # 返回 item 级预测分数，形状: [B, V]
    #准备阶段2需要的 中间量
    def forward_train(self, history_seq: torch.Tensor, target_item: torch.Tensor):
        # stage2 训练前向流程，返回计算损失所需的中间结果。
        #
        # 输入：
        # history_seq: [B, L]
        # target_item: [B]
        #
        # 输出：
        # logits: [B, M, SID_VOCAB_SIZE]
        # true_sids: [B, M]
        # mask_bool: [B, M]
        # target_latent: [B, M, D]
        # view_b_seq: [B, M]
        # commit_loss: [B] 或其他可按 batch 求均值的形状

        batch_size, seq_len = history_seq.shape
        # 取 batch 大小与序列长度。
        # batch_size = B
        # seq_len = L

        device = history_seq.device
        # 当前输入所在设备，如 cuda:0 / cpu。

        e_target = self.item_embedding(target_item)
        # 查 target item 的 embedding。
        # target_item: [B]
        # e_target: [B, D]

        _, true_sids, commit_loss = self.quantizer(e_target)
        # 用量化器把 target embedding 映射为 semantic ids。
        #
        # e_target: [B, D]
        # quantizer 输出通常为：
        # 第1个返回值: quantized_out -> [B, D]（这里不用，所以用 _ 接）
        # true_sids: [B, M]
        # commit_loss: [B] 或可 batch-mean 的张量

        true_sids = true_sids + 1
        # 将 quantizer 返回的 codebook 下标整体 +1。
        # 原因：
        # codebook 原始索引一般是 [0, K-1]
        # 而这里 semantic token 的 0 保留给 PAD，
        # 所以真实 code 要映射到 [1, K]
        # true_sids: [B, M]

        t = self.scheduler.get_random_t(
            batch_size,  # B
            device
        )
        # 随机采样每个样本的时间步 t。
        # t 的常见形状: [B]

        masked_sids, mask_bool = self.scheduler.add_noise(
            true_sids,
            t
        )
        # 对真实 semantic ids 加噪/加 mask。
        #
        # true_sids: [B, M]
        # t: [B]
        # masked_sids: [B, M]
        # mask_bool: [B, M]，bool
        # True 表示该位置被 mask，需要在 CE 中预测

        history_pad_mask = (history_seq == self.PAD_TOKEN)
        # 历史序列的 PAD mask。
        # history_pad_mask: [B, L]，bool

        logits, target_latent = self.transformer(
            history_seq,       # [B, L]
            masked_sids,       # [B, M]
            history_pad_mask   # [B, L]
        )
        # 调用去噪 transformer。
        # logits: [B, M, SID_VOCAB_SIZE]
        # target_latent: [B, M, D]

        walk_length = self.m_layers
        # 随机游走长度设为 semantic id 层数 M。

        view_b_seq = self.graph_builder.sample_view_B(
            target_item,
            walk_length
        )
        # 采样图视图 B。
        # target_item: [B]
        # walk_length: M
        # view_b_seq: [B, M]

        return logits, true_sids, mask_bool, target_latent, view_b_seq, commit_loss
        # 返回训练所需中间结果：
        # logits: [B, M, SID_VOCAB_SIZE]
        # true_sids: [B, M]
        # mask_bool: [B, M]
        # target_latent: [B, M, D]
        # view_b_seq: [B, M]
        # commit_loss: [B] 或其他

    # 在阶段2的预测损失上加上了 相似度损失
    def calculate_loss(self, history_seq: torch.Tensor, target_item: torch.Tensor,
                       stage: int = 2) -> tuple:
        # 计算训练损失。
        #
        # 输入：
        # history_seq: [B, L]
        # target_item: [B]
        # stage: int
        #   1 -> 量化 + 图视图对比学习
        #   2 -> semantic CE + CL + commit + (可选)item loss
        #
        # 输出：
        # loss_total: 标量
        # loss_ce: 标量
        # loss_cl: 标量

        if stage == 1:
            # =========================
            # Stage 1
            # 计算：
            # 1. quantizer 输出
            # 2. 图视图对比损失
            # 3. commit loss
            # =========================

            e_target = self.item_embedding(target_item)
            # target_item: [B]
            # e_target: [B, D]

            quantized_out, true_sids, commit_loss = self.quantizer(e_target)
            # 量化 target embedding。
            # quantized_out: [B, D]
            # true_sids: [B, M]
            # commit_loss: [B] 或其他可 batch-mean 的形状

            walk_length = self.m_layers
            # 随机游走长度 = semantic 层数 M。

            view_b_seq = self.graph_builder.sample_view_B(target_item, walk_length)  # [B, M]
            # 基于 target item 采样图视图序列。
            # view_b_seq: [B, M]

            e_view_b = self.item_embedding(view_b_seq)  # [B, M, D]
            # 查图视图序列中每个 item 的 embedding。
            # e_view_b: [B, M, D]

            z_b = e_view_b.mean(dim=1)  # [B, D]
            # 在 walk/semantic 长度维上平均池化，得到图视图表示。
            # z_b: [B, D]

            loss_cl = self.cl_loss_fn(quantized_out, z_b)
            # 计算对比损失。
            # quantized_out: [B, D]
            # z_b: [B, D]
            # loss_cl: 标量

            loss_total = loss_cl + commit_loss.mean()
            # 总损失 = 对比损失 + 平均量化损失。
            # loss_total: 标量

            loss_ce = torch.tensor(0.0, device=loss_total.device)
            # stage1 没有 semantic CE，因此用 0 占位。
            # loss_ce: 标量

            return loss_total, loss_ce, loss_cl
            # 返回三个标量：
            # loss_total, loss_ce, loss_cl

        else:
            # =========================
            # Stage 2
            # 计算：
            # 1. semantic CE
            # 2. target latent 与图视图的 CL
            # 3. commit loss
            # 4. 可选的 item-level CE
            # =========================

            logits, true_sids, mask_bool, target_latent, view_b_seq, commit_loss = \
                self.forward_train(history_seq, target_item)
            # 从 forward_train 获取中间结果。
            # logits: [B, M, SID_VOCAB_SIZE]
            # true_sids: [B, M]
            # mask_bool: [B, M]
            # target_latent: [B, M, D]
            # view_b_seq: [B, M]
            # commit_loss: [B] 或其他

            flat_logits = logits.reshape(-1, self.SID_VOCAB_SIZE)  # [B*M, SID_VOCAB_SIZE]
            # 把 logits 展平成二维，方便做交叉熵。
            # logits: [B, M, SID_VOCAB_SIZE]
            # flat_logits: [B*M, SID_VOCAB_SIZE]

            flat_targets = true_sids.reshape(-1)  # [B*M]
            # 把真实 semantic ids 展平。
            # true_sids: [B, M]
            # flat_targets: [B*M]

            flat_mask = mask_bool.reshape(-1)  # [B*M]
            # 把 mask 也展平。
            # mask_bool: [B, M]
            # flat_mask: [B*M]，bool

            if not flat_mask.any():
                # 如果没有任何被 mask 的位置，则无法计算 stage2 的 CE。
                raise RuntimeError("No masked positions found in stage2 CE computation.")

            loss_ce = F.cross_entropy(
                flat_logits[flat_mask],
                flat_targets[flat_mask]
            )
            # 只在被 mask 的位置上计算 semantic CE。
            #
            # flat_logits[flat_mask]: [N_mask, SID_VOCAB_SIZE]
            # flat_targets[flat_mask]: [N_mask]
            # loss_ce: 标量

            z_a = target_latent.mean(dim=1)  # [B, D]
            # 对 transformer 输出的 target latent 做平均池化。
            # target_latent: [B, M, D]
            # z_a: [B, D]

            e_view_b = self.item_embedding(view_b_seq)  # [B, M, D]
            # 图视图序列的 embedding。
            # view_b_seq: [B, M]
            # e_view_b: [B, M, D]

            z_b = e_view_b.mean(dim=1)  # [B, D]
            # 图视图平均池化表示。
            # z_b: [B, D]

            loss_cl = self.cl_loss_fn(z_a, z_b)
            # 计算 stage2 的对比损失。
            # z_a: [B, D]
            # z_b: [B, D]
            # loss_cl: 标量

            commit_loss_mean = commit_loss.mean()
            # 对量化损失取 batch 平均。
            # commit_loss_mean: 标量

            # Loss_i = exp(-si) * li + si
            # 这是 uncertainty weighting 的标准形式之一：
            # 用可学习 log variance 自动平衡多个损失项。

            precision_ce = torch.exp(-self.log_vars[0])
            # CE 项对应的精度系数。
            # self.log_vars[0]: 标量参数
            # precision_ce: 标量

            loss_ce_weighted = precision_ce * loss_ce + self.log_vars[0]
            # 加权后的 CE 损失。
            # loss_ce_weighted: 标量

            precision_cl = torch.exp(-self.log_vars[1])
            # CL 项对应的精度系数。
            # precision_cl: 标量

            loss_cl_weighted = precision_cl * loss_cl + self.log_vars[1]
            # 加权后的 CL 损失。
            # loss_cl_weighted: 标量

            precision_commit = torch.exp(-self.log_vars[2])
            # commit 项对应的精度系数。
            # precision_commit: 标量

            loss_commit_weighted = precision_commit * commit_loss_mean + self.log_vars[2]
            # 加权后的 commit 损失。
            # loss_commit_weighted: 标量

            loss_total = self.stage2_aux_loss_weight * (
                loss_ce_weighted + loss_cl_weighted + loss_commit_weighted
            )
            # stage2 基础总损失。
            # loss_total: 标量

            if self.use_item_ranker:
                # 若启用了 item-level 排序头，则再额外计算 item CE。

                item_logits = self.predict_item_logits(history_seq, target_latent)
                # item_logits: [B, V]

                loss_item = F.cross_entropy(item_logits, target_item)
                # item 级分类损失。
                # item_logits: [B, V]
                # target_item: [B]
                # loss_item: 标量

                loss_total = loss_total + self.item_loss_weight * loss_item
                # 把 item loss 加到总损失中。
                # loss_total: 标量

                loss_ce = loss_ce + loss_item
                # 这里把返回给外部的 loss_ce 也加上了 item loss。
                # 所以这里的 loss_ce 其实变成了：
                # semantic_ce + item_ce
                # loss_ce: 标量

            return loss_total, loss_ce, loss_cl
            # 返回：
            # loss_total: 标量
            # loss_ce: 标量
            # loss_cl: 标量
