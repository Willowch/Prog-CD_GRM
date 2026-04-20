from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenoisingTransformer(nn.Module):
    """
    去噪 Transformer：
    输入用户历史 item 序列 history_seq 和被 mask 的目标 semantic id 序列 masked_sid，
    输出每个 semantic id 位置上的分类 logits，以及目标位置的隐藏表示 target_latent。
    """

    def __init__(
        self,
        item_vocab_size: int,
        sid_vocab_size: int,
        embed_dim: int = 128,
        num_layers: int = 2,
        nhead: int = 4,
        max_seq_len: int = 200,
        dropout: float = 0.1
    ):
        # item_vocab_size: item 词表大小（含 PAD）
        # sid_vocab_size: semantic id 词表大小（含 PAD）
        # embed_dim:      embedding 维度 D
        # num_layers:     TransformerEncoder 堆叠层数
        # nhead:          多头注意力头数
        # max_seq_len:    历史序列最大长度
        # dropout:        dropout 概率
        #
        # 本函数主要初始化：
        # 1. item embedding
        # 2. sid embedding
        # 3. position embedding
        # 4. Transformer encoder
        # 5. 预测层、归一化层、dropout 层

        super(DenoisingTransformer, self).__init__()
        # 调用父类 nn.Module 的初始化逻辑。

        self.embed_dim = embed_dim
        # 保存 embedding 维度 D。
        # self.embed_dim: int

        self.item_embedding = nn.Embedding(
            item_vocab_size,      # item 词表大小
            embed_dim,            # 每个 item 对应的 embedding 维度 D
            padding_idx=0         # 索引 0 表示 PAD，PAD 的 embedding 会特殊处理
        )
        # self.item_embedding.weight 形状: [item_vocab_size, embed_dim]
        # 输入:
        #   history_seq: [B, L]
        # 输出:
        #   e_hist: [B, L, D]

        self.sid_embedding = nn.Embedding(
            sid_vocab_size,       # semantic id 词表大小
            embed_dim,            # 每个 sid 对应的 embedding 维度 D
            padding_idx=0         # 索引 0 表示 PAD
        )
        # self.sid_embedding.weight 形状: [sid_vocab_size, embed_dim]
        # 输入:
        #   masked_sid: [B, M]
        # 输出:
        #   e_target: [B, M, D]

        self.position_embedding = nn.Embedding(
            max_seq_len + 50,     # 位置编码表长度，给历史长度 + 目标 token 预留空间
            embed_dim             # 每个位置的 embedding 维度 D
        )
        # self.position_embedding.weight 形状: [max_seq_len + 50, D]
        # 输入:
        #   positions: [T]
        # 输出:
        #   pos_emb: [T, D]
        # 其中 T = L + M

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,            # 输入特征维度 D
            nhead=nhead,                  # 多头注意力头数 H
            dim_feedforward=embed_dim * 4,# 前馈网络隐藏层维度，一般为 4D
            dropout=dropout,              # dropout 概率
            activation='gelu',            # 激活函数
            batch_first=True              # 输入输出格式为 [B, T, D]
        )
        # encoder_layer 输入形状:  [B, T, D]
        # encoder_layer 输出形状:  [B, T, D]

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,                # 单层 encoder
            num_layers=num_layers         # 堆叠 num_layers 层
        )
        # self.transformer_encoder 输入形状: [B, T, D]
        # self.transformer_encoder 输出形状: [B, T, D]

        self.prediction_head = nn.Linear(
            embed_dim,                    # 输入维度 D
            sid_vocab_size                # 输出维度为 sid 词表大小
        )
        # 输入:
        #   target_latent: [B, M, D]
        # 输出:
        #   logits: [B, M, sid_vocab_size]

        self.layer_norm = nn.LayerNorm(embed_dim)
        # LayerNorm 在最后一个维度 D 上做归一化。
        # 输入输出形状不变:
        #   [B, T, D] -> [B, T, D]

        self.dropout = nn.Dropout(dropout)
        # Dropout 不改变张量形状。
        # 输入输出形状不变:
        #   [B, T, D] -> [B, T, D]

    def forward(
        self,
        history_seq: torch.Tensor,
        masked_sid: torch.Tensor,
        history_pad_mask: torch.Tensor = None
    ):
        """
        参数:
            history_seq:
                用户历史 item 序列
                形状: [B, L]
                B = batch_size
                L = 历史序列长度

            masked_sid:
                被 mask / 加噪后的目标 semantic id 序列
                形状: [B, M]
                M = semantic id 层数 / token 数

            history_pad_mask:
                历史序列的 PAD 掩码
                形状: [B, L]
                True  表示该位置是 PAD，需要 mask
                False 表示该位置是有效 token

        返回:
            logits:
                每个目标 semantic token 位置对整个 sid 词表的分类 logits
                形状: [B, M, sid_vocab_size]

            target_latent:
                目标位置对应的隐藏表示
                形状: [B, M, D]
        """

        batch_size, seq_len = history_seq.shape
        # history_seq: [B, L]
        # batch_size = B
        # seq_len = L

        _, m_layers = masked_sid.shape
        # masked_sid: [B, M]
        # m_layers = M

        total_len = seq_len + m_layers
        # 拼接后的总 token 长度 T = L + M

        e_hist = self.item_embedding(history_seq)
        # history_seq: [B, L]
        # self.item_embedding(history_seq) -> e_hist: [B, L, D]
        # 含义：把历史 item id 查表映射为连续向量表示

        e_target = self.sid_embedding(masked_sid)
        # masked_sid: [B, M]
        # self.sid_embedding(masked_sid) -> e_target: [B, M, D]
        # 含义：把目标 semantic id token 查表映射为连续向量表示

        h_0 = torch.cat([e_hist, e_target], dim=1)
        # e_hist:   [B, L, D]
        # e_target: [B, M, D]
        # 在序列维 dim=1 上拼接
        # h_0:      [B, L + M, D] = [B, T, D]
        # 含义：把“历史序列表示”和“目标语义 token 表示”拼成一个长序列输入 Transformer

        positions = torch.arange(
            total_len,
            device=history_seq.device
        )
        # torch.arange(total_len): [T]
        # 内容一般为 [0, 1, 2, ..., T-1]
        # 这些是位置索引。

        pos_emb = self.position_embedding(positions)
        # positions: [T]
        # position_embedding(positions) -> pos_emb: [T, D]
        # 含义：得到每个位置对应的位置向量

        pos_emb_expanded = pos_emb.unsqueeze(0).expand(
            batch_size,
            total_len,
            self.embed_dim
        )
        # pos_emb: [T, D]
        # pos_emb.unsqueeze(0): [1, T, D]
        # expand(batch_size, total_len, self.embed_dim): [B, T, D]
        # 含义：把同一套位置向量广播/扩展到 batch 中每个样本

        h_0 = h_0 + pos_emb_expanded
        # h_0:              [B, T, D]
        # pos_emb_expanded: [B, T, D]
        # 相加后 h_0:       [B, T, D]
        # 含义：给 token embedding 加上位置编码

        h_0 = self.layer_norm(h_0)
        # h_0: [B, T, D] -> [B, T, D]
        # 含义：对最后一维 D 做层归一化，稳定训练

        h_0 = self.dropout(h_0)
        # h_0: [B, T, D] -> [B, T, D]
        # 含义：做 dropout，缓解过拟合

        if history_pad_mask is not None:
            # 如果历史序列提供了 PAD mask，就需要构造完整输入序列的 mask

            target_pad_mask = torch.zeros(
                (batch_size, m_layers),
                dtype=torch.bool,
                device=history_seq.device
            )
            # 新建目标部分的 PAD mask
            # 因为这里默认目标 semantic token 都是有效位置，所以全 0 / False
            # target_pad_mask: [B, M]

            full_pad_mask = torch.cat(
                [history_pad_mask, target_pad_mask],
                dim=1
            )
            # history_pad_mask: [B, L]
            # target_pad_mask:  [B, M]
            # 在序列维拼接
            # full_pad_mask:    [B, L + M] = [B, T]
            # 含义：形成与 h_0 对齐的完整 padding mask

        else:
            full_pad_mask = None
            # 如果历史没有 pad mask，则不提供 src_key_padding_mask

        hidden_states = self.transformer_encoder(
            h_0,
            src_key_padding_mask=full_pad_mask
        )
        # h_0: [B, T, D]
        # full_pad_mask: [B, T] 或 None
        # hidden_states: [B, T, D]
        # 含义：经过多层 Transformer Encoder 编码后的隐藏状态

        target_latent = hidden_states[:, -m_layers:, :]
        # hidden_states: [B, T, D]
        # 取最后 M 个位置（即目标 semantic token 对应位置）
        # target_latent: [B, M, D]
        # 含义：拿到目标部分的隐藏表示，后面用于预测 semantic id

        logits = self.prediction_head(target_latent)
        # target_latent: [B, M, D]
        # prediction_head(target_latent) -> logits: [B, M, sid_vocab_size]
        # 含义：对每个目标位置输出一个对 sid 词表的分类分数

        return logits, target_latent
        # logits:        [B, M, sid_vocab_size]
        # target_latent: [B, M, D]


class SeqGraphFusionRanker(nn.Module):
    """
    序列信息 + 图信息融合排序器。

    核心思路：
    1. 先用 Transformer 对历史序列编码，得到上下文表示；
    2. 再结合 semantic_context 和 graph_context 做目标感知的历史重读；
    3. 最后通过 gate 融合三类上下文，得到最终排序上下文 fused_context。
    """

    def __init__(
        self,
        item_vocab_size: int,
        embed_dim: int = 128,
        num_layers: int = 2,
        nhead: int = 4,
        max_seq_len: int = 200,
        dropout: float = 0.2
    ):
        # item_vocab_size: item 词表大小
        # embed_dim:       embedding 维度 D
        # num_layers:      TransformerEncoder 层数
        # nhead:           多头注意力头数
        # max_seq_len:     最大历史长度
        # dropout:         dropout 概率

        super(SeqGraphFusionRanker, self).__init__()
        # 初始化父类 nn.Module。

        self.embed_dim = embed_dim
        # 保存 embedding 维度 D。

        self.item_embedding = nn.Embedding(
            item_vocab_size,
            embed_dim,
            padding_idx=0
        )
        # self.item_embedding.weight: [item_vocab_size, D]
        # 输入:
        #   history_seq: [B, L]
        # 输出:
        #   item_emb: [B, L, D]

        self.position_embedding = nn.Embedding(
            max_seq_len,
            embed_dim
        )
        # self.position_embedding.weight: [max_seq_len, D]
        # 输入:
        #   positions: [L]
        # 输出:
        #   pos_emb: [L, D]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,             # 输入维度 D
            nhead=nhead,                   # 多头注意力头数
            dim_feedforward=embed_dim * 4, # 前馈层隐藏维度 4D
            dropout=dropout,               # dropout 概率
            activation='gelu',             # 激活函数
            batch_first=True               # 输入格式 [B, L, D]
        )
        # 单层 encoder 输入输出:
        # [B, L, D] -> [B, L, D]

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        # 堆叠后的 encoder 输入输出:
        # [B, L, D] -> [B, L, D]

        self.layer_norm = nn.LayerNorm(embed_dim)
        # 输入输出形状不变:
        # [B, L, D] -> [B, L, D]
        # 或 [B, D] -> [B, D]

        self.dropout = nn.Dropout(dropout)
        # Dropout 不改变形状。

        self.context_gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            # 输入:
            #   gate_input: [B, 3D]
            # 输出:
            #   [B, 2D]

            nn.GELU(),
            # 形状不变:
            # [B, 2D] -> [B, 2D]

            nn.Dropout(dropout),
            # 形状不变:
            # [B, 2D] -> [B, 2D]

            nn.Linear(embed_dim * 2, embed_dim * 2),
            # [B, 2D] -> [B, 2D]

            nn.Sigmoid()
            # [B, 2D] -> [B, 2D]
            # 输出范围在 (0, 1)，作为门控系数
        )
        # 整体输入输出:
        # gate_input: [B, 3D]
        # gate_output: [B, 2D]

        self.output_norm = nn.LayerNorm(embed_dim)
        # 输入输出形状不变:
        # [B, D] -> [B, D]

        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            # [B, D] -> [B, D]

            nn.GELU(),
            # [B, D] -> [B, D]

            nn.Dropout(dropout)
            # [B, D] -> [B, D]
        )
        # 整体输入输出:
        # [B, D] -> [B, D]

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # 构造因果 mask（上三角 mask）。
        # 作用：
        #   让位置 t 只能看到自己和自己之前的位置，不能看到未来位置。
        # 参数:
        #   seq_len: 序列长度 L
        # 返回:
        #   causal_mask: [L, L]
        #   dtype=bool
        #   True 表示该位置被 mask，不可见

        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1
        )
        # torch.ones(seq_len, seq_len): [L, L]
        # torch.triu(..., diagonal=1):
        #   保留主对角线以上的上三角为 True
        # 结果 causal_mask: [L, L]
        #
        # 例如 L=4 时:
        # [[False, True,  True,  True ],
        #  [False, False, True,  True ],
        #  [False, False, False, True ],
        #  [False, False, False, False]]

    #对历史序列用transformer编码，得到上下文表示；
    #返回最后一个有效历史item的表示 作为序列表示；
    def encode_history(
        self,
        history_seq: torch.Tensor,
        history_pad_mask: torch.Tensor | None = None
    ):
        """
        编码历史序列。

        参数:
            history_seq:
                历史 item 序列
                形状: [B, L]

            history_pad_mask:
                历史 PAD mask
                形状: [B, L]
                True 表示 PAD

        返回:
            encoded:
                编码后的每个历史位置表示
                形状: [B, L, D]

            seq_context:
                序列级上下文表示，取最后一个有效位置的 hidden state
                形状: [B, D]
        """

        batch_size, seq_len = history_seq.shape
        # history_seq: [B, L]
        # batch_size = B
        # seq_len = L

        positions = torch.arange(seq_len, device=history_seq.device)
        # positions: [L]
        # 内容为 [0, 1, ..., L-1]

        pos_emb = self.position_embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)
        # position_embedding(positions): [L, D]
        # unsqueeze(0): [1, L, D]
        # expand(batch_size, -1, -1): [B, L, D]
        # 含义：把位置向量扩展到 batch 维

        hidden = self.item_embedding(history_seq) + pos_emb
        # self.item_embedding(history_seq): [B, L, D]
        # pos_emb:                         [B, L, D]
        # hidden:                         [B, L, D]
        # 含义：item embedding + position embedding

        hidden = self.layer_norm(hidden)
        # hidden: [B, L, D] -> [B, L, D]

        hidden = self.dropout(hidden)
        # hidden: [B, L, D] -> [B, L, D]

        causal_mask = self._build_causal_mask(seq_len, history_seq.device)
        # causal_mask: [L, L]
        # True 表示未来位置不可见

        encoded = self.transformer_encoder(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=history_pad_mask
        )
        # hidden: [B, L, D]
        # causal_mask: [L, L]
        # history_pad_mask: [B, L] 或 None
        # encoded: [B, L, D]
        # 含义：对历史序列做因果编码，防止看到未来交互

        if history_pad_mask is None:
            # 如果没有 pad mask，说明默认每个位置都有效

            last_indices = torch.full(
                (batch_size,),
                seq_len - 1,
                dtype=torch.long,
                device=history_seq.device
            )
            # 构造每个样本“最后一个有效位置”的索引
            # last_indices: [B]
            # 所有值都等于 L - 1

        else:
            valid_lengths = (~history_pad_mask).sum(dim=1).clamp(min=1)
            # history_pad_mask: [B, L]
            # ~history_pad_mask: [B, L]
            #   True( PAD ) -> False
            #   False(有效) -> True
            # sum(dim=1): [B]
            #   统计每个样本的有效长度
            # clamp(min=1):
            #   防止全 PAD 时长度为 0
            # valid_lengths: [B]

            last_indices = valid_lengths - 1
            # last_indices: [B]
            # 表示每个样本最后一个有效 token 的下标

        batch_indices = torch.arange(batch_size, device=history_seq.device)
        # batch_indices: [B]
        # 内容为 [0, 1, ..., B-1]

        seq_context = encoded[batch_indices, last_indices]
        # encoded: [B, L, D]
        # batch_indices: [B]
        # last_indices:  [B]
        # 高级索引后:
        # seq_context: [B, D]
        # 含义：取每个样本最后一个有效历史位置的表示，作为序列级上下文

        return encoded, seq_context
        # encoded:     [B, L, D]
        # seq_context: [B, D]

    #将encoded_history与 图，sid,seq的融合表示 计算得到 相似度权重a；
    #encode_history经过权重加和 得到每个样本的 相似度上下文表示[B,D]；
    #与seq_context残差连接；
    #用目标相关的 query 对encoded_history表示重新加权池化。
    def _pool_target_aware_history(
        self,
        encoded_history: torch.Tensor,
        seq_context: torch.Tensor,
        semantic_context: torch.Tensor,
        graph_context: torch.Tensor,
        history_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        用目标相关的 query 对历史表示重新加权池化。

        参数:
            encoded_history:
                历史每个位置的编码表示
                形状: [B, L, D]

            seq_context:
                最后一个有效位置的序列上下文
                形状: [B, D]

            semantic_context:
                语义上下文
                形状: [B, D]

            graph_context:
                图上下文
                形状: [B, D]

            history_pad_mask:
                历史 PAD mask
                形状: [B, L]

        返回:
            pooled_history_context:
                目标感知的历史池化结果，与 seq_context 残差融合后输出
                形状: [B, D]
        """

        target_query = F.normalize(seq_context + semantic_context + graph_context, dim=-1)
        # seq_context:      [B, D]
        # semantic_context: [B, D]
        # graph_context:    [B, D]
        # 相加后:           [B, D]
        # F.normalize(..., dim=-1): [B, D]
        # 含义：构造一个“目标相关查询向量”，并做 L2 归一化

        history_keys = F.normalize(encoded_history, dim=-1)
        # encoded_history: [B, L, D]
        # normalize 后 history_keys: [B, L, D]
        # 含义：把每个历史位置表示也归一化，便于做相似度计算

        attn_scores = torch.sum(history_keys * target_query.unsqueeze(1), dim=-1) * (self.embed_dim ** 0.5)
        # target_query: [B, D]
        # target_query.unsqueeze(1): [B, 1, D]
        # history_keys: [B, L, D]
        # history_keys * target_query.unsqueeze(1): [B, L, D]
        # sum(dim=-1): [B, L]
        # 得到每个历史位置与 target_query 的点积相似度
        # 再乘以 (D ** 0.5)
        # attn_scores: [B, L]

        if history_pad_mask is not None:
            attn_scores = attn_scores.masked_fill(history_pad_mask, -1e9)
            # attn_scores: [B, L]
            # history_pad_mask: [B, L]
            # 对 PAD 位置填充一个极小值，softmax 后这些位置权重约等于 0
            # attn_scores: [B, L]

        attn_weights = torch.softmax(attn_scores, dim=-1)
        # attn_scores: [B, L]
        # softmax(dim=-1) 后 attn_weights: [B, L]
        # 每个样本在 L 个历史位置上的权重和为 1

        pooled_history = torch.sum(encoded_history * attn_weights.unsqueeze(-1), dim=1)
        # attn_weights: [B, L]
        # attn_weights.unsqueeze(-1): [B, L, 1]
        # encoded_history: [B, L, D]
        # encoded_history * attn_weights.unsqueeze(-1): [B, L, D]
        # sum(dim=1): [B, D]
        # pooled_history: [B, D]
        # 含义：对历史位置做加权（权重为target_query的点积相似度）求和，得到目标感知的历史聚合表示

        return self.output_norm(seq_context + pooled_history)
        # seq_context:    [B, D]
        # pooled_history: [B, D]
        # 相加后:         [B, D]
        # output_norm 后: [B, D]
        # 含义：把最后状态和加权历史池化结果做残差融合

    #用门融合 pool_target_aware_history+sid表示+图表示
    #返回最终融合表示
    def forward(
        self,
        history_seq: torch.Tensor,
        semantic_context: torch.Tensor,
        graph_context: torch.Tensor,
        history_pad_mask: torch.Tensor | None = None
    ):
        """
        参数:
            history_seq:
                历史 item 序列
                形状: [B, L]

            semantic_context:
                语义上下文表示
                形状: [B, D]

            graph_context:
                图上下文表示
                形状: [B, D]

            history_pad_mask:
                历史 PAD 掩码
                形状: [B, L]

        返回:
            fused_context:
                最终融合后的上下文表示
                形状: [B, D]

            seq_context:
                原始序列上下文（最后一个有效位置）
                形状: [B, D]
        """

        encoded_history, seq_context = self.encode_history(history_seq, history_pad_mask)
        # history_seq: [B, L]
        # history_pad_mask: [B, L] 或 None
        # encoded_history: [B, L, D]
        # seq_context: [B, D]

        pooled_seq_context = self._pool_target_aware_history(
            encoded_history,
            seq_context,
            semantic_context,
            graph_context,
            history_pad_mask,
        )
        # encoded_history:  [B, L, D]
        # seq_context:      [B, D]
        # semantic_context: [B, D]
        # graph_context:    [B, D]
        # history_pad_mask: [B, L] 或 None
        # pooled_seq_context: [B, D]
        # 含义：得到“目标感知重读后的序列上下文”

        gate_input = torch.cat([pooled_seq_context, semantic_context, graph_context], dim=-1)
        # pooled_seq_context: [B, D]
        # semantic_context:   [B, D]
        # graph_context:      [B, D]
        # 在最后一维拼接
        # gate_input:         [B, 3D]

        semantic_gate, graph_gate = self.context_gate(gate_input).chunk(2, dim=-1)
        # context_gate(gate_input): [B, 2D]
        # .chunk(2, dim=-1) 沿最后一维均分成两块
        # semantic_gate: [B, D]
        # graph_gate:    [B, D]
        # 含义：
        #   semantic_gate 控制 semantic_context 注入多少
        #   graph_gate    控制 graph_context 注入多少

        fused_context = pooled_seq_context + semantic_gate * semantic_context + graph_gate * graph_context
        # pooled_seq_context:         [B, D]
        # semantic_gate * semantic_context: [B, D]
        # graph_gate * graph_context:       [B, D]
        # 三者相加后 fused_context:  [B, D]
        # 含义：门控融合三类上下文

        fused_context = self.output_norm(fused_context + self.output_proj(fused_context))
        # fused_context: [B, D]
        # output_proj(fused_context): [B, D]
        # 残差相加后: [B, D]
        # output_norm 后: [B, D]
        # 含义：再经过一个小型前馈投影 + 残差 + 归一化，提升表达能力

        return fused_context, seq_context
        # fused_context: [B, D]
        # seq_context:   [B, D]