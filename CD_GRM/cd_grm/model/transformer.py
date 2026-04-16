
import torch
import torch.nn as nn


class DenoisingTransformer(nn.Module):
    """

    """
    def __init__(self, item_vocab_size: int, sid_vocab_size: int,
                 embed_dim: int = 128, num_layers: int = 2,
                 nhead: int = 4, max_seq_len: int = 200, dropout: float = 0.1):
        # max_seq_len：样本最大输入token数量
        #初始化embedding，transformer_encoder,归一化，丢弃，预测层；
        super(DenoisingTransformer, self).__init__()
        self.embed_dim = embed_dim  # 保存 embedding 维度 d

        self.item_embedding = nn.Embedding(
            item_vocab_size,      # item 词表大小
            embed_dim,            # embedding 维度
            padding_idx=0         # 0 表示 padding
        )# shape: [item_vocab_size, embed_dim]


        self.sid_embedding = nn.Embedding(
            sid_vocab_size,       # sid大小：256^4
            embed_dim,
            padding_idx=0
        )# shape: [sid_vocab_size, embed_dim]

        self.position_embedding = nn.Embedding(
            max_seq_len + 50,     # 预留空间
            embed_dim
        )# shape: [max_seq_len+50, embed_dim]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,        # 输入特征维度
            nhead=nhead,              # 多头注意力机制头数
            dim_feedforward=embed_dim * 4,  # FFN 隐藏层大小
            dropout=dropout,          # dropout 概率
            activation='gelu',        # 激活函数
            batch_first=True          # 输入格式: [batch, seq, dim]
        )
        # 堆叠多层 encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        # 输入:  [batch, m_layers, embed_dim]
        # 输出:  [batch, m_layers, sid_vocab_size]
        self.prediction_head = nn.Linear(
            embed_dim,
            sid_vocab_size
        )
        # LayerNorm
        self.layer_norm = nn.LayerNorm(embed_dim)
        # Dropout
        self.dropout = nn.Dropout(dropout)


    def forward(self,
                history_seq: torch.Tensor,
                masked_sid: torch.Tensor,
                history_pad_mask: torch.Tensor = None):
        """
        Args:
        history_seq shape: [batch_size, seq_len]
        masked_sid shape: [batch_size, m_layers]
        history_pad_mask shape: [batch_size, seq_len]
        """
        batch_size, seq_len = history_seq.shape
        _, m_layers = masked_sid.shape
        total_len = seq_len + m_layers

        e_hist = self.item_embedding(history_seq)
        # history_seq shape:[bs, seq_l] -> e_hist shape:[bs, seq_l, emb_l]
        e_target = self.sid_embedding(masked_sid)
        # masked_sid shape:[bs, m_layer] -> e_target shape:[bs, m_layer,emb_l]
        h_0 = torch.cat([e_hist, e_target], dim=1)
        # h_0 shape:[bs, seq_l+m_layer, emb_l]
        positions = torch.arange(
            total_len,
            device=history_seq.device
        )# positions: [seq_l+m_layer]
        pos_emb = self.position_embedding(positions)
        # pos_emb shape:[seq_l+m_layer, emb_l]
        pos_emb_expanded = pos_emb.unsqueeze(0).expand(
            batch_size,
            total_len,
            self.embed_dim
        )#每个样本扩展 pos_emb_expanded:[bs,seq_l+m_layer, emb_l]

        # 加入位置编码
        h_0 = h_0 + pos_emb_expanded
        # h_0 shape:[bs,seq_l+m_layer, emb_l]
        # 层归一
        h_0 = self.layer_norm(h_0)
        # shape:[bs,seq_l+m_layer, emb_l]
        # dropout
        h_0 = self.dropout(h_0)

        if history_pad_mask is not None:
            target_pad_mask = torch.zeros(
                (batch_size, m_layers),
                dtype=torch.bool,
                device=history_seq.device
            )# target_pad_mask:[bs, m_layer]
            full_pad_mask = torch.cat(
                [history_pad_mask, target_pad_mask],
                dim=1
            )# full_pad_mask shape:[B,seq_l+m_layer]
        else:
            full_pad_mask = None

        hidden_states = self.transformer_encoder(
            h_0,
            src_key_padding_mask=full_pad_mask
        )# hidden_states shape:[bs,seq_l+m_layer, emb_l]

        target_latent = hidden_states[:, -m_layers:, :]# target_latent:[bs,m_layer,emb_l]
        logits = self.prediction_head(target_latent)# logits:[bs,m_layer, sid_vocab_size]
        # 返回预测结果 + latent 表示
        return logits, target_latent