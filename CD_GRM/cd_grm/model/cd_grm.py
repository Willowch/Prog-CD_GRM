
import torch
import torch.nn as nn
import torch.nn.functional as F


class CD_GRM_Loss_Engine(nn.Module):
    def __init__(self, item_vocab_size: int, embed_dim: int = 128, m_layers: int = 4,
                codebook_size: int = 256,
                 quantizer: nn.Module = None, scheduler: nn.Module = None,
                 transformer: nn.Module = None, graph_builder=None,
                 cl_loss_fn: nn.Module = None):
        super(CD_GRM_Loss_Engine, self).__init__()
        self.item_vocab_size = item_vocab_size
        self.embed_dim = embed_dim
        #self.lambda_cl = lambda_cl
        self.m_layers = m_layers
        self.log_vars = nn.Parameter(torch.zeros(3))
        self.item_embedding = nn.Embedding(
            item_vocab_size,    # 所有Item数量
            embed_dim,          # embedding 维度
            padding_idx=0       # 0 作为 PAD token，不参与训练
        )
        # 【必须添加这一行】强制限制物品向量的初始化范围，防止“黑洞效应”吞噬所有死码
        nn.init.uniform_(self.item_embedding.weight, -0.05, 0.05)
        # MASK token
        self.MASK_TOKEN = codebook_size + 1
        # PAD token
        self.PAD_TOKEN = 0
        # Semantic ID 词表大小包含：codebook_size + PAD + MASK
        self.SID_VOCAB_SIZE = codebook_size + 2

        self.quantizer = quantizer
        self.scheduler = scheduler
        self.transformer = transformer
        self.graph_builder = graph_builder
        self.cl_loss_fn = cl_loss_fn
        self._tie_embeddings()
    def _tie_embeddings(self):
        if self.transformer is not None: # 如果 transformer已经初始化
            self.transformer.item_embedding = self.item_embedding

    def forward_train(self, history_seq: torch.Tensor, target_item: torch.Tensor):
        #调用各层，返回所需数据
        batch_size, seq_len = history_seq.shape
        device = history_seq.device

        e_target = self.item_embedding(target_item)
        _, true_sids, commit_loss = self.quantizer(e_target)
        true_sids = true_sids + 1#返回的是codebook下标，+1略过padtoken
        t = self.scheduler.get_random_t(
            batch_size, # batch大小
            device
        )
        masked_sids, mask_bool = self.scheduler.add_noise(
            true_sids,
            t
        )
        history_pad_mask = (history_seq == self.PAD_TOKEN)
        logits, target_latent = self.transformer(
            history_seq,
            masked_sids,
            history_pad_mask
        )
        walk_length = self.m_layers   # 随机游走长度 = SID 层数
        view_b_seq = self.graph_builder.sample_view_B(
            target_item,
            walk_length
        )
        # 返回所有训练需要的数据
        return logits, true_sids,mask_bool, target_latent, view_b_seq, commit_loss
        #logits:[bs,m_layer,sid_vocab_size];true_sids:[bs,m_layer];target_latent:[bs,m_layer,emb_l]
        #view_b_seq:[bs,m_layer]

    def calculate_loss(self, history_seq: torch.Tensor, target_item: torch.Tensor, stage: int = 2) -> tuple:
        if stage == 1:
            #计算量化和图的对比损失 + 量化损失
            e_target = self.item_embedding(target_item)
            quantized_out, true_sids, commit_loss = self.quantizer(e_target)

            walk_length = self.m_layers
            view_b_seq = self.graph_builder.sample_view_B(target_item, walk_length)#[bs,m_layer]
            e_view_b = self.item_embedding(view_b_seq)#[bs,m_layer,emb_l]
            z_b = e_view_b.mean(dim=1)#[bs,emb_l]

            # 对比损失：强制量化表征与图游走的邻居表征对齐
            loss_cl = self.cl_loss_fn(quantized_out, z_b)
            loss_total = loss_cl + commit_loss.mean()
            loss_ce = torch.tensor(0.0, device=loss_total.device)#预测损失为0

            return loss_total, loss_ce, loss_cl

        else:
            #计算 预测损失 预测和图的对比损失 量化损失 + 动态权重调整
            logits, true_sids,mask_bool, target_latent, view_b_seq, commit_loss = \
                self.forward_train(history_seq, target_item)

            flat_logits = logits.reshape(-1, self.SID_VOCAB_SIZE)#[bs*m_layer,sid_vocab_size]
            flat_targets = true_sids.reshape(-1)#[bs*m_layer]
            flat_mask = mask_bool.reshape(-1)#[bs*m_layer]
            if not flat_mask.any():
                raise RuntimeError("No masked positions found in stage2 CE computation.")

            loss_ce = F.cross_entropy(
                flat_logits[flat_mask],
                flat_targets[flat_mask]
            )#交叉熵对比

            z_a = target_latent.mean(dim=1)#[bs,emb_l]
            e_view_b = self.item_embedding(view_b_seq)#[bs,m_layer,emb_l]
            z_b = e_view_b.mean(dim=1)#[bs,emb_l]
            loss_cl = self.cl_loss_fn(z_a, z_b)
            commit_loss_mean = commit_loss.mean()#对比损失

            # Loss_i = exp(-si) * li + si
            precision_ce = torch.exp(-self.log_vars[0])
            loss_ce_weighted = precision_ce * loss_ce + self.log_vars[0]

            precision_cl = torch.exp(-self.log_vars[1])
            loss_cl_weighted = precision_cl * loss_cl + self.log_vars[1]

            precision_commit = torch.exp(-self.log_vars[2])
            loss_commit_weighted = precision_commit * commit_loss_mean + self.log_vars[2]

            loss_total = loss_ce_weighted + loss_cl_weighted + loss_commit_weighted
            return loss_total, loss_ce, loss_cl
