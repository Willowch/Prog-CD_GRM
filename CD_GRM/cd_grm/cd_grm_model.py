# 文件路径: CD_GRM-Project/cd_grm_model.py
import torch.nn as nn

from .model.semantic_id import ResidualQuantizer
from .model.mask_scheduler import CosineMaskScheduler
from .model.transformer import DenoisingTransformer, SeqGraphFusionRanker
from .model.graph_view import GraphViewBuilder
from .loss.contrastive import InfoNCELoss
from .model.cd_grm import CD_GRM_Loss_Engine
from .inference import ParallelDenoisingEngine


class CD_GRM_Model(nn.Module):
    # 训练损失 和 推理器 组合模型
    def __init__(self, config_dict, dataset):
        super(CD_GRM_Model, self).__init__()

        self.cfg = config_dict['model_config']
        self.item_vocab_size = dataset.item_num
        self.device = config_dict['device']

        mask_token_id = self.cfg['codebook_size'] + 1
        sid_vocab_size = self.cfg['codebook_size'] + 2

        quantizer = ResidualQuantizer(
            num_layers=self.cfg['m_layers'],
            codebook_size=self.cfg['codebook_size'],
            embed_dim=self.cfg['embed_dim'],
            commitment_cost=self.cfg['commitment_cost'],
            decay=self.cfg['decay'],
            eps=self.cfg['eps'],
            restart_thres=self.cfg['restart_thres'],
            # [FIX-DEAD-CODE] Pass dead-code restart controls into the quantizer.
            dead_code_warmup_steps=self.cfg.get('dead_code_warmup_steps', 100),
            dead_code_patience=self.cfg.get('dead_code_patience', 100),
            dead_code_min_usage_ratio=self.cfg.get('dead_code_min_usage_ratio', 0.1)
        )

        scheduler = CosineMaskScheduler(
            num_steps=self.cfg['diffusion_steps'],
            mask_token_id=mask_token_id
        )

        transformer = DenoisingTransformer(
            item_vocab_size=self.item_vocab_size,
            sid_vocab_size=sid_vocab_size,
            embed_dim=self.cfg['embed_dim'],
            num_layers=self.cfg['transformer_layers'],
            nhead=self.cfg['nhead'],
            dropout=self.cfg['dropout'],
        )

        seq_ranker = None
        if self.cfg.get('use_seq_graph_fusion', False):
            seq_ranker = SeqGraphFusionRanker(
                item_vocab_size=self.item_vocab_size,
                embed_dim=self.cfg['embed_dim'],
                num_layers=self.cfg.get('ranker_layers', self.cfg['transformer_layers']),
                nhead=self.cfg['nhead'],
                max_seq_len=config_dict.get('MAX_ITEM_LIST_LENGTH', 50),
                dropout=self.cfg.get('ranker_dropout', self.cfg['dropout']),
            )

        graph_builder = GraphViewBuilder(
            item_vocab_size=self.item_vocab_size,
            dataset=dataset,
            max_degree=self.cfg['max_degree'],
            window_size=self.cfg.get('graph_window_size', 3),
            walk_temp=self.cfg.get('graph_walk_temp', 0.8),
        )

        cl_loss_fn = InfoNCELoss(
            temperature=self.cfg['tau']
        )

        self.engine = CD_GRM_Loss_Engine(
            item_vocab_size=self.item_vocab_size,
            embed_dim=self.cfg['embed_dim'],
            m_layers=self.cfg['m_layers'],
            # lambda_cl=self.cfg['lambda_cl'],
            codebook_size=self.cfg['codebook_size'],
            quantizer=quantizer,
            scheduler=scheduler,
            transformer=transformer,
            seq_ranker=seq_ranker,
            graph_builder=graph_builder,
            cl_loss_fn=cl_loss_fn,
            item_loss_weight=self.cfg.get('item_loss_weight', 0.0),
            hybrid_semantic_weight=self.cfg.get('hybrid_semantic_weight', 0.0),
            stage2_aux_loss_weight=self.cfg.get('stage2_aux_loss_weight', 1.0),
            finetune_item_embedding_stage2=self.cfg.get('finetune_item_embedding_stage2', False),
            item_embedding_stage2_lr_scale=self.cfg.get('item_embedding_stage2_lr_scale', 0.5),
            graph_context_decay=self.cfg.get('graph_context_decay', 0.8),
            graph_prior_weight=self.cfg.get('graph_prior_weight', 0.0),
            use_target_aware_graph_prior=self.cfg.get('use_target_aware_graph_prior', False),
            graph_prior_sim_scale=self.cfg.get('graph_prior_sim_scale', 1.0),
            graph_prior_recency_strength=self.cfg.get('graph_prior_recency_strength', 1.0),
            graph_prior_second_hop_weight=self.cfg.get('graph_prior_second_hop_weight', 0.0),
            history_repeat_prior_weight=self.cfg.get('history_repeat_prior_weight', 0.0),
            last_item_prior_weight=self.cfg.get('last_item_prior_weight', 0.0),
        )

        self.infer_engine = ParallelDenoisingEngine(
            model=self.engine,
            total_steps=self.cfg['diffusion_steps'],
            m_layers=self.cfg['m_layers']
        )

    def calculate_loss(self, history_seq, target_item, stage=2):
        return self.engine.calculate_loss(history_seq, target_item, stage=stage)

    def full_sort_predict(self, interaction):
        return self.infer_engine.predict_scores(interaction["item_id_list"])
