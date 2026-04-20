from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParallelDenoisingEngine(nn.Module):
    def __init__(self, model: nn.Module, total_steps: int = 10, m_layers: int = 4):
        super(ParallelDenoisingEngine, self).__init__()
        self.model = model
        self.total_steps = total_steps
        self.m_layers = m_layers

    @torch.no_grad()
    def _run_parallel_denoising(self, history_seq: torch.Tensor):
        batch_size = history_seq.shape[0]
        device = history_seq.device
        history_pad_mask = history_seq == self.model.PAD_TOKEN

        x_t = torch.full(
            (batch_size, self.m_layers),
            self.model.MASK_TOKEN,
            dtype=torch.long,
            device=device,
        )
        final_target_latent = None

        for t in reversed(range(1, self.total_steps + 1)):
            logits, target_latent = self.model.transformer(
                history_seq,
                x_t,
                history_pad_mask,
            )
            logits[:, :, self.model.PAD_TOKEN] = -float("inf")
            logits[:, :, self.model.MASK_TOKEN] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            if t == 1:
                final_target_latent = target_latent
                break

            max_probs, pred_ids = torch.max(probs, dim=-1)
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
            )

            x_t = pred_ids.clone()
            x_t.scatter_(
                dim=-1,
                index=mask_indices,
                value=self.model.MASK_TOKEN,
            )

        return final_target_latent

    @torch.no_grad()
    def predict_scores(self, history_seq: torch.Tensor):
        if float(getattr(self.model, "hybrid_semantic_weight", 0.0)) > 0:
            raise RuntimeError(
                "hybrid_semantic_weight > 0 requires the removed semantic-id scoring fallback. "
                "Set hybrid_semantic_weight=0 for item-ranker-only inference."
            )
        final_target_latent = self._run_parallel_denoising(history_seq)
        return self.model.predict_item_logits(history_seq, final_target_latent)
