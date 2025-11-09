import torch
from torch import nn, Tensor
from .loss_utils import cos_sim, mismatched_sizes_all_gather
import torch.nn.functional as F


class HardNegativeNLLLoss:
    def __init__(
        self,
        scale: float = 20.0,
        similarity_fct=cos_sim,
    ):
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def __call__(
        self,
        q_reps: Tensor,
        d_reps_pos: Tensor,
        d_reps_neg: Tensor = None,
    ):
        if d_reps_neg is None:
            d_reps_neg = d_reps_pos[:0, :]

        if torch.distributed.is_initialized():
            full_d_reps_pos = mismatched_sizes_all_gather(d_reps_pos)
            full_d_reps_pos = torch.cat(full_d_reps_pos)

            full_q_reps = mismatched_sizes_all_gather(q_reps)
            full_q_reps = torch.cat(full_q_reps)

            full_d_reps_neg = mismatched_sizes_all_gather(d_reps_neg)
            full_d_reps_neg = torch.cat(full_d_reps_neg)
        else:
            full_d_reps_pos = d_reps_pos
            full_q_reps = q_reps
            full_d_reps_neg = d_reps_neg

        d_reps = torch.cat([full_d_reps_pos, full_d_reps_neg], dim=0)
        scores = self.similarity_fct(full_q_reps, d_reps) * self.scale
        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )

        loss = self.cross_entropy_loss(scores, labels)
        return loss


class MixConstrastiveLoss:
    def __init__(
        self,
        scale: float = 20.0,
        similarity_fct=cos_sim,
    ):
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def __call__(
        self,
        q_reps: Tensor,
        d_reps_pos: Tensor,
        d_reps_neg: Tensor = None,
    ):
        if d_reps_neg is None:
            d_reps_neg = d_reps_pos[:0, :]

        if torch.distributed.is_initialized():
            full_d_reps_pos = mismatched_sizes_all_gather(d_reps_pos)
            full_d_reps_pos = torch.cat(full_d_reps_pos)

            full_q_reps = mismatched_sizes_all_gather(q_reps)
            full_q_reps = torch.cat(full_q_reps)

            full_d_reps_neg = mismatched_sizes_all_gather(d_reps_neg)
            full_d_reps_neg = torch.cat(full_d_reps_neg)
        else:
            full_d_reps_pos = d_reps_pos
            full_q_reps = q_reps
            full_d_reps_neg = d_reps_neg

        # add new loss function of MixCL
        d_reps = torch.cat([full_d_reps_pos, full_d_reps_neg], dim=0)
        scores = self.similarity_fct(full_q_reps, d_reps) * self.scale
        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )

        pos_loss = F.cross_entropy(scores, labels, reduction='none')
        neg_loss = F.nll_loss(torch.log(torch.clamp((1.0 - F.softmax(logits)), min=1e-5)), labels, reduction='none')

        loss = self.cross_entropy_loss(scores, labels)
        return loss

    