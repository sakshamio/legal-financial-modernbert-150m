"""Losses for the hybrid Stage-2 trainer. Each is a pure tensor function, unit-tested in isolation.

  matryoshka_gist_infonce -- dense contrastive at every nested dim, with GISTEmbed false-negative
                             masking driven by a guide model's similarities.
  splade_infonce          -- sparse contrastive + FLOPS sparsity regularization.
  margin_mse              -- distil a cross-encoder reranker's relevance margins into the bi-encoder.
"""
import torch
import torch.nn.functional as F


def _info_nce(scores, labels, tau):
    return F.cross_entropy(scores / tau, labels)


def matryoshka_gist_infonce(q_dense, p_dense, tau=0.02, dims=(768, 512, 256, 128, 64),
                            weights=None, guide_q=None, guide_p=None, gist_margin=0.0):
    """In-batch InfoNCE at each Matryoshka prefix dim, summed.

    Batch layout: row i's positive is column i; all other columns are negatives.

    GISTEmbed false-negative masking: if a GUIDE model thinks candidate j is at least as relevant to
    query i as i's own positive (guide_sim(i,j) > guide_sim(i,i) - gist_margin), that candidate is
    probably a true positive in disguise, so we mask it out of the negatives (set its score to -inf).
    This matters a lot for synthetic data, where many passages in a batch are plausibly relevant.
    """
    n = q_dense.size(0)
    labels = torch.arange(n, device=q_dense.device)
    dims = [d for d in dims if d <= q_dense.size(1)]
    weights = weights or [1.0] * len(dims)

    mask = None
    if guide_q is not None and guide_p is not None:
        gq = F.normalize(guide_q.float(), dim=-1)
        gp = F.normalize(guide_p.float(), dim=-1)
        gsim = gq @ gp.T                                          # [n, n] guide similarities
        pos = gsim.diag().unsqueeze(1)                            # guide sim to own positive
        # off-diagonal candidates more relevant than the positive -> false negatives -> mask
        mask = (gsim > pos - gist_margin)
        mask.fill_diagonal_(False)                               # never mask the true positive

    total = 0.0
    for d, w in zip(dims, weights):
        qs = F.normalize(q_dense[:, :d], dim=-1)
        ps = F.normalize(p_dense[:, :d], dim=-1)
        scores = qs @ ps.T                                       # [n, n]
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        total = total + w * _info_nce(scores, labels, tau)
    return total / sum(weights)


def splade_infonce(q_sparse, p_sparse, tau=0.02):
    """Sparse contrastive InfoNCE + FLOPS sparsity, returned SEPARATELY.

    Returning ce and flops separately lets the trainer weight them independently -- essential, because
    at init the FLOPS term is huge (the untrained SPLADE head activates most of the vocab) and, if
    folded into one number, drowns both the sparse contrastive signal and the dense loss. FLOPS also
    needs its own (warmed-up) coefficient, per standard SPLADE practice.

    SPLADE scores are dot products (not cosine): magnitude carries term-importance information. To keep
    the contrastive CE in a sane range regardless of activation scale, scores are divided by sqrt(dim).
    """
    n = q_sparse.size(0)
    labels = torch.arange(n, device=q_sparse.device)
    scale = q_sparse.size(1) ** 0.5
    scores = (q_sparse @ p_sparse.T) / scale
    ce = _info_nce(scores, labels, tau)
    flops = (q_sparse.mean(0) ** 2).sum() + (p_sparse.mean(0) ** 2).sum()
    return ce, flops


def margin_mse(q_dense, p_dense, neg_dense, teacher_margin):
    """Distil a reranker: match the model's (pos - neg) score gap to the teacher's.

    q/p/neg_dense: [n, d].  teacher_margin: [n] = reranker(q,pos) - reranker(q,neg).
    The model's margin uses dot products (order-preserving with cosine after normalization).
    """
    qn = F.normalize(q_dense, dim=-1)
    s_pos = (qn * F.normalize(p_dense, dim=-1)).sum(-1)
    s_neg = (qn * F.normalize(neg_dense, dim=-1)).sum(-1)
    return F.mse_loss(s_pos - s_neg, teacher_margin)
