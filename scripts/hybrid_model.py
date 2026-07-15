"""Hybrid dense+sparse embedding model for Stage 2.

Three literature-backed upgrades over plain mean-pooled contrastive training, plus a sparse head:

  * LATENT-ATTENTION POOLING (NV-Embed, 2024). A small set of learned latent queries cross-attends over
    the token hidden states, replacing mean pooling. Consistently beats mean/last-token pooling on MTEB.
  * SPLADE SPARSE HEAD (Formal 2021 / BGE-M3). log(1+relu(MLM-logits)) max-pooled over tokens gives a
    vocab-dimensional sparse vector that does exact lexical matching. Legal/financial retrieval leans on
    exact terms -- citations, defined terms, tickers -- exactly where dense embeddings are weakest and
    BM25 already competes, so a learned-sparse head is unusually apt for this domain. It reuses the
    Stage-1 MLM head, so the 156 custom citation tokens feed straight into it.
  * MATRYOSHKA dense output (nested 768..64), unchanged from Stage 2.

The model returns a dense vector (for MRL contrastive + reranker distillation) and a sparse vector (for
lexical contrastive), trained jointly. At inference either or both can be used (hybrid = a weighted sum
of dense-cosine and sparse-dot scores).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ModernBertForMaskedLM, ModernBertModel


class LatentAttentionPooling(nn.Module):
    """NV-Embed-style latent-attention pooling.

    n_latents learned query vectors cross-attend (multi-head) over the token states; the attended
    latents are passed through an MLP and mean-reduced to a single dense vector. Padding is masked.
    """

    def __init__(self, dim, n_latents=8, n_heads=8, mlp_ratio=2):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, dim) * dim**-0.5)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
                                 nn.Linear(dim * mlp_ratio, dim))

    def forward(self, hidden, attention_mask):
        B = hidden.size(0)
        q = self.latents.unsqueeze(0).expand(B, -1, -1)          # [B, L, D]
        key_pad = attention_mask == 0                             # True where padded -> ignored
        attended, _ = self.attn(q, hidden, hidden, key_padding_mask=key_pad)  # [B, L, D]
        attended = attended + self.mlp(self.norm(attended))
        return attended.mean(dim=1)                               # [B, D]


class SpladeHead(nn.Module):
    """SPLADE sparse representation from the (frozen or fine-tuned) MLM head.

    sparse[v] = max over tokens of log(1 + relu(logit_{t,v})), masked to real tokens. Produces a
    vocab-dimensional, mostly-zero vector whose nonzeros are the terms the model thinks the text is
    "about" -- learned lexical matching that subsumes BM25.
    """

    def __init__(self, mlm_head):
        super().__init__()
        self.mlm_head = mlm_head    # reuse Stage-1 decoder: [D] -> [vocab]

    def forward(self, hidden, attention_mask):
        logits = self.mlm_head(hidden)                           # [B, T, V]
        activated = torch.log1p(F.relu(logits))
        activated = activated.masked_fill(attention_mask.unsqueeze(-1) == 0, 0.0)
        return activated.max(dim=1).values                       # [B, V]


class HybridEmbedder(nn.Module):
    def __init__(self, stage1_ckpt, n_latents=8, sparse=True, matryoshka_dims=(768, 512, 256, 128, 64)):
        super().__init__()
        # encoder body (no MLM head) for the dense path
        self.encoder = ModernBertModel.from_pretrained(stage1_ckpt)
        dim = self.encoder.config.hidden_size
        self.pool = LatentAttentionPooling(dim, n_latents=n_latents)
        self.matryoshka_dims = matryoshka_dims
        self.sparse = sparse
        if sparse:
            # pull the MLM head off a ForMaskedLM view of the same checkpoint so the sparse head starts
            # from the trained decoder (and its 156 custom-token rows)
            mlm = ModernBertForMaskedLM.from_pretrained(stage1_ckpt)
            self.mlm_head = mlm.decoder if hasattr(mlm, "decoder") else mlm.get_output_embeddings()
            self.splade = SpladeHead(self.mlm_head)

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        dense = self.pool(h, attention_mask)                     # [B, D]
        out = {"dense": dense}
        if self.sparse:
            out["sparse"] = self.splade(h, attention_mask)       # [B, V]
        return out

    @staticmethod
    def load_tokenizer_fix(tok):
        # our tokenizer advertises token_type_ids, which ModernBert rejects
        tok.model_input_names = ["input_ids", "attention_mask"]
        return tok
