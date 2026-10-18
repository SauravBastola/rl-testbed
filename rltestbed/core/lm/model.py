"""A tiny character-level GPT and the sampling / log-probability utilities that
RL fine-tuning needs.

Small enough (~100k parameters) to pretrain and fine-tune on a laptop CPU in
seconds, yet structurally the same as a large language model: token and
position embeddings, causal self-attention blocks, a next-token head. Every
RL-for-LLM algorithm in this package only touches it through ``sample`` and
``completion_logprobs``, exactly as large-scale implementations do.

Prompts within a task have a fixed length, so batches need no padding; the
completion is masked after the first end-of-sequence token.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

PAD, BOS, EOS = 0, 1, 2
CHARS = "0123456789+-=>abcdefghijklmnopqrstuvwxyz "
VOCAB = ["<pad>", "<bos>", "<eos>"] + list(CHARS)
STOI = {c: i for i, c in enumerate(VOCAB)}


def encode(text: str) -> List[int]:
    return [STOI[c] for c in text]


def decode(tokens) -> str:
    out = []
    for t in tokens:
        t = int(t)
        if t == EOS:
            break
        if t > EOS:
            out.append(VOCAB[t])
    return "".join(out)


class TinyGPT(nn.Module):
    def __init__(self, vocab: int = len(VOCAB), d_model: int = 64, layers: int = 2,
                 heads: int = 4, context: int = 32):
        super().__init__()
        self.context = context
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(context, d_model)
        block = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout=0.0,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        h = self.tok(x) + self.pos(torch.arange(t, device=x.device))
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=x.device)
        h = self.blocks(h, mask=mask, is_causal=True)
        return self.head(self.ln(h))


@torch.no_grad()
def sample(model: TinyGPT, prompts: torch.Tensor, max_new: int, temperature: float = 1.0,
           greedy: bool = False) -> torch.Tensor:
    """Autoregressively extend ``prompts`` (B, P) by up to ``max_new`` tokens."""
    x = prompts
    for _ in range(max_new):
        logits = model(x)[:, -1, :]
        logits[:, PAD] = -1e9  # never emit padding
        if greedy:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            nxt = torch.multinomial(torch.softmax(logits / max(temperature, 1e-4), -1), 1)
        x = torch.cat([x, nxt], dim=1)
    return x


def completion_mask(seqs: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """1 for completion tokens up to and including the first EOS, else 0."""
    comp = seqs[:, prompt_len:]
    is_eos = (comp == EOS).int()
    after = torch.cumsum(is_eos, dim=1) - is_eos  # EOS tokens seen before this position
    return (after == 0).float()


def completion_logprobs(model: TinyGPT, seqs: torch.Tensor, prompt_len: int
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token log-probabilities and entropies of the completion part of ``seqs``."""
    logits = model(seqs[:, :-1])[:, prompt_len - 1:, :]
    logp_all = torch.log_softmax(logits, dim=-1)
    targets = seqs[:, prompt_len:]
    logp = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    entropy = -(logp_all.exp() * logp_all).sum(-1)
    return logp, entropy


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None) -> torch.Tensor:
    if dim is None:
        return (x * mask).sum() / mask.sum().clamp_min(1.0)
    return (x * mask).sum(dim) / mask.sum(dim).clamp_min(1.0)
