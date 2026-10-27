"""Checkpoint-compatible research architecture; no training or network download here."""
import torch
from torch import nn
from torch.nn import functional as F


class MaskedFusionNet(nn.Module):
    def __init__(self, hidden=96, dropout=0.25):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU())
            for d in (768, 74, 35)
        ])
        self.gate = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(3)])
        self.availability = nn.Linear(3, hidden, bias=False)
        self.position = nn.Embedding(50, hidden)
        self.conv1 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.layer_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.Linear(hidden, 1)
        self.shared = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden, 3)
        self.regressor = nn.Linear(hidden, 1)

    def forward(self, text, audio, vision, available):
        h = [layer(x) for layer, x in zip(self.proj, (text, audio, vision))]
        scores = torch.cat([layer(x) for layer, x in zip(self.gate, h)], dim=-1)
        scores = scores.masked_fill(~available, -1e4)
        weight = scores.softmax(dim=-1) * available.float()
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        fused = sum(weight[..., j:j+1] * h[j] for j in range(3))
        active = available.any(dim=-1)
        pos = self.position(torch.arange(text.shape[1], device=text.device))[None]
        z = (fused + self.availability(available.float()) + pos) * active[..., None]
        z = z + self.dropout(F.gelu(self.conv1(z.transpose(1, 2)).transpose(1, 2)))
        z = self.layer_norm(z + self.dropout(F.gelu(self.conv2(z.transpose(1, 2)).transpose(1, 2))))
        z = z * active[..., None]
        weights = self.attention(z).squeeze(-1).masked_fill(~active, -1e4).softmax(dim=-1)
        attended = (weights[..., None] * z).sum(dim=1)
        mean = z.sum(dim=1) / active.sum(dim=1, keepdim=True).clamp_min(1)
        shared = self.shared(torch.cat([attended, mean], dim=-1))
        return self.classifier(shared), self.regressor(shared).squeeze(-1)
