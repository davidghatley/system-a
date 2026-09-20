#!/usr/bin/env python3
"""Tiny check of Laya's published proper-reward formula and gradient path."""

import torch


def proper_reward(q, target, qtype, mask, w_sph=0.5, w_rps=1.0, log_floor=-9.21):
  q = q * mask
  log_score = (target * torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)).sum(-1)
  spherical = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
  reward = log_score + w_sph * spherical
  is_score = (qtype == 1).float()
  k = mask.sum(-1).clamp(min=2).float()
  rps = (((torch.cumsum(q, -1) - torch.cumsum(target, -1)) ** 2) * mask).sum(-1) / (k - 1)
  return reward - w_rps * rps * is_score


logits = torch.tensor([[1.2, -0.2, 0.4], [0.1, 0.2, 0.3]], requires_grad=True)
mask = torch.ones_like(logits, dtype=torch.bool)
target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
qtype = torch.tensor([0, 1])
probs = torch.softmax(logits, dim=-1)
reward = proper_reward(probs, target, qtype, mask)
loss = -reward.mean()
loss.backward()

assert torch.isfinite(reward).all()
assert logits.grad is not None and torch.isfinite(logits.grad).all()

ordinal_target = torch.tensor([[0.0, 1.0, 0.0]])
ordinal_type = torch.tensor([1])
ordinal_mask = torch.ones((1, 3), dtype=torch.bool)
near = proper_reward(torch.tensor([[0.1, 0.8, 0.1]]), ordinal_target, ordinal_type, ordinal_mask)
far = proper_reward(torch.tensor([[0.8, 0.1, 0.1]]), ordinal_target, ordinal_type, ordinal_mask)
assert near.item() > far.item()

print("PASS: rewards and gradients are finite; ordinal near-target reward exceeds far-target reward")
