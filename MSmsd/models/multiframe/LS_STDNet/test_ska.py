# test_ska_group_agg.py
import torch
import torch.nn.functional as F
from deepmist.models.multiframe.LS_STDNet.ska_mulch import SKA_GroupAgg  # ensure file is in PYTHONPATH or same dir
import os
import time
import numpy as np

device = 'cuda'

# config
N = 2
C = 8   # desired output channels
IC = 3 * C  # input channels
OC = C
H = 16
W = 16
KS = 3
G = 8  # groups; ensure IC % G == 0 and OC % G == 0

assert IC % G == 0 and OC % G == 0
assert G == G  # trivial

# create random tensors
torch.manual_seed(0)
x = torch.randn(N, IC, H, W, device=device, dtype=torch.float32, requires_grad=True)

# create dummy w in expected shape (N, G, KS*KS, H, W)
# In practice, your LKP should produce this; here we random-init.
w = torch.randn(N, G, KS*KS, H, W, device=device, dtype=torch.float32, requires_grad=True)

ska = SKA_GroupAgg().to(device)

# forward
start = time.time()
o = ska(x, w, OC)
torch.cuda.synchronize()
end = time.time()
print("forward time (s):", end - start)
print("o.shape:", o.shape)  # expect (N, OC, H, W)

# simple loss and backward
loss = o.sum()
loss.backward()
print("x.grad exists:", x.grad is not None)
print("w.grad exists:", w.grad is not None)
print("x.grad.norm():", x.grad.norm().item() if x.grad is not None else None)
print("w.grad.norm():", w.grad.norm().item() if w.grad is not None else None)

# quick finite-diff check for x (random element)
with torch.no_grad():
    idx = (0, 0, H//2, W//2)
    eps = 1e-3
    old = x[idx].item()
    x[idx] = old + eps
    o_pos = ska(x, w, OC).sum().item()
    x[idx] = old - eps
    o_neg = ska(x, w, OC).sum().item()
    x[idx] = old
    num_grad = (o_pos - o_neg) / (2 * eps)
    analytic = x.grad[idx].item()
    print("num_grad:", num_grad, "analytic:", analytic, "ratio:", analytic / (num_grad + 1e-12))
