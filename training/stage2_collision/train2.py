# 충돌 순간 판별기: 프레임 t 주변 6장(t-3..t+2, 160x90 회색) -> 점수. 클립 단위 softmax CE.
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, sys, time
SEED = int([a for a in sys.argv if a.startswith("--seed=")][0][7:]) if any(a.startswith("--seed=") for a in sys.argv) else 0
torch.manual_seed(SEED); np.random.seed(SEED)
z = np.load('C:/bbw/cm/ccd_crash.npz'); X = z['X']; on = z['onset']; ego = z['ego']
h = np.load('C:/bbw/cm/heur.npz')
EGO_ONLY = '--all' not in sys.argv
idx = np.arange(len(X)); rng = np.random.RandomState(0)
egoi = idx[ego]; rng.shuffle(egoi); val = np.sort(egoi[:160])
tr = np.array([i for i in idx if i not in set(val) and (ego[i] or not EGO_ONLY)])
print('train', len(tr), 'val', len(val))
K0, K1 = 3, 2  # 과거 3, 미래 2
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        def b(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        s.f = nn.Sequential(b(K0 + K1 + 1 + K0 + K1, 32), b(32, 64), b(64, 96), b(96, 128), nn.AdaptiveAvgPool2d(1))
        s.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 1))
    def forward(s, x): return s.h(s.f(x)).squeeze(-1)
def windows(clips):  # clips: B,T,H,W float -> B*T, C, H, W
    B, T, H, W = clips.shape
    p = torch.cat([clips[:, :1].expand(B, K0, H, W), clips, clips[:, -1:].expand(B, K1, H, W)], 1)
    fr = torch.stack([p[:, k:k + T] for k in range(K0 + K1 + 1)], 2)  # B,T,6,H,W
    df = (fr[:, :, 1:] - fr[:, :, :-1]) * 4.0
    return torch.cat([fr, df], 2).reshape(B * T, -1, H, W)
dev = 'cuda'; net = Net().to(dev)
opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
T = 50; tt = torch.arange(T, dtype=torch.float32)
def target(o):
    g = torch.exp(-0.5 * ((tt[None] - torch.tensor(o, dtype=torch.float32)[:, None]) / 1.0) ** 2)
    return g / g.sum(1, keepdim=True)
def scores(ids, aug=False):
    c = torch.from_numpy(X[ids]).float().to(dev) / 255.0
    if aug:
        if np.random.rand() < 0.5: c = c.flip(-1)
        c = c * (0.7 + 0.6 * torch.rand(len(ids), 1, 1, 1, device=dev)) + 0.1 * (torch.rand(len(ids), 1, 1, 1, device=dev) - 0.5)
    c = (c - 0.45) / 0.25
    return net(windows(c)).reshape(len(ids), T)
EP = 16; steps = EP * (len(tr) // 16)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
def evaluate():
    net.eval(); S = []
    with torch.no_grad():
        for i in range(0, len(val), 32): S.append(scores(val[i:i + 32]).cpu())
    net.train(); return torch.cat(S).numpy()
t0 = time.time(); best = -1
for ep in range(EP):
    perm = np.random.permutation(tr)
    for i in range(0, len(perm) - 15, 16):
        ids = np.sort(perm[i:i + 16])
        loss = -(target(on[ids]).to(dev) * F.log_softmax(scores(ids, True), 1)).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    S = evaluate(); pred = S.argmax(1); hit = (np.abs(pred - on[val]) <= 3).mean()
    print('ep %d loss %.3f val hit %.3f (%.0fs)' % (ep, loss.item(), hit, time.time() - t0), flush=True)
    best = max(best, hit); torch.save(net.state_dict(), 'C:/bbw/cm/coll_%s_s%d_ep%d.pt' % ('ego' if EGO_ONLY else 'all', SEED, ep)); np.save('C:/bbw/cm/valS_%s_s%d_ep%d.npy' % ('ego' if EGO_ONLY else 'all', SEED, ep), S)
hh = (np.abs(h['col'][val] - on[val]) <= 3).mean()
print('heuristic val hit %.3f | model best %.3f' % (hh, best))
