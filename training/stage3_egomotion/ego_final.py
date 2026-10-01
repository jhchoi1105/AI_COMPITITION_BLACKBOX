# -*- coding: utf-8 -*-
"""배포용 최종 학습 - 전체 주행을 쓴다.

검증(주행 단위 분리)으로 구조와 epoch 수는 이미 정했다.
    속력 상관 0.961 / 요레이트 상관 0.845 / 속력 MAE 1.94 m/s  (ep35 부근)
이제 검증에 떼어 뒀던 주행까지 포함해 같은 레시피로 다시 학습한다.
표본이 약 28% 늘어난다.

**정규화 상수(mu_s, sd_s, sd_y)를 체크포인트에 같이 저장한다.** 추론 코드가
이 값을 알아야 m/s 로 되돌릴 수 있다.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "C:/bbw")
from egomotion_train import Net, augment, CACHE

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("C:/bbw/ego_final.pt")
EPOCHS, BS = 40, 128


def main():
    z = np.load(CACHE)
    X, SP, YR, DRV = z["X"], z["sp"], z["yr"], z["drive"]
    print("전체 %d표본 / 주행 %d개" % (len(X), len(np.unique(DRV))))

    mu_s, sd_s = float(SP.mean()), float(SP.std() + 1e-6)
    sd_y = float(YR.std() + 1e-6)
    tgt = np.stack([(SP - mu_s) / sd_s, YR / sd_y], 1).astype(np.float32)
    Xt, Tt = torch.from_numpy(X), torch.from_numpy(tgt)
    idx = np.arange(len(X))

    net = Net().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, 3e-3, total_steps=EPOCHS * max(1, len(idx) // BS))
    lossf = nn.SmoothL1Loss()
    rng = np.random.default_rng(1)

    for ep in range(EPOCHS):
        net.train()
        rng.shuffle(idx)
        tot = 0.0
        for b in range(0, len(idx) - BS + 1, BS):
            sel = idx[b:b + BS]
            x, flip = augment(Xt[sel].to(DEV))
            xb = torch.cat([x, (x[:, 1:2] - x[:, 0:1])], dim=1)
            yb = Tt[sel].to(DEV).clone()
            yb[flip, 1] = -yb[flip, 1]
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print("  ep%02d  loss %.4f" % (ep, tot / max(1, len(idx) // BS)), flush=True)

    # 학습 데이터 자체에 대한 적합도(검증 아님, 정상 수렴 확인용)
    net.eval()
    preds = []
    with torch.inference_mode():
        for b in range(0, len(idx), 512):
            sel = np.arange(b, min(b + 512, len(X)))
            x = Xt[sel].to(DEV).float() / 255.0
            xb = torch.cat([x, (x[:, 1:2] - x[:, 0:1])], dim=1)
            preds.append(net(xb).cpu().numpy())
    p = np.concatenate(preds)
    ps, py = p[:, 0] * sd_s + mu_s, p[:, 1] * sd_y
    print("  (학습셋 적합) 속력 상관 %.3f / 요레이트 상관 %.3f / MAE %.2f m/s"
          % (np.corrcoef(ps, SP)[0, 1], np.corrcoef(py, YR)[0, 1], np.abs(ps - SP).mean()))

    torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                "mu_s": mu_s, "sd_s": sd_s, "sd_y": sd_y,
                "input_hw": [96, 160],
                "source": "comma2k19 (commaai, MIT License) Chunk_1, 60 segments / 9 drives",
                "note": "frame_velocities + frame_orientations 를 프레임 단위 정답으로 사용. "
                        "비공개 평가 데이터는 일절 사용하지 않았다."},
               OUT)
    print("저장: %s  %.2f MB" % (OUT, OUT.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
