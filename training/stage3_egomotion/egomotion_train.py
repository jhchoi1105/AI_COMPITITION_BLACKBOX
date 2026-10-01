# -*- coding: utf-8 -*-
"""Stage 3 를 휴리스틱에서 **학습 모델**로 바꾼다 - 타당성 시험.

===========================================================================
왜 지금까지 안 했나 (내 착오)
===========================================================================
pose_label.py 의 주석에 이렇게 써 뒀었다:

    "핵심: 학습이 아니라 라벨 정의를 알아내는 것이 목적이다. (...)
     외부 데이터를 학습에 쓰지 않으므로 규정 논란이 없다."

**규정을 잘못 읽었다.** 대회 규칙 2-1 은 명시적으로 이렇게 말한다:

    "사전 학습 모델, API, 외부 데이터 수집 및 생성 등 사용에 법적 제한이
     없다면 모두 활용 가능합니다."

금지는 두 개뿐이다 - 비공개 평가 데이터로 학습(3항), 파일 간 정보 사용(4항).
**외부 데이터 학습은 허용된다.** 나는 있지도 않은 제약을 스스로 걸고 가장 큰
지렛대를 20회 넘는 제출 동안 쓰지 않았다.

===========================================================================
지금 상태
===========================================================================
Stage 3 는 Farneback 광류 + 손으로 맞춘 문턱이다. 실측(comma2k19 실제 속도 정답):

    현행 속도 대리값의 속력 상관 = 0.36
    기울기-가속도 상관         = 0.23

comma2k19 는 프레임마다 frame_velocities(ECEF 속도벡터)와
frame_orientations(쿼터니언)를 준다. 즉 **프레임 단위 진짜 정답**이 있다.
지금까지 이걸 채점자로만 썼고 학습 데이터로 쓴 적이 없다.

===========================================================================
이 시험이 답할 것
===========================================================================
작은 CNN 이 프레임 쌍에서 속력과 요레이트를 회귀하면 상관이 얼마나 오르는가.
**세그먼트 단위로 분리**해서 검증한다(프레임 단위로 섞으면 누출된다).

0.36 대비 크게 오르면 Stage 3 전체를 교체한다. 안 오르면 이 길은 접는다.
"""
import sys
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn

sys.path.insert(0, "C:/bbw")
from pose_label import segment_signals

ROOT = Path("C:/bbw/c2k19")
CACHE = Path("C:/bbw/ego_cache.npz")
H, W = 96, 160          # 입력 해상도
STRIDE = 2              # 20Hz -> 10Hz (평가와 같은 표본율)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build_cache():
    """세그먼트마다 (프레임쌍, 속력, 요레이트) 를 만들어 저장한다."""
    xs, sp, yr, seg_id, drive_id, car_id = [], [], [], [], [], []
    drives, cars = {}, {}
    segs = sorted(p for p in ROOT.iterdir() if p.is_dir())
    for si, d in enumerate(segs):
        try:
            t, speed, accel, yaw = segment_signals(d)
        except Exception as exc:
            print("  건너뜀 %s (%s)" % (d.name[:34], type(exc).__name__))
            continue
        # 새로 받은 세그먼트는 video.hevc 만 있다. OpenCV 가 직접 읽는다.
        video = d / "video.mp4" if (d / "video.mp4").exists() else d / "video.hevc"
        cap = cv2.VideoCapture(str(video))
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (W, H),
                                     interpolation=cv2.INTER_AREA))
        cap.release()
        n = min(len(frames) - 1, len(t) - 1)
        if n < 10:
            print("  건너뜀 %s (프레임 %d)" % (d.name[:34], len(frames)))
            continue
        # 주행 ID = 세그먼트 이름에서 끝의 _번호 를 뗀 것.
        # 같은 주행의 이웃 세그먼트는 도로·조명이 거의 같으므로 세그먼트 단위로
        # 나누면 누출된다. **주행 단위**로 나눠야 전이 성능이 정직하게 나온다.
        dname = d.name.rsplit("_", 1)[0]
        di = drives.setdefault(dname, len(drives))
        # 차량(dongle) ID = 세그먼트 이름의 첫 토큰. **카메라가 바뀌는 단위**다.
        ci = cars.setdefault(d.name.split("_")[0], len(cars))
        idx = list(range(0, n, STRIDE))
        for i in idx:
            xs.append(np.stack([frames[i], frames[i + 1]]))
            sp.append(speed[i])
            yr.append(yaw[i])
            seg_id.append(si)
            drive_id.append(di)
            car_id.append(ci)
        print("  %-38s %4d표본  속력 %5.1f~%5.1f m/s"
              % (d.name[:36], len(idx), speed[idx].min(), speed[idx].max()), flush=True)
    X = np.stack(xs).astype(np.uint8)
    np.savez(CACHE, X=X, sp=np.array(sp, np.float32), yr=np.array(yr, np.float32),
             seg=np.array(seg_id, np.int32), drive=np.array(drive_id, np.int32),
             car=np.array(car_id, np.int32))
    print("주행 %d개 / 차량 %d개:" % (len(drives), len(cars)), sorted(cars))
    print("캐시 저장: %s  %s  %.0f MB" % (CACHE, X.shape, CACHE.stat().st_size / 1e6))


class Net(nn.Module):
    """프레임 쌍 -> (속력, 요레이트). 작은 망이면 충분하다 - 표본이 수만 개다."""

    def __init__(self):
        super().__init__()
        def blk(i, o, s=2):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.f = nn.Sequential(
            blk(3, 24), blk(24, 48), blk(48, 96), blk(96, 128), blk(128, 160),
            nn.AdaptiveAvgPool2d(1))
        self.h = nn.Linear(160, 2)

    def forward(self, x):
        return self.h(self.f(x).flatten(1))


def augment(batch_u8):
    """장착 차이를 흉내낸다 - 배율·평행이동·좌우반전·밝기.

    평가셋은 다양한 블랙박스인데 학습 데이터는 **카메라 한 대**다. 그 격차를
    좁히는 유일한 수단이 증강이다.

    배율을 흔들면 절대 속력은 모호해지지만(같은 영상을 확대하면 더 멀리서 더
    빠른 것과 구별되지 않는다) 우리 용도에는 문제가 없다. 가감속은 **변화량**
    이고 문턱을 클립별 std 로 잡으며, STOPPED 는 0 근처 판정이라 배율과 무관하다.
    오히려 배율 불변성을 학습시키는 편이 전이에 유리하다.

    두 프레임에 **같은 변환**을 적용해야 흐름 기하가 일관된다.
    좌우반전은 요레이트 부호를 뒤집는다.
    """
    b, _, h, w = batch_u8.shape
    out = batch_u8.clone()
    flip = torch.rand(b, device=batch_u8.device) < 0.5
    out[flip] = torch.flip(out[flip], dims=[3])
    x = out.float() / 255.0
    scale = 1.0 + (torch.rand(b, device=x.device) - 0.5) * 0.35      # 0.825~1.175
    tx = (torch.rand(b, device=x.device) - 0.5) * 0.16
    ty = (torch.rand(b, device=x.device) - 0.5) * 0.16
    theta = torch.zeros(b, 2, 3, device=x.device)
    theta[:, 0, 0] = scale
    theta[:, 1, 1] = scale
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
    x = torch.nn.functional.grid_sample(x, grid, align_corners=False, padding_mode="border")
    gain = 0.75 + torch.rand(b, 1, 1, 1, device=x.device) * 0.5
    bias = (torch.rand(b, 1, 1, 1, device=x.device) - 0.5) * 0.15
    x = (x * gain + bias).clamp(0, 1)
    return x, flip


def make_input(batch_u8):
    """2프레임 + 차분을 3채널로. 차분을 명시적으로 주면 수렴이 빠르다."""
    x = batch_u8.float() / 255.0
    return torch.cat([x, (x[:, 1:2] - x[:, 0:1])], dim=1)


def main():
    if not CACHE.exists():
        print("=== 캐시 생성 ===")
        build_cache()
    z = np.load(CACHE)
    X, SP, YR, SEG, DRV, CAR = z["X"], z["sp"], z["yr"], z["seg"], z["drive"], z["car"]
    print("총 %d표본 / 세그먼트 %d / 주행 %d / **차량 %d**"
          % (len(X), len(np.unique(SEG)), len(np.unique(DRV)), len(np.unique(CAR))))

    # **세그먼트 단위 분리.** 프레임 단위로 섞으면 이웃 프레임이 양쪽에 들어가
    # 상관이 부풀려진다.
    # **차량 단위 분리.** 지금까지는 주행 단위였는데 전부 같은 카메라였다.
    # 평가셋은 다양한 블랙박스이므로 이것이 진짜 전이 시험이다.
    rng = np.random.default_rng(0)
    cars = np.unique(CAR)
    hold = cars[-1]                       # 마지막 차량을 통째로 뺀다
    tr = CAR != hold
    va = ~tr
    print("학습 %d표본(차량 %d대) / **검증 %d표본(학습에 없던 차량 1대)**"
          % (tr.sum(), len(cars) - 1, va.sum()))

    mu_s, sd_s = float(SP[tr].mean()), float(SP[tr].std() + 1e-6)
    sd_y = float(YR[tr].std() + 1e-6)
    tgt = np.stack([(SP - mu_s) / sd_s, YR / sd_y], 1).astype(np.float32)

    Xt = torch.from_numpy(X)
    Tt = torch.from_numpy(tgt)
    itr = np.where(tr)[0]
    iva = np.where(va)[0]

    net = Net().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    EPOCHS, BS = 40, 128
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, 3e-3, total_steps=EPOCHS * max(1, len(itr) // BS))
    lossf = nn.SmoothL1Loss()

    best = None
    for ep in range(EPOCHS):
        net.train()
        rng.shuffle(itr)
        tot = 0.0
        for b in range(0, len(itr) - BS + 1, BS):
            sel = itr[b:b + BS]
            x, flip = augment(Xt[sel].to(DEV))
            xb = torch.cat([x, (x[:, 1:2] - x[:, 0:1])], dim=1)
            yb = Tt[sel].to(DEV).clone()
            yb[flip, 1] = -yb[flip, 1]      # 좌우반전은 요레이트 부호를 뒤집는다
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        net.eval()
        preds = []
        with torch.inference_mode():
            for b in range(0, len(iva), 256):
                sel = iva[b:b + 256]
                preds.append(net(make_input(Xt[sel].to(DEV))).cpu().numpy())
        p = np.concatenate(preds)
        ps = p[:, 0] * sd_s + mu_s
        py = p[:, 1] * sd_y
        r_s = float(np.corrcoef(ps, SP[iva])[0, 1])
        r_y = float(np.corrcoef(py, YR[iva])[0, 1])
        mae = float(np.abs(ps - SP[iva]).mean())
        if best is None or r_s > best[0]:
            best = (r_s, r_y, mae, ep)
            torch.save(net.state_dict(), "C:/bbw/ego_net.pt")
        print("  ep%02d  loss %.4f   검증 속력상관 %.3f  요레이트상관 %.3f  속력MAE %.2f m/s"
              % (ep, tot / max(1, len(itr) // BS), r_s, r_y, mae), flush=True)

    print()
    print("=== 결과 (**차량 단위** 분리 검증 - 학습에 없던 카메라) ===")
    print("  최고 속력 상관   **%.3f**   (현행 광류 대리값 0.36)" % best[0])
    print("  같은 시점 요레이트 상관 %.3f" % best[1])
    print("  속력 MAE %.2f m/s  (ep%d)" % (best[2], best[3]))
    print()
    print("이 숫자는 **학습에 한 번도 안 나온 카메라**에 대한 것이다.")
    print("주행 단위 분리(같은 카메라)에서는 속력 상관 0.961 이었다. 차이가")
    print("곧 카메라 전이 손실이고, 평가셋의 다양한 블랙박스에 대한 현실적 상한이다.")


if __name__ == "__main__":
    main()
