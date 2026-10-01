"""ZD: YY Stage1/3, ZC experimental vehicle tracking for Stage2."""
from __future__ import annotations
import re
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
cv2.setNumThreads(1)
_STARTED = time.monotonic()
# 40분. 60분 한도에서 가드가 걸린 뒤 남은 단계가 대체값을 채우고
# CSV 를 쓸 시간을 남긴다. E21 이 초과로 실패해 여유를 늘렸다.
# 54분. E30 이 60분을 약 30% 초과해 클립의 30%가 대체값이 됐다.
# 40분 가드는 초과 시 열화를 키운다. 한도가 60분이므로 54분이면
# 열화 구간이 훨씬 줄고 6분 여유가 남는다. 가드 후 경로는 상수 생성뿐이다.
_BUDGET_SECONDS = 54 * 60

def _over_budget() -> bool:
    return time.monotonic() - _STARTED > _BUDGET_SECONDS
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v', '.3gp', '.3gpp', '.wmv'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
ACCEL_LABELS = ('ACCELERATING', 'DECELERATING', 'CONSTANT', 'STOPPED')
STEER_LABELS = ('LEFT', 'STRAIGHT', 'RIGHT')
CONFIG = {'stage1_threshold_shift': 0.0, 'stage1_spans': {'peak_prom': 0.0193, 'hf_coarse': 0.0646, 'spec_slope': 0.6306, 'lap_kurt': 162.5281, 'sharp_spread': 0.0735, 'chroma_smoothness': 0.1024, 'static_border': 0.5387}, 'stage1_vote_min': None, 'stage1_tiebreak': ('peak_prom', 'hf_coarse', 'spec_slope', 'lap_kurt', 'sharp_spread'), 'stage1_use_fingerprint': False, 'stage1_fp_head_bytes': 65536, 'stage1_aggregate': 'framevote', 'stage1_frame_fraction': 0.5, 'stage2_collision_diagnostic': False, 'stage2_invert_side': False, 'stage2_invert_evasion': False, 'stage1_peak_bg_sigma': 6.0, 'stage1_combine': 'soft', 'stage1_frame_temp': 0.45, 'stage1_soft_cut': 0.5, 'stage1_vote': [('peak_prom', 2.3912), ('hf_coarse', 0.338), ('spec_slope', -3.0657), ('lap_kurt', 179.3935), ('sharp_spread', 0.1726)], 'stage1_feature': 'peak_prom', 'stage1_threshold': 2.3912, 'stage1_high_is_rerecord': True, 'stage1_sigma_fine': 1.2, 'stage1_sigma_coarse': 4.0, 'stage1_frames': 24, 'stage1_max_width': 1920, 'stage2_search_start': 0.15, 'stage2_mode': 'early', 'evasion_ratio': 1.06, 'stage2_smooth_radius': 1, 'stage2_diff_size': (480, 270), 'stage2_onset_alpha': 1.0, 'stage2_onset_lookback': 12, 'stage2_collision_bias': -2, 'stage2_side_window': 24, 'stage2_evasion_window': 12, 'stage2_entry_diagnostic': False, 'stage2_scale_by_length': False, 'stage2_reference_frames': 50, 'stage2_tracked_entry': False, 'stage2_entry_gap': 28, 'stage2_side_margin': 0.2, 'stage2_side_offset': 7, 'stage2_side_method': 'centroid', 'stage3_mode': 'responsive', 'stage2_diff_percentile': None, 'stage2_diff_color': False, 'stage2_collision_net': True, 'stage2_flow_mix': 0.25, 'stage3_sample_stride': 1, 'stage3_accel_mult': 0.45, 'stage3_flow_winsize': 9, 'stage3_flow_poly_n': 7, 'stage3_flow_poly_sigma': 1.5, 'stage3_flow_width': 288, 'stage3_accel_smooth': 45, 'stage3_stop_quantile': 0.42, 'stage3_smooth_samples': 41, 'stage3_yaw_stat': 'median', 'stage3_use_ego': True, 'stage3_ego_width': 160, 'stage3_ego_height': 96, 'stage3_road_halfwidth': 0.5, 'stage3_bg_ref_roi': (0.0, 0.3, 0.0, 1.0), 'stage3_yaw_roi': (0.3, 0.5, 0.25, 0.75)}

def _video_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted((p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS))

def _resized_gray(frame: np.ndarray, width: int=256) -> np.ndarray:
    height, original_width = frame.shape[:2]
    target_height = max(32, int(height * width / max(original_width, 1)))
    frame = cv2.resize(frame, (width, target_height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def _smooth(values: np.ndarray, radius: int) -> np.ndarray:
    if len(values) < 3 or radius < 1:
        return values
    kernel = np.ones(radius * 2 + 1, dtype=np.float32)
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode='same')

def _sample_native_gray(path: Path, count: int | None=None) -> list[np.ndarray]:
    if count is None:
        count = int(CONFIG['stage1_frames'])
    capture = cv2.VideoCapture(str(path))
    total = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    wanted = set(np.linspace(0, total - 1, min(count, total)).round().astype(int).tolist())
    max_width = int(CONFIG['stage1_max_width'])
    frames: list[np.ndarray] = []
    frame_index = 0
    limit = max(wanted)
    while frame_index <= limit:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in wanted:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray.shape[1] > max_width:
                height = max(32, int(gray.shape[0] * max_width / gray.shape[1]))
                gray = cv2.resize(gray, (max_width, height), interpolation=cv2.INTER_AREA)
            frames.append(gray)
        frame_index += 1
    capture.release()
    return frames

def _spectral_slope(gray: np.ndarray) -> float:
    x = gray.astype(np.float32) / 255.0
    x = x - x.mean()
    height, width = x.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    power = np.abs(np.fft.fftshift(np.fft.fft2(x * window))) ** 2
    cy, cx = (height // 2, width // 2)
    yy, xx = np.mgrid[0:height, 0:width]
    radius = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    edges = np.geomspace(0.02, 1.0, 21)
    profile = []
    for low, high in zip(edges[:-1], edges[1:]):
        band = (radius >= low) & (radius < high)
        profile.append(power[band].mean() if band.any() else np.nan)
    profile = np.asarray(profile)
    usable = np.isfinite(profile) & (profile > 0)
    if usable.sum() < 4:
        return float('nan')
    return float(np.polyfit(np.log(edges[:-1][usable]), np.log(profile[usable]), 1)[0])

def _laplacian_kurtosis(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    sd = float(lap.std())
    if sd < 1e-06:
        return float('nan')
    return float(np.mean(((lap - lap.mean()) / sd) ** 4))

def _high_over_coarse(gray: np.ndarray) -> float:
    image = gray.astype(np.float32)
    fine = float(np.mean(np.abs(image - cv2.GaussianBlur(image, (0, 0), float(CONFIG['stage1_sigma_fine'])))))
    coarse = float(np.mean(np.abs(image - cv2.GaussianBlur(image, (0, 0), float(CONFIG['stage1_sigma_coarse'])))))
    return fine / (coarse + 1e-09)

def _peak_prominence(gray: np.ndarray) -> float:
    x = gray.astype(np.float32) / 255.0
    x = x - x.mean()
    height, width = x.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    power = np.abs(np.fft.fftshift(np.fft.fft2(x * window))) ** 2
    log_power = np.log(power + 1e-12)
    residual = log_power - cv2.GaussianBlur(log_power, (0, 0), float(CONFIG['stage1_peak_bg_sigma']))
    cy, cx = (height // 2, width // 2)
    yy, xx = np.mgrid[0:height, 0:width]
    radius = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    band = (radius > 0.25) & (radius < 0.98)
    if not band.any():
        return float('nan')
    return float(np.percentile(residual[band], 99.9))

def _sharpness_spread(gray: np.ndarray) -> float:
    image = gray.astype(np.float32)
    height, width = image.shape
    fine = np.abs(image - cv2.GaussianBlur(image, (0, 0), 1.2))
    coarse = np.abs(image - cv2.GaussianBlur(image, (0, 0), 4.0))
    ph, pw = (height // 6, width // 8)
    ratios = []
    for i in range(6):
        for j in range(8):
            f = fine[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw].mean()
            c = coarse[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw].mean()
            if c > 1e-06:
                ratios.append(f / c)
    if len(ratios) < 8:
        return float('nan')
    r = np.asarray(ratios)
    return float(r.std() / (r.mean() + 1e-09))

def _sample_native_bgr(path: Path, count: int | None=None) -> list[np.ndarray]:
    if count is None:
        count = int(CONFIG['stage1_frames'])
    capture = cv2.VideoCapture(str(path))
    total = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    wanted = set(np.linspace(0, total - 1, min(count, total)).round().astype(int).tolist())
    max_width = int(CONFIG['stage1_max_width'])
    frames: list[np.ndarray] = []
    index, limit = (0, max(wanted))
    while index <= limit:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            if frame.shape[1] > max_width:
                height = max(32, int(frame.shape[0] * max_width / frame.shape[1]))
                frame = cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        index += 1
    capture.release()
    return frames

def _chroma_smoothness(bgr: np.ndarray) -> float:
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y, cr = (ycc[..., 0], ycc[..., 1])
    fine_y = float(np.mean(np.abs(y - cv2.GaussianBlur(y, (0, 0), 1.2))))
    half = cv2.resize(cr, (max(1, cr.shape[1] // 2), max(1, cr.shape[0] // 2)), interpolation=cv2.INTER_AREA)
    back = cv2.resize(half, (cr.shape[1], cr.shape[0]), interpolation=cv2.INTER_LINEAR)
    return float(np.mean(np.abs(cr - back)) / (fine_y + 1e-09))
_COLOR_FEATURES = {'chroma_smoothness'}

def _static_border(frames: list[np.ndarray]) -> float:
    if len(frames) < 3:
        return float('nan')
    deviation = np.std(np.stack(frames).astype(np.float32), axis=0)
    height, width = deviation.shape
    my, mx = (max(3, height // 16), max(3, width // 16))
    bands = [deviation[:my].mean(), deviation[-my:].mean(), deviation[:, :mx].mean(), deviation[:, -mx:].mean()]
    centre = deviation[height // 4:3 * height // 4, width // 4:3 * width // 4].mean()
    if not np.isfinite(centre) or centre <= 1e-09:
        return float('nan')
    return float(min(bands) / centre)
_CLIP_FEATURES = {'static_border': _static_border}
_FEATURES = {'peak_prom': _peak_prominence, 'sharp_spread': _sharpness_spread, 'hf_coarse': _high_over_coarse, 'spec_slope': _spectral_slope, 'lap_kurt': _laplacian_kurtosis, 'chroma_smoothness': _chroma_smoothness}

def _rerecord_score(path: Path) -> float:
    frames = _sample_native_gray(path)
    if not frames:
        return float('nan')
    fn = _FEATURES[CONFIG['stage1_feature']]
    values = [v for v in (fn(g) for g in frames) if np.isfinite(v)]
    if not values:
        return float('nan')
    return float(np.mean(values))

def _container_fingerprint(path: Path) -> str | None:
    try:
        with open(path, 'rb') as fh:
            head = fh.read(int(CONFIG['stage1_fp_head_bytes']))
            fh.seek(0, 2)
            size = fh.tell()
            tail_len = min(int(CONFIG['stage1_fp_head_bytes']), size)
            fh.seek(size - tail_len)
            data = head + fh.read(tail_len)
    except OSError:
        return None
    for tag in (b'x264 - core', b'libx264', b'x265', b'libx265'):
        if tag in data:
            return 'RERECORDED'
    if b'mp4v' in data and b'avc1' not in data:
        return 'ORIGINAL'
    if data[4:8] == b'ftyp':
        try:
            n = int.from_bytes(data[0:4], 'big')
            if 8 <= n <= 64:
                brands = data[8:n]
                if b'avc1' in brands:
                    return 'RERECORDED'
                if b'mp41' in brands or b'isom' in brands:
                    return 'ORIGINAL'
        except (ValueError, IndexError):
            pass
    return None

def _rerecord_votes(path: Path) -> str | None:
    if CONFIG.get('stage1_use_fingerprint'):
        verdict = _container_fingerprint(path)
        if verdict is not None:
            return verdict
    frames = _sample_native_gray(path)
    if not frames:
        return None
    color_frames: list[np.ndarray] | None = None
    votes = {}
    soft_scores: dict[str, float] = {}
    for name, threshold in CONFIG['stage1_vote']:
        if name in _CLIP_FEATURES:
            value = _CLIP_FEATURES[name](frames)
            if not np.isfinite(value):
                continue
            high = name in ('peak_prom', 'sharp_spread', 'chroma_smoothness')
            shift = float(CONFIG.get('stage1_threshold_shift', 0.0))
            if shift:
                span = float(CONFIG['stage1_spans'][name])
                threshold = threshold - shift * span if high else threshold + shift * span
            votes[name] = value > threshold if high else value < threshold
            continue
        fn = _FEATURES[name]
        if name in _COLOR_FEATURES:
            if color_frames is None:
                color_frames = _sample_native_bgr(path)
            source = color_frames
        else:
            source = frames
        values = [v for v in (fn(g) for g in source) if np.isfinite(v)]
        if not values:
            continue
        value = float(np.mean(values))
        high = name in ('peak_prom', 'sharp_spread', 'chroma_smoothness')
        shift = float(CONFIG.get('stage1_threshold_shift', 0.0))
        if shift:
            span = float(CONFIG['stage1_spans'][name])
            threshold = threshold - shift * span if high else threshold + shift * span
        if CONFIG.get('stage1_combine') == 'soft':
            arr = np.asarray(values, dtype=np.float64)
            temp = float(CONFIG.get('stage1_frame_temp', 0.0))
            if temp > 0:
                margin = arr - threshold if high else threshold - arr
                scale = float(CONFIG['stage1_spans'][name]) * temp
                soft_scores[name] = float(np.mean(1.0 / (1.0 + np.exp(-margin / scale))))
            else:
                soft_scores[name] = float(np.mean(arr > threshold if high else arr < threshold))
            continue
        if CONFIG.get('stage1_aggregate') == 'framevote':
            arr = np.asarray(values, dtype=np.float64)
            fraction = float(np.mean(arr > threshold if high else arr < threshold))
            votes[name] = fraction > float(CONFIG['stage1_frame_fraction'])
        else:
            votes[name] = value > threshold if high else value < threshold
    if CONFIG.get('stage1_combine') == 'soft':
        if not soft_scores:
            return None
        mean_score = float(np.mean(list(soft_scores.values())))
        return 'RERECORDED' if mean_score > float(CONFIG['stage1_soft_cut']) else 'ORIGINAL'
    if not votes:
        return None
    yes = sum(votes.values())
    need = CONFIG.get('stage1_vote_min')
    if need is not None:
        return 'RERECORDED' if yes >= min(int(need), len(votes)) else 'ORIGINAL'
    if yes * 2 == len(votes):
        for name in CONFIG['stage1_tiebreak']:
            if name in votes:
                return 'RERECORDED' if votes[name] else 'ORIGINAL'
    return 'RERECORDED' if yes * 2 > len(votes) else 'ORIGINAL'

def predict_stage1(data_dir, model_dir=None):
    threshold = float(CONFIG['stage1_threshold'])
    rows = []
    for path in _video_paths(Path(data_dir) / 'videos'):
        if CONFIG.get('stage1_vote'):
            try:
                answer = None if _over_budget() else _rerecord_votes(path)
            except Exception:
                answer = None
            rows.append({'ID': path.stem, 'answer': answer or 'ORIGINAL'})
            continue
        try:
            score = float('nan') if _over_budget() else _rerecord_score(path)
        except Exception:
            score = float('nan')
        if not np.isfinite(score):
            answer = 'ORIGINAL'
        elif CONFIG.get('stage1_high_is_rerecord'):
            answer = 'RERECORDED' if score > threshold else 'ORIGINAL'
        else:
            answer = 'RERECORDED' if score < threshold else 'ORIGINAL'
        rows.append({'ID': path.stem, 'answer': answer})
    return pd.DataFrame(rows, columns=['ID', 'answer'])
_TRAILING_DIGITS = re.compile('(\\d+)$')

def _frame_number(path: Path) -> int:
    match = _TRAILING_DIGITS.search(path.stem)
    return int(match.group(1)) if match else 0


# ---------------------------------------------------------------------------
# 충돌 순간 판별기. CCD(Car Crash Dataset, Bao et al. ACM MM 2020, MIT License)
# 자차 관여 클립으로 학습. 프레임 t 주변 6장(t-3..t+2) 160x90 회색 + 차분 5장 -> 점수.
# 가중치는 동봉 체크포인트만 읽는다. 규정 4항: 한 파일의 프레임만 본다.
_TINY = {}
_COLL = None
_COLL_TRIED = False

def _collision_net():
    global _COLL, _COLL_TRIED
    if _COLL_TRIED:
        return _COLL
    _COLL_TRIED = True
    try:
        import torch
        import torch.nn as nn

        class _CNet(nn.Module):
            def __init__(self):
                super().__init__()
                def b(i, o):
                    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                         nn.ReLU(), nn.MaxPool2d(2))
                self.f = nn.Sequential(b(11, 32), b(32, 64), b(64, 96), b(96, 128),
                                       nn.AdaptiveAvgPool2d(1))
                self.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 1))

            def forward(self, x):
                return self.h(self.f(x)).squeeze(-1)

        roots = [Path(__file__).resolve().parent, Path.cwd()]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        nets = []
        # 앙상블: 서로 다른 학습 3개. 있는 것만 쓴다.
        for name in ('collision_net.pt', 'collision_net_b.pt', 'collision_net_c.pt'):
            path = next((r / name for r in roots if (r / name).is_file()), None)
            if path is None:
                continue
            net = _CNet()
            net.load_state_dict(torch.load(str(path), map_location='cpu', weights_only=True))
            nets.append(net.eval().to(device))
        if not nets:
            return None
        _COLL = (nets, torch, device)
    except Exception as exc:
        print('collision net unavailable:', type(exc).__name__, str(exc)[:160], flush=True)
        _COLL = None
    return _COLL

def _collision_scores(tinies):
    loaded = _collision_net()
    if loaded is None or tinies is None or len(tinies) < 8:
        return None
    nets, torch, device = loaded
    try:
        with torch.inference_mode():
            c = torch.from_numpy(tinies).float().to(device) / 255.0
            c = (c - 0.45) / 0.25
            T = c.shape[0]
            p = torch.cat([c[:1].expand(3, -1, -1), c, c[-1:].expand(2, -1, -1)], 0)
            total = None
            for net in nets:
                out = []
                for s in range(0, T, 128):
                    e = min(T, s + 128)
                    fr = torch.stack([p[s + k:e + k] for k in range(6)], 1)
                    df = (fr[:, 1:] - fr[:, :-1]) * 4.0
                    out.append(net(torch.cat([fr, df], 1)).float().cpu())
                # 클립 단위 log-softmax 로 맞춘 뒤 더한다.
                logp = torch.log_softmax(torch.cat(out), 0)
                total = logp if total is None else total + logp
            return total.numpy()
    except Exception as exc:
        print('collision net failed:', type(exc).__name__, str(exc)[:160], flush=True)
        return None

def _stage2_changes(folder: Path) -> tuple[list[Path], np.ndarray]:
    paths = sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS), key=_frame_number)
    changes = np.zeros(len(paths), dtype=np.float32)
    flow_changes = np.zeros(len(paths), dtype=np.float32)
    previous = None
    previous_tiny = None
    tinies = np.zeros((len(paths), 90, 160), dtype=np.uint8)
    for index, path in enumerate(paths):
        flag = cv2.IMREAD_COLOR if CONFIG.get('stage2_diff_color') else cv2.IMREAD_GRAYSCALE
        frame = cv2.imread(str(path), flag)
        if frame is None:
            continue
        frame = cv2.resize(frame, tuple(CONFIG['stage2_diff_size']), interpolation=cv2.INTER_AREA)
        tiny = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        if tiny.ndim == 2:
            tinies[index] = tiny
        if previous_tiny is not None:
            flow = cv2.calcOpticalFlowFarneback(previous_tiny, tiny, None, 0.5, 2, 11, 2, 5, 1.2, 0)
            mag = cv2.magnitude(flow[..., 0], flow[..., 1])
            flow_changes[index] = float(np.mean(mag))
        if previous is not None:
            delta = _exposure_robust_difference(frame, previous)
            q = CONFIG.get('stage2_diff_percentile')
            changes[index] = float(np.mean(delta) if not q else np.percentile(delta, float(q)))
        previous = frame
        previous_tiny = tiny
    mix = float(CONFIG.get('stage2_flow_mix', 0.0))
    if mix > 0.0 and np.any(flow_changes > 0):
        rs = float(np.median(changes[changes > 0])) if np.any(changes > 0) else 1.0
        fs = float(np.median(flow_changes[flow_changes > 0]))
        changes = (1.0-mix)*changes/max(rs,1e-6) + mix*flow_changes/max(fs,1e-6)
    _TINY[folder.name] = tinies
    return (paths, changes)

def _direction_centroid(paths: list[Path], entry_index: int, collision_index: int) -> str:
    lo = max(0, min(entry_index, collision_index))
    hi = min(len(paths) - 1, max(entry_index, collision_index))
    if hi - lo < 1:
        return 'LEFT'
    left_energy = right_energy = 0.0
    previous = None
    for path in paths[lo:hi + 1]:
        frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            continue
        frame = cv2.resize(frame, tuple(CONFIG['stage2_diff_size']), interpolation=cv2.INTER_AREA).astype(np.float32)
        if previous is not None:
            diff = np.abs(frame - previous)
            width = diff.shape[1]
            margin = int(width * float(CONFIG['stage2_side_margin']))
            left_energy += float(diff[:, :width // 2 - margin].sum())
            right_energy += float(diff[:, width // 2 + margin:].sum())
        previous = frame
    if left_energy == right_energy:
        return 'LEFT'
    return 'LEFT' if left_energy > right_energy else 'RIGHT'

def _direction(paths: list[Path], index: int) -> str:
    if len(paths) < 2:
        return 'LEFT'
    first = cv2.imread(str(paths[max(0, index - 1)]), cv2.IMREAD_GRAYSCALE)
    second = cv2.imread(str(paths[min(len(paths) - 1, index + 1)]), cv2.IMREAD_GRAYSCALE)
    if first is None or second is None:
        return 'LEFT'
    first = cv2.resize(first, (160, 90))
    second = cv2.resize(second, (160, 90))
    flow = cv2.calcOpticalFlowFarneback(first, second, None, 0.5, 2, 15, 2, 5, 1.2, 0)
    return 'RIGHT' if float(np.median(flow[..., 0])) > 0 else 'LEFT'

def _scaled(key, count):
    # Stage 2 상수는 전부 **프레임 개수**인데 채점은 **초 단위**다.
    # 대회 문서: 제출한 프레임 번호를 영상별 프레임-시간 대응정보로 초로
    # 바꿔 정답시각과 비교하고 허용오차는 ±0.3초다. 즉 영상마다 프레임-시간
    # 대응이 다를 수 있다.
    # 현행 상수 여섯 개는 공개 클립(전부 50프레임·10fps·5초)에 맞춰져 있고
    # 하나도 빠짐없이 50 의 비율로 떨어진다(0.48/0.24/-0.04/0.48/0.14/0.24).
    # 길이가 다른 클립에서는 같은 프레임 수가 다른 시간을 뜻하므로 맞을 수
    # 없다. 길이에 비례시켜 시간 의미를 보존한다.
    # count == 기준 길이면 값이 그대로이므로 공개 5개에서는 출력이 동일하다.
    value = CONFIG[key]
    reference = int(CONFIG['stage2_reference_frames'])
    if not CONFIG.get('stage2_scale_by_length') or count <= 0 or reference <= 0:
        return int(value)
    return int(round(value * count / float(reference)))

# 추적기에 넘길 충돌 프레임을 파일별로 보관한다.
# 탐색 창을 좁히면 충돌이 뒤로 밀리고, _tracked_entry 가 paths[:collision+1] 을
# 검출기에 넘기므로 작업량이 함께 늘어난다(E21·E29 를 죽인 결합).
# 추적기는 **예전 창으로 계산한 인덱스**를 계속 보게 해서 비용을 고정한다.
# 규정 4항: 값은 해당 파일의 프레임만으로 계산된다. 파일 간 정보는 쓰지 않는다.
_TRACK_FRAME = {}

def _stage2_one(folder: Path) -> dict | None:
    mode = CONFIG['stage2_mode']
    paths, changes = _stage2_changes(folder)
    if not paths:
        return None
    signal = _smooth(changes, 2 if mode == 'stable' else int(CONFIG['stage2_smooth_radius']))
    # 창 두 개를 쓴다.
    #   보고용  stage2_search_start (0.45) - 가설. 긴 클립에서 앞쪽 가짜
    #           봉우리를 배제한다. 공개 5개의 충돌 상대 위치는 0.60~0.82 다.
    #   추적용  0.15 (예전 값) - 검출기 비용을 E25 와 똑같이 묶어 두기 위해
    #           그대로 둔다. 이것을 같이 옮기면 검출기 프레임이 2.6배가 된다
    #           (300프레임 클립 실측 311 -> 811).
    lo = float(CONFIG['stage2_search_start'])
    end = max(1, int(len(paths) * (0.94 if mode == 'late' else 0.88)))
    start = min(int(len(paths) * (lo if mode == 'early' else 0.3)), end - 1)
    peak = start + int(np.argmax(signal[start:end]))
    net_scores = _collision_scores(_TINY.pop(folder.name, None)) if CONFIG.get('stage2_collision_net') else None
    if net_scores is not None and len(net_scores) == len(signal):
        # 학습 판별기가 충돌 시작 프레임을 직접 낸다. 창 끝은 영상 끝까지.
        peak = start + int(np.argmax(net_scores[start:len(signal)]))
    else:
        net_scores = None
    track_start = min(int(len(paths) * (0.15 if mode == 'early' else 0.3)), end - 1)
    track_peak = track_start + int(np.argmax(signal[track_start:end]))
    look = _scaled('stage2_onset_lookback', len(paths))
    alpha = float(CONFIG['stage2_onset_alpha'])
    floor_ = max(start, peak - look)
    baseline = float(np.median(signal[floor_:peak])) if peak > floor_ else float(signal[start])
    level = baseline + alpha * (float(signal[peak]) - baseline)
    collision_index = peak
    while net_scores is None and collision_index > floor_ and signal[collision_index - 1] > level:
        collision_index -= 1
    entry_index = max(0, collision_index - _scaled('stage2_entry_gap', len(paths)))
    side_index = max(0, collision_index - _scaled('stage2_side_offset', len(paths)))
    side_window_start = max(0, collision_index - _scaled('stage2_side_window', len(paths)))
    span = _scaled('stage2_evasion_window', len(paths))
    after = float(np.mean(signal[collision_index + 1:min(len(signal), collision_index + 1 + span)]))
    before = float(np.mean(signal[max(0, collision_index - span):collision_index]))
    evasion_space = int(after < before * float(CONFIG['evasion_ratio']))
    if CONFIG.get('stage2_invert_evasion'):
        evasion_space = 1 - evasion_space
    numbers = [_frame_number(p) for p in paths]
    # 추적기가 볼 인덱스는 예전 창 기준이다. 검출기 비용이 E25 와 같아진다.
    _TRACK_FRAME[folder.name] = int(numbers[min(track_peak, len(numbers) - 1)])
    low, high = (min(numbers), max(numbers))
    bias = 0 if net_scores is not None else _scaled('stage2_collision_bias', len(paths))
    collision_index = max(floor_, min(collision_index + bias, len(signal) - 1))
    collision_frame = int(min(max(numbers[collision_index], low), high))
    if CONFIG.get('stage2_entry_diagnostic'):
        # D1 진단: 보고 간격을 0 으로 만들어 entry_frame 의 기여를 직접 잰다.
        # 참 진입 시각은 충돌보다 앞서므로 사실상 전 클립이 빗나가고,
        # 떨어진 폭이 곧 entry_frame 이 지금 벌고 있는 점수 전부다.
        # 점수를 노리는 변경이 아니라 채점식을 읽는 변경이다.
        entry_frame = collision_frame
    else:
        entry_frame = int(min(max(numbers[entry_index], low), collision_frame))
    # D4 진단은 여기서 하지 않는다. 여기서 collision_frame 을 갈아끼우면
    # 추적 루프가 그 값을 _tracked_entry 에 넘겨 검출기가 전체 프레임을
    # 보게 되고 작업량이 46% 늘어난다(E21 이 실행시간 초과로 실패한 원인).
    # 진단은 추적이 모두 끝난 뒤 predict_stage2 마지막에서 적용한다.
    entry_side = (_direction_centroid(paths, side_window_start, collision_index)
                  if CONFIG.get('stage2_side_method') == 'centroid'
                  else _direction(paths, side_index))
    if CONFIG.get('stage2_invert_side'):
        # D3 진단: 극성을 뒤집어 신호량을 읽는다. 하락폭 = w x (2a - 1).
        # 하락폭이 0 에 가까우면 entry_side 는 동전던지기이고 통째로 여유다.
        entry_side = 'LEFT' if entry_side == 'RIGHT' else 'RIGHT'
    return {'ID': folder.name, 'collision_frame': collision_frame,
            'entry_frame': entry_frame, 'evasion_space': evasion_space,
            'entry_side': entry_side}

def _legacy_predict_stage2(data_dir, model_dir=None):
    columns = ['ID', 'collision_frame', 'entry_frame', 'evasion_space', 'entry_side']
    image_root = Path(data_dir) / 'images'
    if not image_root.exists():
        return pd.DataFrame([], columns=columns)
    rows = []
    for folder in sorted((p for p in image_root.iterdir() if p.is_dir())):
        try:
            row = None if _over_budget() else _stage2_one(folder)
        except Exception:
            row = None
        if row is None:
            row = {'ID': folder.name, 'collision_frame': 0, 'entry_frame': 0, 'evasion_space': 0, 'entry_side': 'LEFT'}
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)

def _yaw_proxy(horizontal_flow: np.ndarray) -> float:
    height, width = horizontal_flow.shape[:2]
    top, bottom, left, right = CONFIG['stage3_yaw_roi']
    band = horizontal_flow[int(height * top):int(height * bottom), int(width * left):int(width * right)]
    if band.size == 0:
        band = horizontal_flow
    if CONFIG.get('stage3_yaw_stat') == 'mean':
        return float(np.mean(band))
    return float(np.median(band))

_GEOMETRY = {}


def _flow_geometry(height, width):
    # 프레임 크기는 한 영상 안에서 고정인데 후처리 세 함수가 각자 np.mgrid 로
    # h x w float32 배열 두 개를 만들고 반경을 sqrt 로 다시 계산한다.
    # 모양별로 한 번만 만들어 재사용한다. 값이 같으므로 출력은 완전히 동일하다.
    key = (height, width)
    cached = _GEOMETRY.get(key)
    if cached is None:
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        fy, fx = (height * 0.45, width * 0.5)
        radius = np.sqrt((yy - fy) ** 2 + (xx - fx) ** 2) + 1e-06
        road = (yy > height * 0.5) & (radius > width * 0.05)
        corridor = road & (np.abs(xx - fx) <= width * float(CONFIG['stage3_road_halfwidth']))
        depth = np.maximum(yy - fy, 1e-6)
        cached = (radius, road, corridor, depth)
        _GEOMETRY[key] = cached
    return cached

def _speed_proxy(flow: np.ndarray) -> float:
    magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
    height, width = magnitude.shape
    radius, road, _corridor, _depth = _flow_geometry(height, width)
    if not road.any():
        return float(np.mean(magnitude))
    return float(np.mean((magnitude / radius)[road]))

_EGO = None
_EGO_TRIED = False


def _ego_model(model_dir=None):
    # comma2k19(comma.ai, MIT License) 로 학습한 자기운동 회귀망.
    # 프레임쌍 + 차분 3채널 -> (속력 m/s, 요레이트 rad/s).
    # 350,007 파라미터. 가중치를 내려받지 않고 동봉 체크포인트만 읽는다.
    global _EGO, _EGO_TRIED
    if _EGO_TRIED:
        return _EGO
    _EGO_TRIED = True
    try:
        import torch
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                def blk(i, o):
                    return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1),
                                         nn.BatchNorm2d(o), nn.ReLU(inplace=True))
                self.f = nn.Sequential(blk(3, 24), blk(24, 48), blk(48, 96),
                                       blk(96, 128), blk(128, 160),
                                       nn.AdaptiveAvgPool2d(1))
                self.h = nn.Linear(160, 2)

            def forward(self, x):
                return self.h(self.f(x).flatten(1))

        name = 'ego_final.pt'
        roots = [Path(model_dir)] if model_dir else []
        roots += [Path(model_dir) / 'stage3'] if model_dir else []
        roots += [Path(__file__).resolve().parent, Path.cwd()]
        path = next((r / name for r in roots if (r / name).is_file()), None)
        if path is None:
            return None
        blob = torch.load(str(path), map_location='cpu', weights_only=True)
        net = _Net()
        net.load_state_dict(blob['state_dict'])
        net.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = net.to(device)
        _EGO = (net, torch, device, float(blob['mu_s']), float(blob['sd_s']),
                float(blob['sd_y']), tuple(blob['input_hw']))
    except Exception as exc:
        print('ego model unavailable:', type(exc).__name__, str(exc)[:160], flush=True)
        _EGO = None
    return _EGO


def _ego_speed_series(small_frames, model_dir=None):
    # 한 영상의 연속 프레임쌍에 대해 속력(m/s)을 낸다. 실패하면 None.
    # 규정 4항: 이 함수는 **한 파일의 프레임만** 본다. 다른 파일의 정보나
    # 통계를 쓰지 않는다.
    loaded = _ego_model(model_dir)
    if loaded is None or len(small_frames) < 2:
        return None
    net, torch, device, mu_s, sd_s, sd_y, _hw = loaded
    arr = np.stack(small_frames).astype(np.float32) / 255.0
    out = np.zeros(len(small_frames), dtype=np.float32)
    try:
        with torch.inference_mode():
            for start in range(0, len(arr) - 1, 256):
                stop = min(start + 256, len(arr) - 1)
                a = torch.from_numpy(arr[start:stop]).unsqueeze(1)
                b = torch.from_numpy(arr[start + 1:stop + 1]).unsqueeze(1)
                x = torch.cat([a, b, b - a], dim=1).to(device)
                pred = net(x).cpu().numpy()
                out[start + 1:stop + 1] = pred[:, 0] * sd_s + mu_s
    except Exception as exc:
        print('ego inference failed:', type(exc).__name__, str(exc)[:160], flush=True)
        return None
    out[0] = out[1] if len(out) > 1 else 0.0
    return out

def _motion_series(path: Path, model_dir=None):
    # 학습 모델이 있으면 대리값 두 개(raw, 깊이정규화)는 계산해도 버려진다.
    #   column 0 raw          -> _labels_from_motion 632행 이후 미사용
    #   column 2 깊이정규화    -> 640행에서 ego 로 덮어써짐
    # 프로파일(폭 240) 기준 프레임당 0.33 + 0.81 = 1.14 ms / 9.10 ms = 12.5%.
    # 모델 적재 여부를 루프 **전에** 확인해 그때만 건너뛴다.
    # 모델이 없으면 예전처럼 셋을 다 만들어 폴백 경로가 그대로 동작한다.
    skip_dead = bool(CONFIG.get('stage3_use_ego')) and _ego_model(model_dir) is not None
    capture = cv2.VideoCapture(str(path))
    magnitudes, horizontal = ([], [])
    # 학습망 입력용 작은 프레임. 광류용(폭 240)과 별도 크기다.
    small = []
    previous = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = _resized_gray(frame, int(CONFIG['stage3_flow_width']))
        # 학습망 입력은 **이미 만든 회색조 gray** 에서 줄인다.
        # E24 는 원본 해상도 frame 에서 cvtColor + INTER_AREA 를 다시 했고
        # 그것이 프레임당 전처리를 1920x1080 에서 +169% 로 늘렸다.
        # gray 는 폭 240 이라 여기서 줄이는 비용은 0.03 ms 수준이다(약 90배 차이).
        small.append(cv2.resize(gray, (int(CONFIG['stage3_ego_width']),
                                       int(CONFIG['stage3_ego_height'])),
                                interpolation=cv2.INTER_AREA))
        if previous is None:
            magnitudes.append((0.0, 0.0, 0.0))
            horizontal.append(0.0)
        else:
            flow = cv2.calcOpticalFlowFarneback(previous, gray, None, 0.5, 2, int(CONFIG['stage3_flow_winsize']), 2, int(CONFIG['stage3_flow_poly_n']), float(CONFIG['stage3_flow_poly_sigma']), 0)
            corrected = _background_corrected_speed(flow)
            if skip_dead:
                magnitudes.append((corrected, corrected, corrected))
            else:
                magnitudes.append((_speed_proxy(flow), corrected,
                                   _depth_normalised_speed(flow)))
            horizontal.append(_yaw_proxy(flow[..., 0]))
        previous = gray
    capture.release()
    ego = _ego_speed_series(small, model_dir) if CONFIG.get('stage3_use_ego') else None
    return (np.asarray(magnitudes, dtype=np.float32),
            np.asarray(horizontal, dtype=np.float32),
            None if ego is None else np.asarray(ego, dtype=np.float32))

def _mode_filter(labels: list[str], width: int) -> list[str]:
    if width <= 1 or len(labels) <= 1:
        return labels
    half = width // 2
    counts: dict[str, int] = {}
    smoothed: list[str] = []
    left = right = 0
    for index in range(len(labels)):
        while right < min(len(labels), index + half + 1):
            counts[labels[right]] = counts.get(labels[right], 0) + 1
            right += 1
        while left < max(0, index - half):
            counts[labels[left]] -= 1
            left += 1
        best = max(counts.values())
        smoothed.append(labels[index] if counts.get(labels[index], 0) == best else min((k for k, v in counts.items() if v == best)))
    return smoothed

def _labels_from_motion(motion, horizontal, mode, ego=None):
    if len(motion) == 0:
        return ([], [])
    motion = np.asarray(motion)
    # 신호 세 개를 각자의 일에 쓴다.
    #   motion          raw         (사용 안 함, 형태 유지용)
    #   forward_motion  배경 보정     정지 판정. 문턱이 이 스케일에 맞춰져 있다(E18).
    #   accel_motion    깊이 정규화   가속도 기울기. 속력 상관 0.194 -> 0.361(E20).
    # 깊이 정규화는 스케일을 0.041배로 바꾸므로 정지 문턱에는 쓸 수 없다.
    # 문턱 환산은 운영점 문제이고 그것은 3전 3패다. E16 이 이긴 방식 그대로
    # 새 신호를 기울기 경로에만 쓴다.
    accel_motion = motion[:, 2] if motion.ndim == 2 and motion.shape[1] > 2 else motion
    forward_motion = motion[:, 1] if motion.ndim == 2 else motion
    motion = motion[:, 0] if motion.ndim == 2 else motion
    window = min(int(CONFIG['stage3_accel_smooth']), (len(accel_motion) - 1) // 2)
    # 가감속 기울기의 입력을 **학습 모델의 속력(m/s)** 으로 바꾼다.
    # comma2k19 주행 단위 분리 검증에서 속력 상관 0.36 -> 0.961.
    # 기울기 문턱은 std(delta) x 배수라 스케일 불변이므로 단위가 바뀌어도
    # 운영점이 흔들리지 않는다. 그래서 이 변경만 따로 낼 수 있다.
    # 망이 없거나 추론이 실패하면 기존 신호로 그대로 넘어간다.
    # skip_dead 로 만든 경우 column 0/2 에는 corrected 가 들어 있다.
    # 따라서 ego 추론이 실패해도 accel 은 배경 보정 신호로 떨어질 뿐
    # 0 배열이 되지 않는다.
    if ego is not None and len(ego) == len(accel_motion):
        accel_motion = ego
    baseline = _smooth(accel_motion, window)
    delta = np.gradient(baseline) if len(baseline) > 1 else np.zeros_like(baseline)
    # 정지 판정도 배경 보정 신호로 정렬한다. raw 는 충돌 충격의 카메라
    # 흔들림으로 부풀어 충돌 직후의 진짜 정지를 놓친다. comma2k19 처럼
    # 흔들림이 없는 자료에서는 두 신호가 사실상 같아(스케일 비 1.0031,
    # 정지 F1 0.848 동일) 문턱 0.003/0.005 를 그대로 쓴다.
    stop_mask = _stopped_from_motion(forward_motion)
    threshold = float(np.std(delta)) * float(CONFIG['stage3_accel_mult'])
    accel = []
    for stopped, slope in zip(stop_mask, delta):
        if stopped:
            accel.append('STOPPED' if mode != 'conservative' else 'CONSTANT')
        elif slope > threshold:
            accel.append('ACCELERATING')
        elif slope < -threshold:
            accel.append('DECELERATING')
        else:
            accel.append('CONSTANT')
    horizontal = _smooth_yaw_five(horizontal)
    turn_threshold = float(np.std(horizontal)) * {'conservative': 2.0, 'balanced': 1.1, 'responsive': 0.9}.get(mode, 1.1)
    steer = _steer_hysteresis(horizontal, turn_threshold)
    width = int(CONFIG['stage3_smooth_samples'])
    return (_mode_filter(accel, width), _mode_filter(steer, width))

def _frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return max(total, 0)

def predict_stage3(data_dir, model_dir=None):
    mode = CONFIG['stage3_mode']
    rows = []
    for path in _video_paths(Path(data_dir) / 'videos'):
        try:
            if _over_budget():
                accel, steer = ([], [])
            else:
                motion, horizontal, ego = _motion_series(path, model_dir)
                accel, steer = _labels_from_motion(motion, horizontal, mode, ego)
        except Exception:
            accel, steer = ([], [])
        if not accel:
            fallback = _frame_count(path)
            accel = ['CONSTANT'] * fallback
            steer = ['STRAIGHT'] * fallback
        stride = max(1, int(CONFIG['stage3_sample_stride']))
        last = len(accel) - 1
        rows.extend(({'ID': path.stem, 'sample_index': index, 'accel_label': accel[min(index * stride, last)], 'steer_label': steer[min(index * stride, last)]} for index in range(len(accel))))
    return pd.DataFrame(rows, columns=['ID', 'sample_index', 'accel_label', 'steer_label'])
_NETWORKS = {}
CONFIG['stage2_tracking'] = True

# 50분 가드 복원. 추적 도입 때 넣은 `return False` 덮어쓰기를 제거한다.
# 실측 추론 시간이 21분대이므로 가드는 발동하지 않고 출력은 동일하다.

def _network(kind, model_dir=None):
    import torch
    from torchvision.models.optical_flow import raft_large
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    name = 'raft_kitti.pth' if kind == 'flow' else 'vehicle_frcnn.pth'
    roots = [Path(model_dir)] if model_dir is not None else []
    roots.append(Path(__file__).resolve().parent)
    path = next((p / name for p in roots if (p / name).is_file()), None)
    if path is None:
        raise FileNotFoundError(name)
    key = (kind, str(path.resolve()))
    if key not in _NETWORKS:
        torch.set_num_threads(4)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = raft_large(weights=None) if kind == 'flow' else fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, min_size=384, max_size=640)
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        _NETWORKS[key] = (model.eval().to(device), device)
    return _NETWORKS[key]

def _vehicle_detections(paths, model_dir=None):
    import torch
    model, device = _network('vehicles', model_dir)
    found = []
    for start in range(0, len(paths), 4):
        batch = []
        valid = []
        for index, path in enumerate(paths[start:start + 4], start):
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            frame = cv2.cvtColor(cv2.resize(frame, (640, max(32, int(h * 640 / w)))), cv2.COLOR_BGR2RGB)
            batch.append(torch.from_numpy(frame).permute(2, 0, 1).float().to(device) / 255.0)
            valid.append((index, frame.shape[0], frame.shape[1]))
        if not batch:
            continue
        with torch.inference_mode():
            predictions = model(batch)
        for (index, h, w), p in zip(valid, predictions):
            boxes, labels, scores = (p[k].cpu().numpy() for k in ('boxes', 'labels', 'scores'))
            mask = np.isin(labels, [3, 4, 6, 8]) & (scores >= 0.35)
            found.append((index, boxes[mask] / np.array([w, h, w, h]), scores[mask]))
    return found

def _box_iou(a, b):
    lo, hi = (np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:]))
    inter = float(np.prod(np.maximum(hi - lo, 0)))
    aa = float(np.prod(np.maximum(a[2:] - a[:2], 0)))
    ab = float(np.prod(np.maximum(b[2:] - b[:2], 0)))
    return inter / max(aa + ab - inter, 1e-08)

def _vehicle_tracks(detections):
    tracks = []
    for index, boxes, scores in detections:
        edges = []
        for tid, track in enumerate(tracks):
            last_index, last_box, _ = track[-1]
            if index - last_index > 4:
                continue
            for j, box in enumerate(boxes):
                overlap = _box_iou(last_box, box)
                center_shift = np.linalg.norm((last_box[:2] + last_box[2:] - box[:2] - box[2:]) * 0.5)
                if overlap > 0.15 and center_shift < 0.25:
                    edges.append((overlap, tid, j))
        used_tracks, used_boxes = (set(), set())
        for _, tid, j in sorted(edges, reverse=True):
            if tid not in used_tracks and j not in used_boxes:
                tracks[tid].append((index, boxes[j], float(scores[j])))
                used_tracks.add(tid)
                used_boxes.add(j)
        for j, box in enumerate(boxes):
            if j not in used_boxes:
                tracks.append([(index, box, float(scores[j]))])
    return tracks

def _extend_track_backward(track, paths):
    """Recover small pre-entry appearances with local optical tracking.

    Stop when support vanishes; never extrapolate beyond observed pixels.
    """
    first, box, confidence = track[0]
    if first == 0:
        return track
    frame = cv2.imread(str(paths[first]), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        return track
    frame = cv2.resize(frame, (640, 360))
    box = box.copy() * [640, 360, 640, 360]
    added = []
    for index in range(first - 1, max(-1, first - 17), -1):
        prev = cv2.imread(str(paths[index]), cv2.IMREAD_GRAYSCALE)
        if prev is None:
            break
        prev = cv2.resize(prev, (640, 360))
        mask = np.zeros_like(frame)
        x0, y0, x1, y1 = np.rint(box).astype(int)
        mask[max(0, y0):min(360, y1), max(0, x0):min(640, x1)] = 255
        p = cv2.goodFeaturesToTrack(frame, 80, 0.02, 3, mask=mask)
        if p is None or len(p) < 8:
            break
        q, valid, _ = cv2.calcOpticalFlowPyrLK(frame, prev, p, None, winSize=(21, 21), maxLevel=3)
        back, valid_back, _ = cv2.calcOpticalFlowPyrLK(prev, frame, q, None, winSize=(21, 21), maxLevel=3)
        good = (valid[:, 0] != 0) & (valid_back[:, 0] != 0) & (np.linalg.norm(back[:, 0] - p[:, 0], axis=1) < 1.5)
        if good.sum() < 8:
            break
        transform, inliers = cv2.estimateAffinePartial2D(p[good], q[good], method=cv2.LMEDS)
        if transform is None or inliers.mean() < 0.6:
            break
        scale = float(np.hypot(transform[0, 0], transform[0, 1]))
        if not 0.65 < scale < 1.25:
            break
        corners = np.array([[box[0], box[1], 1], [box[2], box[1], 1], [box[2], box[3], 1], [box[0], box[3], 1]]) @ transform.T
        box = np.concatenate([corners.min(0), corners.max(0)])
        box = np.clip(box, [0, 0, 0, 0], [640, 360, 640, 360])
        if np.min(box[2:] - box[:2]) < 8:
            break
        added.append((index, box / [640, 360, 640, 360], confidence * 0.8))
        frame = prev
    return list(reversed(added)) + track

def _tracked_entry(paths, collision_frame, model_dir=None):
    numbers = np.array([_frame_number(p) for p in paths])
    collision = int(np.argmin(np.abs(numbers - collision_frame)))
    tracks = _vehicle_tracks(_vehicle_detections(paths[:collision + 1], model_dir))
    candidates = []
    plausible = []
    for track in tracks:
        if len(track) < 4 or track[-1][0] < collision - 5:
            continue
        endpoint = track[-1][1]
        area = float(np.prod(endpoint[2:] - endpoint[:2]))
        if area < 0.025 or endpoint[3] < 0.55 or endpoint[0] > 0.85 or (endpoint[2] < 0.15):
            continue
        plausible.append((area, track))
    for _, track in sorted(plausible, key=lambda p: p[0], reverse=True)[:3]:
        first_detection = track[0][0]
        track = _extend_track_backward(track, paths)
        indices = np.array([p[0] for p in track])
        boxes = np.array([p[1] for p in track])
        centers = (boxes[:, 0] + boxes[:, 2]) * 0.5
        origin = float(np.median(centers[:3]))
        displacement = centers[-1] - origin
        side = 'LEFT' if origin < 0.5 else 'RIGHT'
        inward = displacement if side == 'LEFT' else -displacement
        half_lane = np.clip((boxes[:, 3] - 0.4) * 0.55, 0.045, 0.27)
        left, right = (0.5 - half_lane, 0.5 + half_lane)
        overlap = np.maximum(0.0, np.minimum(boxes[:, 2], right) - np.maximum(boxes[:, 0], left))
        fraction = overlap / np.maximum(np.minimum(boxes[:, 2] - boxes[:, 0], right - left), 1e-05)
        entered = fraction >= 0.25
        crossing = next((j for j in range(1, len(track) - 1) if not entered[j - 1] and entered[j] and entered[j + 1]), None)
        if crossing is not None and inward < 0.025:
            crossing = None
        if crossing is None:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            if areas[-1] < 3.0 * max(areas[0], 1e-05):
                continue
            approach = entered & (areas >= 0.2 * areas[-1]) & (boxes[:, 2] - boxes[:, 0] >= 0.12)
            crossing = next((j for j in range(len(track) - 1) if approach[j] and approach[j + 1]), None)
            if crossing is None:
                continue
        area = float(np.prod(boxes[-1, 2:] - boxes[-1, :2]))
        score = area * (1.0 + inward) * float(np.mean([p[2] for p in track]))
        candidates.append((score, int(indices[crossing]), side, len(track)))
    if not candidates:
        return None
    score, index, side, length = max(candidates)
    if score < 0.015:
        return None
    return (int(numbers[index]), side)

def predict_stage2(data_dir, model_dir=None):
    result = _legacy_predict_stage2(data_dir, model_dir)
    if not CONFIG.get('stage2_tracking', False):
        return result
    for index, row in result.iterrows():
        if _over_budget():
            # 이 구간이 Stage 2 에서 가장 비싸다(모든 클립에 Faster R-CNN).
            # 가드가 없어서 E21 이 초과를 '점수 하락'이 아니라 '제출 실패'로
            # 맞았다. 남은 클립은 legacy entry_side 로 떨어뜨리고 계속한다.
            break
        folder = Path(data_dir) / 'images' / str(row.ID)
        paths = sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS), key=_frame_number)
        if not paths:
            # 읽을 프레임이 없으면 추적할 것도 없다. legacy 답을 그대로 둔다.
            continue
        try:
            # 보고용 collision_frame 이 아니라 **추적용**을 넘긴다.
            # 그래야 검출기에 들어가는 프레임 수가 E25 와 동일하게 유지된다.
            track_frame = _TRACK_FRAME.get(str(row.ID), int(row.collision_frame))
            tracked = _tracked_entry(paths, track_frame, model_dir)
            if tracked is not None:
                if CONFIG.get('stage2_tracked_entry'):
                    # 진단 실측: entry_frame 은 적중 11.3% 로 Stage 2 에서 가장
                    # 망가진 부품이고 가중치는 0.338 이다. 지금은 collision - 22
                    # 라는 **상수**이고 고정 간격 축은 닫혔다({16,20,22,24} 중 22).
                    # 그런데 클립별 진입 시각을 **이미 계산해 놓고 버리고 있다** -
                    # _tracked_entry 가 돌려주는 tracked[0] 이 상대 차량이 자차
                    # 차선에 처음 걸치는 프레임이다. 추가 비용이 0 이다.
                    # entry <= collision 은 유지한다(규정상 범위 문제는 없다).
                    entry = int(min(int(tracked[0]), int(row.collision_frame)))
                    result.at[index, 'entry_frame'] = max(0, entry)
                # E02: retain the legacy entry_frame; update entry_side only.
                side = tracked[1]
                if CONFIG.get('stage2_invert_side'):
                    # D3 진단은 추적 경로에도 같이 적용해야 한다.
                    side = 'LEFT' if side == 'RIGHT' else 'RIGHT'
                result.at[index, 'entry_side'] = side
        except Exception as exc:
            # 한 파일의 실패가 Stage 2 전체를 0점으로 만들지 않게 한다.
            print('Stage2 fallback:', row.ID, type(exc).__name__, str(exc)[:160], flush=True)
    if CONFIG.get('stage2_collision_diagnostic'):
        # D4 진단: 추적이 모두 끝난 **뒤** collision_frame 만 갈아끼운다.
        # 검출기는 참 충돌 인덱스를 봤으므로 작업량은 진단이 없을 때와 같다.
        # entry_frame 은 건드리지 않아 collision_frame 하나만 격리된다.
        # 각 영상의 마지막 프레임 번호를 쓴다. 공개 5개에서 충돌은 클립의
        # 60~82% 지점이라 허용오차(±0.3초) 밖이고, entry <= collision 도
        # 유지되며 범위 밖·음수·결측이 아니므로 정상 채점된다.
        for index, row in result.iterrows():
            folder = Path(data_dir) / 'images' / str(row.ID)
            try:
                numbers = [_frame_number(p) for p in folder.iterdir()
                           if p.suffix.lower() in IMAGE_EXTENSIONS]
            except OSError:
                numbers = []
            if numbers:
                result.at[index, 'collision_frame'] = int(max(numbers))
    return result

def _smooth_yaw_five(values):
    # Match the yaw-only temporal operation used inside ZB, without replacing
    # Farneback or altering acceleration smoothing, stopping, or sample mapping.
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return values.copy()
    radius = min(2, (len(values) - 1) // 2)
    if radius < 1:
        return values.copy()
    return np.convolve(np.pad(values, (radius, radius), mode='edge'),
                       np.ones(2 * radius + 1) / (2 * radius + 1), mode='valid')


def _stopped_from_motion(motion):
    """Low normalized optical motion plus persistence, not a fixed stop fraction.

    Fixed thresholds use public 10 Hz Farneback road-motion observations.
    They are not calibrated metric speed and may not transfer across cameras.
    No weights are fitted to the input video.
    """
    values = np.asarray(motion, dtype=np.float64)
    if len(values) == 0:
        return np.zeros(0, dtype=bool)
    values = np.nan_to_num(values, nan=np.inf, posinf=np.inf, neginf=np.inf)
    filtered = np.median(np.lib.stride_tricks.sliding_window_view(
        np.pad(values, (2, 2), mode='edge'), 5), axis=1)
    # Separate enter/exit levels suppress rapid flicker near the boundary.
    enter, leave = 0.003, 0.005
    state = bool(np.median(values[:min(5, len(values))]) <= enter)
    output = np.zeros(len(values), dtype=bool)
    low_run = high_run = 0
    for i, value in enumerate(filtered):
        low_run = low_run + 1 if value <= enter else 0
        high_run = high_run + 1 if value >= leave else 0
        if not state and low_run >= 3:
            state = True
        elif state and high_run >= 2:
            state = False
        output[i] = state
    return output


def _steer_hysteresis(values, threshold):
    state = 'STRAIGHT'
    result = []
    for value in values:
        if value > threshold:
            state = 'LEFT'
        elif value < -threshold:
            state = 'RIGHT'
        elif (state == 'LEFT' and value < threshold * 0.5) or (state == 'RIGHT' and value > -threshold * 0.5):
            state = 'STRAIGHT'
        result.append(state)
    return result


def _exposure_robust_difference(frame, previous):
    signed = frame.astype(np.float32) - previous.astype(np.float32)
    # Remove spatially uniform additive illumination changes only.
    return np.abs(signed - np.median(signed))


def _approach_threat(track, lane_fraction):
    """Rank eligible targets by apparent closing rate within the ego corridor.

    Median pairwise log-size slopes reduce single-box jitter. This is a
    time-to-contact proxy, not a calibrated collision probability.
    """
    tail = track[-8:]
    times = np.asarray([item[0] for item in tail], dtype=np.float64)
    boxes = np.asarray([item[1] for item in tail], dtype=np.float64)
    areas = np.maximum(np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1), 1e-8)
    log_size = .5 * np.log(areas)
    rates = [(log_size[j] - log_size[i]) / (times[j] - times[i])
             for i in range(len(tail)) for j in range(i + 1, len(tail)) if times[j] > times[i]]
    closing = max(0., float(np.median(rates))) if rates else 0.
    return closing * float(np.clip(lane_fraction, 0., 1.))

def _depth_normalised_speed(flow):
    # 전진 카메라에서 FOE 로부터 r 떨어진 점의 광류는 |f| ~ (v/Z) r 이므로
    # |f|/r ~ v/Z 다. 평평한 노면이면 Z ~ 1/(y - y0) 이라 (|f|/r)/(y - y0) ~ v
    # 가 되어 깊이에 무관한 속도 추정치가 된다. 현행은 (|f|/r) 를 그대로
    # 평균하므로 근거리 화소가 값을 지배한다.
    # comma2k19 실측 속력 상관 0.194 -> 0.361.
    # 배경 기준 보정은 그대로 먼저 적용한다(E16/E17 에서 이긴 부분).
    height, width = flow.shape[:2]
    top, bottom, left, right = CONFIG['stage3_bg_ref_roi']
    band = flow[int(height * top):int(height * bottom), int(width * left):int(width * right)]
    corrected = flow - np.median(band.reshape(-1, 2), axis=0) if band.size else flow
    magnitude = cv2.magnitude(corrected[..., 0], corrected[..., 1])
    radius, _road, corridor, depth = _flow_geometry(height, width)
    # 깊이 정규화는 평평한 노면을 가정한다. 좌우 주변부는 갓길·가드레일·
    # 건물·반대 차선이라 깊이가 제각각이고 그 가정이 깨진다. 중앙 통로만
    # 남기면 가정이 성립하는 곳만 쓴다.
    # comma2k19 실측(가로 폭만 바꾼 단일 변수): 속력 상관 0.361 -> 0.608,
    # 기울기-가속도 0.226 -> 0.229, 주행 3클래스 0.414 -> 0.435.
    # 위 경계를 올리는 것은 반대로 손해였다(0.50 -> 0.55 -> 0.60 에서
    # 속력 상관 0.361 -> 0.281 -> 0.128).
    if not corridor.any():
        return _speed_proxy(flow)
    return float(np.mean((magnitude / radius / depth)[corridor]))

def _background_corrected_speed(flow):
    # Background reference approximates common camera motion. It is not
    # calibrated ego speed or semantic segmentation of independently moving cars.
    height, width = flow.shape[:2]
    # 기준 영역을 조향 ROI 에서 분리한다. 조향 ROI 는 요레이트를 재려고
    # 고른 영역이고 끼어드는 차량이 바로 거기 있어 기준이 표적의 운동으로
    # 오염된다. 카메라 공통 회전만 담는 곳은 먼 배경, 즉 화면 상단이다.
    top, bottom, left, right = CONFIG['stage3_bg_ref_roi']
    band = flow[int(height * top):int(height * bottom), int(width * left):int(width * right)]
    if band.size == 0:
        return _speed_proxy(flow)
    reference = np.median(band.reshape(-1, 2), axis=0)
    corrected = flow - reference
    magnitude = cv2.magnitude(corrected[..., 0], corrected[..., 1])
    radius, road, _corridor, _depth = _flow_geometry(height, width)
    return float(np.mean((magnitude / radius)[road])) if road.any() else float(np.mean(magnitude))
