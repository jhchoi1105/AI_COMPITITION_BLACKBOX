# 챔피언(E42) 휴리스틱을 CCD 전 클립에 돌려 변화량 신호와 충돌 예측을 저장
import cv2, numpy as np, os, sys, shutil
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, 'C:/bbw/cm')
def work(name):
    import inf46 as M
    M.CONFIG['stage2_smooth_radius'] = 1
    d = Path('C:/bbw/cm/tmp/%s' % name); d.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture('C:/bbw/ccd/crash/%s.mp4' % name); i = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        cv2.imwrite(str(d / ('%06d.jpg' % i)), f, [cv2.IMWRITE_JPEG_QUALITY, 95]); i += 1
    _, ch = M._stage2_changes(d)
    row = M._stage2_one(d)
    shutil.rmtree(d)
    return ch.astype(np.float32), row['collision_frame'], row['entry_frame']
if __name__ == '__main__':
    z = np.load('C:/bbw/cm/ccd_crash.npz'); names = list(z['names'])
    with Pool(8) as p:
        res = p.map(work, names, chunksize=4)
    ch = np.stack([r[0] for r in res]); col = np.array([r[1] for r in res]); ent = np.array([r[2] for r in res])
    np.savez('C:/bbw/cm/heur.npz', ch=ch, col=col, ent=ent)
    on = z['onset']; ego = z['ego']
    hit = np.abs(col - on) <= 3
    print('heuristic hit all %.3f ego %.3f nonego %.3f' % (hit.mean(), hit[ego].mean(), hit[~ego].mean()))
