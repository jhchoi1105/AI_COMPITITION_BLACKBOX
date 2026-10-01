# CCD crash 영상 -> 추론과 같은 경로의 160x90 회색조 (jpg 왕복 -> 480x270 -> 160x90)
import cv2, numpy as np, ast
from multiprocessing import Pool
R = 'C:/bbw/ccd/'
def load(name):
    cap = cv2.VideoCapture(R + 'crash/%s.mp4' % name)
    out = []
    while True:
        ok, f = cap.read()
        if not ok: break
        ok, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 95])
        g = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        g = cv2.resize(g, (480, 270), interpolation=cv2.INTER_AREA)
        out.append(cv2.resize(g, (160, 90), interpolation=cv2.INTER_AREA))
    a = np.zeros((50, 90, 160), np.uint8)
    n = min(50, len(out))
    if n: a[:n] = np.stack(out[:n])
    return a, n
if __name__ == '__main__':
    names, onset, ego = [], [], []
    for l in open(R + 'Crash-1500.txt'):
        p = l.split(',[')
        lab = ast.literal_eval('[' + p[1].split(']')[0] + ']')
        names.append(p[0]); onset.append(lab.index(1)); ego.append(l.strip().endswith('Yes'))
    with Pool(8) as pool:
        res = pool.map(load, names, chunksize=8)
    X = np.stack([r[0] for r in res]); lens = np.array([r[1] for r in res])
    np.savez('C:/bbw/cm/ccd_crash.npz', X=X, onset=np.array(onset), ego=np.array(ego), lens=lens, names=np.array(names))
    print(X.shape, np.bincount(lens)[-3:], sum(ego))
