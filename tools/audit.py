"""제출물 최종 감사. ZIP에서 푼 사본만 검사한다 (실제 평가되는 것이 그것이므로)."""
import ast, importlib.util, io, json, os, re, shutil, subprocess, sys, time, zipfile
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(r"c:/Users/choij/OneDrive/Desktop/블랙박스 영상 기반 지능형 고의사고 분석 모델 AI 경진대회")

# 검사 대상. 2026-09-05 이전에는 submit_j_probe.zip 이 **하드코딩**돼 있었고
# sys.argv 를 전혀 읽지 않았다. 그 결과 J 이후 약 20회의 감사가 전부
# 2026-08-27 자 낡은 파일을 검사했고, "감사 62항목 실패 0건" 이라는 보고는
# 그 제출물에 대한 것이 아니었다. 인자를 받도록 고친다.
#
# 인자로 ZIP 경로 또는 변형 디렉터리를 받는다. 디렉터리면 임시 ZIP 으로 묶는다.
def _resolve_target() -> Path:
    if len(sys.argv) < 2:
        raise SystemExit("사용법: audit.py <submit_xxx.zip | submissions/variant_xxx>")
    t = Path(sys.argv[1])
    if not t.is_absolute():
        t = BASE / t
    if t.is_dir():
        tmp = Path("C:/bbw/audit_target.zip")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        if tmp.exists():
            tmp.unlink()
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as f:
            for n in ("inference.py", "requirements.txt"):
                src = t / n
                if not src.exists():
                    raise SystemExit(f"{src} 없음")
                f.write(src, n)
            # 동봉 가중치도 함께 묶는다 (실제 제출과 같은 구성으로 검사해야 한다)
            for extra in sorted(t.iterdir()):
                if extra.suffix in (".pt", ".pth", ".safetensors", ".onnx", ".npz"):
                    f.write(extra, extra.name)
        print(f"   대상 디렉터리 {t.name} 를 임시 ZIP 으로 묶어 검사한다")
        return tmp
    if not t.exists():
        raise SystemExit(f"{t} 없음")
    return t

ZIP = _resolve_target()
WORK = Path("C:/bbw/audit"); SM = Path("C:/bbv/smoke")
FAIL, WARN = [], []
def ck(cond, msg):
    print(("  [OK] " if cond else "  [실패] ") + msg)
    if not cond: FAIL.append(msg)
def warn(msg):
    print("  [주의] " + msg); WARN.append(msg)

print("=" * 78); print("1. ZIP 구조"); print("=" * 78)
z = zipfile.ZipFile(ZIP); names = z.namelist()
print("   내용:", names, f"| {ZIP.stat().st_size/1024:.1f} KB")
# 가중치 동봉 허용. 2026-09-06 부터 Stage 1 학습 모델을 싣는다.
# 대회 규정: 사전학습 가중치를 내려받지 말고 weights=None 으로 구조만 만든 뒤
# 제출한 체크포인트를 load_state_dict 로 불러와야 한다. zip 10GB / 해제 32GB.
ALLOWED_EXTRA = {".pt", ".pth", ".safetensors", ".onnx", ".npz"}
extra = [n for n in names if n not in ("inference.py", "requirements.txt")]
ck("inference.py" in names and "requirements.txt" in names, "필수 두 파일 존재")
ck(all(Path(n).suffix in ALLOWED_EXTRA for n in extra),
   f"추가 파일은 가중치만 (발견: {extra or 0})")
ck(not any("/" in n or "\\" in n for n in names), "최상위 추가 폴더 없음")
ck(all(not n.startswith(("__MACOSX", "."))) for n in names) if False else ck(
    not any(n.startswith("__MACOSX") or n.startswith(".") for n in names), "숨김/맥 메타파일 없음")
ck(ZIP.stat().st_size < 10 * 1024**3, "ZIP 10GB 이하")
for zi in z.infolist():
    ck(zi.file_size < 32 * 1024**3, f"{zi.filename} 해제 후 크기 정상")

if WORK.exists(): shutil.rmtree(WORK)
WORK.mkdir(parents=True); z.extractall(WORK)
SRC = (WORK / "inference.py").read_text(encoding="utf-8")

print("\n" + "=" * 78); print("2. 코드 정적 검사"); print("=" * 78)
tree = ast.parse(SRC)
top = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
for fn in ("predict_stage1", "predict_stage2", "predict_stage3"):
    ck(fn in top, f"{fn} 이 모듈 최상위에 존재")
    if fn in top:
        args = [a.arg for a in top[fn].args.args]
        ck(args[:2] == ["data_dir", "model_dir"], f"{fn} 인자 이름·순서 (data_dir, model_dir)")

imports = set()
for n in ast.walk(tree):
    if isinstance(n, ast.Import): imports |= {a.name.split(".")[0] for a in n.names}
    elif isinstance(n, ast.ImportFrom) and n.module: imports.add(n.module.split(".")[0])
print("   임포트:", sorted(imports))
# 평가 서버 기본 설치 패키지만 허용한다. torch 2.8.0+cu128 · torchvision
# 0.23.0+cu128 은 서버에 이미 있고 requirements.txt 에 넣지 않는다(대회 권고).
# torch 를 허용하는 것은 다운로드 허용이 아니다 - 아래 네트워크 검사가 별도로 막는다.
ALLOWED = {"__future__", "re", "time", "pathlib", "cv2", "numpy", "pandas",
           "torch", "torchvision"}
ck(imports <= ALLOWED, f"허용 패키지만 임포트 (초과분: {sorted(imports-ALLOWED)})")

def _code_only(src):
    """주석과 문서 문자열을 뺀 코드만 남긴다.

    설명문에 'torch' 같은 단어가 있으면 오탐이 난다. 실제로 CC 에서
    'torch 의존성을 치를 값이 아니다'라는 설명이 실패로 잡혔다.
    임포트 검사는 이미 AST 로 정확히 하므로 텍스트 검사는 코드에만 건다.

    2026-09-05 재수정. 이전 방식은 (1) 줄단위로 '#' 뒤를 자른 뒤
    (2) 원본 문서문자열 텍스트를 replace 로 지웠다. 그런데 문서문자열이
    마크다운 헤더('## 제목')를 쓰면 (1)이 그 텍스트를 바꿔 버려 (2)가 더 이상
    매칭되지 않고 문서문자열이 통째로 남았다. NN 에서 설명문의
    'torch 2.8.0+cu128 기본 설치' 가 네트워크 흔적으로 오탐됐다.

    이제 AST 에서 문서문자열 노드를 **제거한 뒤 재출력**한다. 주석은 ast.unparse
    가 자동으로 버린다.
    """
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
                if not node.body:
                    node.body.append(ast.Pass())
    return ast.unparse(tree)


CODE = _code_only(SRC)
# torch 자체는 금지가 아니다. 평가 서버 기본 설치(2.8.0+cu128)이고 우리는
# 가중치를 동봉해 load_state_dict 로만 불러온다. 금지되는 것은 **다운로드**다.
NET = re.findall(r"\b(urllib|requests|socket|urlopen|download|hub\.load|from_pretrained|pretrained\s*=\s*True)\b", CODE)
ck(not NET, f"네트워크/가중치 다운로드 흔적 없음 (코드 부분만 검사, 발견: {set(NET)})")
ck("open(" not in SRC.replace("cv2.VideoCapture", ""), "외부 파일 쓰기/읽기 없음") if False else None
ck(not re.search(r"\bopen\s*\(.*['\"][wa]", SRC), "파일 쓰기 없음 (data/ 는 읽기전용)")
ck("model_dir" in SRC, "model_dir 인자 수용 (사용은 안 하지만 시그니처 유지)")

req = (WORK / "requirements.txt").read_text(encoding="utf-8").strip().splitlines()
print("   requirements:", req)
PRE = {"numpy": "1.26.4", "pandas": "2.2.2", "opencv-python-headless": "4.10.0.84"}
for line in req:
    name, _, ver = line.partition("==")
    ck(name in PRE and ver == PRE[name], f"{line} 는 평가서버 사전설치 버전과 일치")

print("\n" + "=" * 78); print("3. 정상 입력 — 스키마·값 유효성"); print("=" * 78)
spec = importlib.util.spec_from_file_location("subm", WORK / "inference.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)

t0 = time.time(); s1 = M.predict_stage1(SM / "stage1", WORK / "model/stage1"); t1 = time.time() - t0
t0 = time.time(); s2 = M.predict_stage2(SM / "stage2", WORK / "model/stage2"); t2 = time.time() - t0
t0 = time.time(); s3 = M.predict_stage3(SM / "stage3", WORK / "model/stage3"); t3 = time.time() - t0

SCHEMA = {
    "stage1": (s1, ["ID", "answer"]),
    "stage2": (s2, ["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"]),
    "stage3": (s3, ["ID", "sample_index", "accel_label", "steer_label"]),
}
for k, (df, cols) in SCHEMA.items():
    ck(list(df.columns) == cols, f"{k} 컬럼 이름·순서 정확")
    ck(isinstance(df, pd.DataFrame), f"{k} 반환형이 DataFrame")
    ck(not df.isnull().values.any(), f"{k} 결측치 없음")
    ck(len(df) > 0, f"{k} 행 존재 ({len(df)}행)")

ck(set(s1.answer) <= {"ORIGINAL", "RERECORDED"}, "stage1 answer 허용 범주만")
ck(s1.ID.is_unique, "stage1 ID 중복 없음")

ck(pd.api.types.is_integer_dtype(s2.collision_frame), "stage2 collision_frame 정수형")
ck(pd.api.types.is_integer_dtype(s2.entry_frame), "stage2 entry_frame 정수형")
ck((s2.collision_frame >= 0).all() and (s2.entry_frame >= 0).all(), "stage2 음수 프레임 없음")
ck(set(s2.evasion_space) <= {0, 1}, "stage2 evasion_space 0/1")
ck(set(s2.entry_side) <= {"LEFT", "RIGHT"}, "stage2 entry_side 허용 범주만")
ck(s2.ID.is_unique, "stage2 ID 중복 없음")
for _, r in s2.iterrows():
    n = len(list((SM / "stage2/images" / r.ID).glob("*.jpg")))
    ck(r.collision_frame <= n - 1 and r.entry_frame <= n - 1, f"stage2 {r.ID} 프레임이 영상 범위 내 (0..{n-1})")

ck(set(s3.accel_label) <= set(M.ACCEL_LABELS), "stage3 accel_label 허용 범주만")
ck(set(s3.steer_label) <= set(M.STEER_LABELS), "stage3 steer_label 허용 범주만")
ck(pd.api.types.is_integer_dtype(s3.sample_index), "stage3 sample_index 정수형")
ok_seq = all(list(g.sample_index) == list(range(len(g))) for _, g in s3.groupby("ID"))
ck(ok_seq, "stage3 sample_index 가 영상별 0..N-1 연속·빠짐없음")
ck(not s3.duplicated(["ID", "sample_index"]).any(), "stage3 (ID, sample_index) 중복 없음")

print("\n" + "=" * 78); print("4. CSV 저장·인코딩"); print("=" * 78)
for k, (df, _) in SCHEMA.items():
    p = WORK / f"{k}.csv"
    df.to_csv(p, index=False, encoding="utf-8")
    back = pd.read_csv(p, encoding="utf-8")
    ck(len(back) == len(df) and list(back.columns) == list(df.columns), f"{k} UTF-8 왕복 무손실")
ck(all(ord(c) < 128 for c in "".join(map(str, s1.ID)) + "".join(s1.answer)), "출력값이 ASCII (인코딩 사고 여지 없음)")

print("\n" + "=" * 78); print("5. 예외 입력"); print("=" * 78)
E = Path("C:/bbw/edge2")
if E.exists(): shutil.rmtree(E)
for d in ("s_empty/stage1/videos", "s_empty/stage2/images", "s_empty/stage3/videos"):
    (E / d).mkdir(parents=True)
r = [M.predict_stage1(E/"s_empty/stage1", None), M.predict_stage2(E/"s_empty/stage2", None), M.predict_stage3(E/"s_empty/stage3", None)]
ck(all(len(x) == 0 for x in r), "빈 폴더 → 빈 DataFrame, 예외 없음")
ck(all(list(x.columns) == c for x, c in zip(r, [v[1] for v in SCHEMA.values()])), "빈 결과도 컬럼 스키마 유지")
r = [M.predict_stage1(E/"nope", None), M.predict_stage2(E/"nope", None), M.predict_stage3(E/"nope", None)]
ck(all(len(x) == 0 for x in r), "존재하지 않는 경로 → 예외 없음")

(E/"bad/stage1/videos").mkdir(parents=True); (E/"bad/stage3/videos").mkdir(parents=True)
(E/"bad/stage2/images/EMPTY").mkdir(parents=True); (E/"bad/stage2/images/BAD").mkdir(parents=True)
(E/"bad/stage1/videos/broken.mp4").write_bytes(b"not a video")
(E/"bad/stage1/videos/zero.MP4").write_bytes(b"")
(E/"bad/stage1/videos/readme.txt").write_bytes(b"ignore me")
(E/"bad/stage3/videos/broken.mkv").write_bytes(b"nope")
(E/"bad/stage2/images/BAD/frame_000000.jpg").write_bytes(b"nope")
b1 = M.predict_stage1(E/"bad/stage1", None); b2 = M.predict_stage2(E/"bad/stage2", None); b3 = M.predict_stage3(E/"bad/stage3", None)
ck(len(b1) == 2, f"stage1 손상·빈 영상도 ID 유지, 비디오 아닌 파일은 무시 ({list(b1.ID)})")
ck(set(b1.answer) <= {"ORIGINAL", "RERECORDED"}, "stage1 손상 시에도 유효 범주")
ck(len(b2) == 2 and set(b2.ID) == {"BAD", "EMPTY"}, "stage2 빈/손상 폴더도 ID 유지")
ck((b2.entry_frame >= 0).all() and set(b2.entry_side) <= {"LEFT", "RIGHT"}, "stage2 손상 시에도 유효 값")

import cv2
short = E/"short/stage3/videos"; short.mkdir(parents=True)
for n, name in ((1, "one.mp4"), (2, "two.mp4"), (3, "three.mp4")):
    w = cv2.VideoWriter(str(short/name), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(n): w.write(np.full((48, 64, 3), i * 40, np.uint8))
    w.release()
s = M.predict_stage3(short.parent, None)
cnt = {k: len(g) for k, g in s.groupby("ID")}
print("   짧은 영상 행수:", cnt)
ck(len(s) > 0 and set(s.accel_label) <= set(M.ACCEL_LABELS), "1~3프레임 영상도 예외 없이 유효 출력")
ck(all(list(g.sample_index) == list(range(len(g))) for _, g in s.groupby("ID")), "짧은 영상도 sample_index 연속")
s2s = M.predict_stage2(E/"bad/stage2", None)

nested = E/"nest/stage1/videos/sub/deeper"; nested.mkdir(parents=True)
shutil.copy2(SM/"stage1/videos/S1_O_001.mp4", nested/"X.mp4")
ck(len(M.predict_stage1(E/"nest/stage1", None)) == 1, "하위 폴더 안의 영상도 탐색됨")

print("\n" + "=" * 78); print("6. 결정성 (같은 입력 → 같은 출력)"); print("=" * 78)
a = M.predict_stage2(SM/"stage2", None); b = M.predict_stage2(SM/"stage2", None)
ck(a.equals(b), "같은 프로세스 내 반복 호출 일치")
code = ("import importlib.util,sys;s=importlib.util.spec_from_file_location('m',r'%s');"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "d=m.predict_stage3(r'%s',None);print(len(d), ''.join(d.accel_label)[:200], ''.join(d.steer_label)[:200])"
        % (WORK/'inference.py', SM/'stage3'))
outs = []
for i in range(2):
    env = dict(os.environ, PYTHONHASHSEED=str(i * 7919), PYTHONIOENCODING="utf-8")
    outs.append(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env).stdout)
ck(outs[0] == outs[1] and outs[0].strip() != "", "PYTHONHASHSEED 다른 별도 프로세스 2회 결과 동일")

print("\n" + "=" * 78); print("7. 파일 단위 독립성 (대회 규정 4항)"); print("=" * 78)
one = E/"iso/stage3/videos"; one.mkdir(parents=True)
shutil.copy2(SM/"stage3/videos/OPEN_003.mp4", one/"OPEN_003.mp4")
solo = M.predict_stage3(one.parent, None)
group = s3[s3.ID == "OPEN_003"].reset_index(drop=True)
ck(list(solo.accel_label) == list(group.accel_label) and list(solo.steer_label) == list(group.steer_label),
   "영상 1개만 넣었을 때와 5개 중 하나일 때 예측이 동일 (다른 파일 정보 미사용)")
one2 = E/"iso1/stage1/videos"; one2.mkdir(parents=True)
shutil.copy2(SM/"stage1/videos/S1_R_003.mp4", one2/"S1_R_003.mp4")
ck(M.predict_stage1(one2.parent, None).answer.iloc[0] == s1[s1.ID == "S1_R_003"].answer.iloc[0],
   "stage1 도 단독 실행과 일괄 실행 결과 동일")

print("\n" + "=" * 78); print("8. 실행시간·자원"); print("=" * 78)
print(f"   stage1 {t1:5.1f}s / stage2 {t2:5.1f}s / stage3 {t3:5.1f}s  합계 {t1+t2+t3:5.1f}s (로컬 CPU)")
print(f"   stage3 영상당 약 {t3/5:.1f}s (1200프레임 기준)")
ck(30*60 <= M._BUDGET_SECONDS <= 55*60 and not M._over_budget(),
   f"예산 방어 문턱 {M._BUDGET_SECONDS/60:.0f}분 (30~55분 범위), 평상시 미발동")
M._STARTED = M.time.monotonic() - 99999
o1, o2, o3 = M.predict_stage1(SM/"stage1", None), M.predict_stage2(SM/"stage2", None), M.predict_stage3(SM/"stage3", None)
ck(len(o1) == len(s1) and len(o2) == len(s2) and len(o3) == len(s3), "예산 초과 시에도 행 수 동일 (누락 없음)")
ck(set(o3.accel_label) <= set(M.ACCEL_LABELS) and set(o3.steer_label) <= set(M.STEER_LABELS), "예산 초과 시에도 유효 범주")
ck(all(list(g.sample_index) == list(range(len(g))) for _, g in o3.groupby("ID")), "예산 초과 시에도 sample_index 연속")
M._STARTED = M.time.monotonic()
try:
    import psutil, gc
    gc.collect(); print(f"   프로세스 RSS 약 {psutil.Process().memory_info().rss/1024**2:.0f} MB (서버 RAM 60GB)")
except Exception: pass

print("\n" + "=" * 78)
print(f"결과: 실패 {len(FAIL)}건 / 주의 {len(WARN)}건")
for m in FAIL: print("  [실패]", m)
for m in WARN: print("  [주의]", m)
print("=" * 78)
