# -*- coding: utf-8 -*-
"""E57 - 기준 E53 (단일 파일 최고). 한 번도 리더보드에 묻지 않은 축만.

Stage 1  챔피언 (시간 특징·색 특징을 측정했으나 쌍 차이가 영상 간 편차보다 훨씬 작아 기각).
Stage 2  판별기 경로 충돌 편향 0 -> -2.
         판별기는 CCD 라벨 규약(사고 프레임 시작)을 배웠다. 평가 규약은 '실제 접촉 첫 프레임'.
         휴리스틱 시절 리더보드 정점이 편향 -2 였고, 판별기 경로 편향은 한 번도 묻지 않았다.
         entry 는 편향 전 인덱스로 계산되므로 불변 -> collision 만 격리. 비용 0.
         --side: entry_side 를 학습 분류기로 덮어쓴다(CrashX 라벨, 라이선스 미표기).
Stage 3  가감속 속력 평활 창 45 -> 25.
         45 는 최빈값 필터가 1~3 이던 시절에 정한 값이다. 지금 필터는 41(4.1초)이라
         라벨 단계에서 이미 강하게 평활된다. 속력 단계 평활을 줄여 짧은 가감속을 살린다.
         호출 수를 챔피언과 맞추려고 민감도 0.45 -> 0.38 (같은 비율에서 시점 정확도만 비교). 비용 0.
"""
import zipfile, hashlib, sys, re
from pathlib import Path
SIDE = '--side' in sys.argv
BASE = Path(r"c:/Users/choij/OneDrive/Desktop/블랙박스 영상 기반 지능형 고의사고 분석 모델 AI 경진대회")
SRC_ZIP = BASE / "submit_e53_triple.zip"
OUT_ZIP = BASE / ("submit_e57s_side.zip" if SIDE else "submit_e57_triple.zip")
if OUT_ZIP.exists(): raise SystemExit("이미 존재")
Q = chr(39)
zin = zipfile.ZipFile(SRC_ZIP); src = zin.read("inference.py").decode("utf-8")
def repl(s, a, b, why):
    assert s.count(a) == 1, "패치 실패: " + why
    return s.replace(a, b, 1)
src = repl(src, Q+"stage2_flow_mix"+Q+": 0.25", Q+"stage2_net_bias"+Q+": -2, "+Q+"stage2_flow_mix"+Q+": 0.25", "편향 설정")
src = repl(src, "    bias = 0 if net_scores is not None else _scaled('stage2_collision_bias', len(paths))\n",
               "    bias = int(CONFIG.get('stage2_net_bias', 0)) if net_scores is not None else _scaled('stage2_collision_bias', len(paths))\n", "판별기 편향")
src = repl(src, Q+"stage3_accel_smooth"+Q+": 45", Q+"stage3_accel_smooth"+Q+": 25", "평활 창 25")
# 창을 줄이면 기울기 잡음이 늘어 상대 문턱(std x 배수)이 올라간다. 0.38 이면 ACCEL/DECEL 호출 수가 챔피언과 같다(공개 영상 937/998 vs 940/924).
src = repl(src, Q+"stage3_accel_mult"+Q+": 0.45", Q+"stage3_accel_mult"+Q+": 0.38", "민감도 0.38")
extra = {}
if SIDE:
    b51 = open("C:/bbw/build_e51.py", encoding="utf-8").read()
    SIDECODE = re.search(r"SIDECODE = '''(.*?)'''", b51, re.S).group(1)
    src = repl(src, Q+"stage2_flow_mix"+Q+": 0.25", Q+"stage2_side_net"+Q+": True, "+Q+"stage2_flow_mix"+Q+": 0.25", "플래그")
    src = repl(src, "def _stage2_changes(folder: Path)", SIDECODE + "def _stage2_changes(folder: Path)", "분류기 코드")
    src = repl(src,
        "    net_scores = _collision_scores(_TINY.pop(folder.name, None)) if CONFIG.get('stage2_collision_net') else None\n",
        "    _tinies = _TINY.pop(folder.name, None)\n"
        "    net_scores = _collision_scores(_tinies) if CONFIG.get('stage2_collision_net') else None\n", "tiny 보관")
    src = repl(src,
        "    collision_frame = int(min(max(numbers[collision_index], low), high))\n",
        "    collision_frame = int(min(max(numbers[collision_index], low), high))\n"
        "    if CONFIG.get('stage2_side_net'):\n"
        "        _s = _side_predict(_tinies, collision_index - bias)\n"
        "        if _s is not None:\n"
        "            _SIDE[folder.name] = _s\n", "방향 예측(편향 전 인덱스)")
    src = repl(src,
        "    if CONFIG.get('stage2_collision_diagnostic'):\n        # D4",
        "    if CONFIG.get('stage2_side_net'):\n"
        "        for index, row in result.iterrows():\n"
        "            s = _SIDE.get(str(row.ID))\n"
        "            if s is not None:\n"
        "                result.at[index, 'entry_side'] = s\n"
        "    if CONFIG.get('stage2_collision_diagnostic'):\n        # D4", "덮어쓰기")
    extra = {"side_net_%d.pt" % s: "C:/bbw/cm/side_full_s%d.pt" % s for s in range(3)}
compile(src, "inference.py", "exec")
with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zout:
    for info in zin.infolist():
        zout.writestr(info.filename, src.encode("utf-8") if info.filename == "inference.py" else zin.read(info.filename))
    for k, v in extra.items():
        zout.writestr(k, Path(v).read_bytes())
print("생성", OUT_ZIP.name, hashlib.sha256(OUT_ZIP.read_bytes()).hexdigest())
