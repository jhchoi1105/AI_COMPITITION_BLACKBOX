# 전체 Stage 개선 방안 조사 (논문 · GitHub · Hugging Face)

작성: 2026-09-26 (E50 반영) · 단계별 최고 **0.8954991087 / 0.3564263788 / 0.6558223130** (가중합 0.5840)
남은 제출: 09-27 3회, 09-28 3회, 09-29 10:00 전 → 약 9회

---

## 0. 새로 확인한 사실

| 사실 | 근거 |
|---|---|
| 공개 **Stage 1 원본 5개 = CCD crash 000001~000005 와 화소 동일** | 10번째 프레임 차이 0.0 (다른 클립과는 25~34) |
| 공개 **Stage 1 재녹화 5개 = 같은 CCD 클립을 화면에서 다시 찍은 것** | 같은 클립과 차이 4.2~4.9 (다른 클립과는 26~34), 형식은 50프레임·10fps·1280×720 로 맞춰져 있음 |
| 공개 Stage 2 5개 = CCD crash 000001~000005 | (11번 문서) |
| DACON: CCD 등 공개 데이터 학습·검증 허용, 평가 Stage 2 는 전부 자차 관여 충돌, Stage 3 라벨은 CAN 차속·종가속도 기반 | 공식 답변 |

---

## 1. Stage 1 — 재녹화 판별 (가중치 0.2, 남은 여유 0.021)

### 자료

| 자원 | 내용 |
|---|---|
| [Recaptured Video Detection Using Multiple Feature Descriptors](https://www.researchgate.net/publication/316658060_Recaptured_Video_Detection_Using_Multiple_Feature_Descriptors) | 블러·노이즈·색 모멘트·질감 조합 (지금 우리 방식과 같은 계열) |
| [Video recapture detection based on ghosting artifact](https://www.commsp.ee.ic.ac.uk/~pld/publications/2013_ICIp_Tagliasacchi_2.pdf) | 화면 재생률과 카메라 셔터가 어긋나 생기는 **고스팅(이중상)** 을 시간 방향으로 검출 |
| [Modeling Temporal Effects in Re-captured Video](https://www.researchgate.net/publication/310829307_Modeling_Temporal_Effects_in_Re-captured_Video) | 재녹화 시 **프레임 누락·반복**이 생긴다 |
| [Recaptured video detection based on sensor pattern noise](https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-015-0096-z) | 카메라 센서 잡음 패턴 |
| [Domain Generalized Recaptured Screen Image Identification (SWIN)](https://arxiv.org/html/2407.17170v1), [ViT 기반](https://www.sciencedirect.com/science/article/abs/pii/S1047320322002127) | 학습 기반 판별 (정확도 0.97~0.99, 대신 학습 데이터 필요) |
| [NTU-ROSE recaptured images](http://rose1.ntu.edu.tw/datasets/recapturedImages.asp), [RDNet](https://github.com/tju-chengyijia/RDNet), [VD_raw (NeurIPS 2023)](https://github.com/tju-chengyijia/VD_raw) | 재촬영 이미지·영상 데이터셋 (모아레 쌍) |

### 방안

1. **CCD 원본 4,500개로 오탐률 측정·보정 (합법, 즉시 가능).**
   원본 클래스 표본이 5개 → 4,500개로 늘어난다. 지금 특징 5개의 문턱을 "CCD 원본 오탐 x%" 기준으로 다시 맞출 수 있다.
2. **시간 방향 특징 추가:** 프레임 반복·고스팅. 지금 특징 5개는 전부 한 프레임짜리 공간 특징이다. 재녹화는 화면 재생률과 카메라 촬영률이 어긋나 **연속 프레임 간 차이에 주기적 이상**이 생긴다. 비용은 거의 0이다(이미 표집하는 프레임).
3. **합성 재녹화로 소형 CNN 학습:** CCD 원본에 모아레·감마·블러·색 이동·리샘플을 입혀 양성을 만든다. 공개 재녹화 5개는 검증에만 쓴다. 효과는 크지만 시간이 걸린다.

> ⚠ **하지 말아야 할 것:** 평가 클립을 CCD 원본과 **대조(해시·화소 비교)** 해서 "같으면 원본"으로 판정하는 것.
> 학습이 아니라 **공개 원본을 정답표로 쓰는 누수 이용**이라 규정 취지에 어긋난다고 본다. 쓰려면 DACON 확인이 먼저다.
> Stage 2 에서 CCD 라벨을 대조해 쓰는 것도 같은 이유로 제외한다.

---

## 2. Stage 2 — 사고 주요시점 (가중치 0.4)

### 지금까지

- CCD 학습 충돌 판별기 → **0.3549** (+0.0190), 3개 앙상블 → **0.3564** (+0.0015)
- 진입 간격: 22 = 24 (완전 동률), **36 은 −0.0179** → 상수 간격 축은 끝났다

### 자료

| 자원 | 쓸모 | 확보 |
|---|---|---|
| [CrashX](https://github.com/harshalDharpure/Crash-X) — CCD 1,500클립 주석 | **충돌 위치**("front left corner of camera car" 등), 사고 구간(1초 단위), 설명문 | 받음 (`C:/bbw/ccd/crashx_gt.xlsx`). 자차 관여 801개 중 **왼쪽 184 / 오른쪽 156 / 정면 245**. ⚠ 저장소에 LICENSE 파일이 없다 |
| [MM-AU (CVPR 2024)](https://github.com/jeffreychou777/LOTVS-MM-AU) | 11,727개. **사고 시작 · 충돌 시작 시각을 따로** 주석 → entry 에 가장 가까운 라벨 | 학술용, 이메일 신청 → 마감 전 확보 어려움 |
| [BADAS-Open (Nexar)](https://huggingface.co/nexar-ai/BADAS-Open), [GitHub](https://github.com/getnexar/BADAS-Open), [논문](https://arxiv.org/abs/2510.14876) | 자차 중심 충돌 예측, V-JEPA2 백본, HF 가중치 공개 | 무겁다(ViT-L급). 60분 예산에 부담 |
| [DoTA](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly) | `anomaly_start` = 사고가 불가피해진 순간 | YouTube 원본 필요 |
| [VZCrash](https://arxiv.org/html/2606.06074) | 자차 충돌 IMU 데이터 | 영상 아님 |

### 방안

1. **entry_side 학습 분류기 (CrashX 충돌 위치).**
   카메라 차량의 왼쪽이 받혔으면 LEFT, 오른쪽이면 RIGHT (약 340클립).
   충돌 직전 구간의 160×90 프레임으로 소형 CNN 을 학습한다. 지금 적중 62% · 가중치 0.162.
   ⚠ CrashX 주석에 라이선스 표기가 없다. CCD(MIT)에서 파생됐지만 **사용 전 확인이 필요하다.**
2. **판별기 앙상블 확대 (3 → 5).** 학습 진행 중이다. CCD 검증 단일 0.70 → 3개 0.72.
3. **entry_frame = 판별기 점수가 오르기 시작하는 프레임.** 라벨이 없어 리더보드로만 판정한다. 상수 간격이 끝났으니 남은 유일한 축이다.
4. **탐색 창 시작 0.15 → 0.05.** 판별기는 가짜 봉우리에 강하다. 평가 클립은 충돌이 앞쪽에 있다(E32). 비용 0.

---

## 3. Stage 3 — 차량 거동 (가중치 0.4)

### 지금까지

- 최빈값 필터 1 → 3 → 5 → 9 → 15 → 25 → 41 **일곱 연승**. 이득이 줄고 있다(+0.0067 → +0.0017).
- 라벨 = CAN 차속·종가속도 기반(DACON). 임계값은 비공개다.

### 자료

| 자원 | 내용 |
|---|---|
| [Estimation of Kinematic Motion from Dashcam Footage (2025)](https://arxiv.org/html/2512.01104) | 블랙박스 + CAN 18시간. **GRU 시간 모델이 회전·속도에 유리**, 좌우 반전 증강이 가장 효과적 |
| [BDD100K](https://github.com/bdd100k/bdd100k) | 10만 개 블랙박스 영상 + GPS 속도(가속도계는 절반이 부호 오류라 쓰기 어려움) |
| [comma2k19](https://github.com/commaai/comma2k19) (이미 사용) | CAN 속도 100Hz |
| 가감속 문턱 문헌 ([운전자 가속 통계](https://arxiv.org/pdf/1907.01747), [CBANet](https://arxiv.org/pdf/2605.23471)) | 블랙박스 영상 제동·가속 이벤트 문턱 **0.4 m/s²**, 감속 판정 −0.5 m/s², 90% 의 가속은 1.0 m/s² 이하 |

### 방안

1. **최빈값 필터 41 → 61/81** (끝까지 밀어 정점 확인). 비용 0.
2. **종가속도 직접 분류.** comma2k19 CAN 속도를 미분해 ±0.4~0.5 m/s² 로 ACCEL/DECEL/CONSTANT 라벨을 만들고, 4클래스를 직접 학습한다. 지금은 속력 회귀를 미분한다. DACON 라벨도 CAN 종가속도 기반이라 목표가 같아진다.
3. **시간 모델(GRU):** 자기운동 CNN 특징 위에 GRU 를 얹는다. 논문에서 회전·속도 모두 개선됐다.

---

## 4. 추천 순서 (남은 약 9회)

| 우선 | 방안 | 비용 | 기대 |
|---|---|---|---|
| 1 | Stage 2 앙상블 5개 + 탐색 시작 0.05 | 0 | 중 |
| 1 | Stage 3 필터 61 | 0 | 소 |
| 2 | Stage 1 CCD 원본으로 문턱 재보정 + 시간 특징 | 0 | 중 (여유 0.021) |
| 2 | Stage 2 entry = 판별기 점수 상승 시작점 | 0 | 대 (entry 11% · 가중치 0.338) |
| 3 | Stage 2 entry_side 학습 분류기 (CrashX, 라이선스 확인 후) | 소 | 중 |
| 3 | Stage 3 종가속도 직접 분류 | 소 | 중 |

---

## 5. 추가 측정 (2026-09-26) — Stage 1 챔피언을 CCD 원본에 돌린 결과

| 대상 | 판정 |
|---|---|
| CCD crash 원본 무작위 150개 | ORIGINAL 114 / **RERECORDED 36 (오탐 24%)** |
| 공개 원본 5개 | R, R, O, R, R (4개 오탐) |
| 공개 재녹화 5개 | O, R, O, R, R (3개 적중) |

**해석:** CCD 원본처럼 생긴 영상이 평가 원본이었다면, 오탐 24%만으로도 macro-F1 이 0.85 아래로 내려갔어야 한다.
실제 평가 점수는 0.8955 이고, 소프트 컷 0.45·0.55 가 **양쪽 다 졌다**(평가셋 기준 문턱은 이미 최적).
→ **평가 원본은 CCD 원본 파일과 인코딩·처리 이력이 다를 가능성이 높다.**
→ "CCD 원본으로 문턱 재보정"(1절 방안 1)은 평가셋과 어긋날 위험이 커서 **보류**한다.
Stage 1 에서 남은 방향은 **시간 방향 특징**(고스팅·프레임 반복)과 **합성 재녹화 학습**이다.

---

## 6. 학습 결과 (2026-09-26)

### Stage 2 충돌 판별기 추가 시드 (CCD 자차 검증 160클립, ±3 적중)

| 모델 | 적중 |
|---|---:|
| e47 / all_s1 / ego_s2 / all_s3 / ego_s4 | 0.700 / 0.713 / 0.700 / 0.675 / 0.700 |
| 3개 앙상블 (E49, 리더보드 +0.0015) | 0.719 |
| 5개 앙상블 | 0.706 |

→ 5개로 늘려도 검증 이득이 없다(2클립 차이, 잡음 범위). **3개 유지.**

### Stage 2 entry_side 분류기 (CrashX 충돌 위치 라벨)

- 라벨: 자차 관여 CCD 중 카메라차 충돌 위치가 왼쪽/오른쪽으로 명확한 329클립 (L 173 / R 156). 학습 264 / 검증 65
- 입력: 충돌 직전 8장 160×90 + 차분 7장. 좌우 반전 증강 시 라벨도 반전. 반전 TTA
- 검증 정확도: 시드 0 / 1 / 2 = **0.785 / 0.785 / 0.769** (3개 평균 0.754)
- **같은 65클립에서 현재 파이프라인(Faster R-CNN 추적 + 중심 비교) = 0.615** — 역산한 평가 적중 62.1% 와 거의 같다
- 기대 이득: +0.16 × 가중치 0.162 ≈ Stage 2 +0.026 (가중합 +0.010)

⚠ **CrashX 저장소에 LICENSE 가 없다.** 라이선스 표기가 없으면 기본적으로 저작권자 권리가 유보된다.
DACON 답변의 "라이선스상 이용이 허용되는 범위" 조건을 만족한다고 단정할 수 없으므로, **제출에 쓰기 전 사용자 결정이 필요하다.**
