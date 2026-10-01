# 새 전략 9개 비교와 최종 파일 선정

9개 새 후보를 구현하고 세 모델을 실제 학습했다. 새 최고점이 입증된 후보는 없었으므로 최종 폴더에는 **기존 공식 최고 E53과 바이트가 같은 파일 하나**를 남겼다. 이름을 바꿔 새로운 성능 향상처럼 취급하지 않았다.

선정 파일: `final_submission/submit_e53_triple.zip`.
SHA256: `f0bd067b1dd261991edae112e17224f31b52982426f4200919a7f06dcd6699cf`.
공식 기록: **0.8954991087 / 0.3615358678 / 0.6662995310**. 목표 **0.99625 / 0.84098 / 0.83883**에 도달했다는 결과는 없다.

| 후보 | Stage | 내부 비교 지표 | E53/기준 | 후보 |
|---|---:|---|---:|---:|
| N1 noise_coherence | 1 | public 10-example macro-F1 | 0.375000 | 0.333333 |
| N2 collision_rank | 2 | CCD 160 collision hit | 0.718750 | 0.712500 |
| N3 collision_median | 2 | CCD 160 collision hit | 0.718750 | 0.712500 |
| N4 lane_entry | 2 | Public entry_frame changed count; no entry truth | 0.000000 | 0.000000 |
| N5 polynomial_accel | 3 | Other-camera conditional moving accel F1, physical threshold .3 | 0.507344 | 0.450264 |
| N6 viterbi_accel | 3 | Other-camera conditional moving accel F1, physical threshold .3 | 0.507344 | 0.501150 |
| N7 tcn | 3 | Other-camera conditional moving accel F1, physical threshold .3 | 0.507344 | 0.414425 |
| N8 gru | 3 | Other-camera conditional moving accel F1, physical threshold .3 | 0.507344 | 0.357318 |
| N9 robust_collision | 2 | CCD 160 ensemble collision hit | 0.718750 | 0.718750 |

N4의 숫자는 점수가 아니라 출력이 달라진 영상 수다. 모든 다른 숫자도 공식 점수가 아닌 각 자료의 내부 지표이며 서로 다른 Stage 지표를 평균하지 않았다.

## 공개 Stage 3 실제 라벨 50개와 비교

| 후보 | 가감속 Macro-F1 | 주행 중 조향 Macro-F1 |
|---|---:|---:|
| baseline | 0.791384 | 0.701338 |
| polynomial_accel | 0.636389 | 0.701338 |
| viterbi_accel | 0.791384 | 0.701338 |

이 50개 라벨은 이미 반복 사용한 작은 공개 표본이다. 여기서 높은 후보를 곧바로 공식 최고 후보로 선정하지 않는다. 영상은 정답의 frame_index/sample_index 대응으로 10Hz로 변환한 동일 캐시를 사용했다.

## 마지막 충돌 모델 추가 학습

E53의 작은 CNN을 6 epoch 추가 학습했다. 프레임 배율·평행이동·흐림·명암·양자화·잡음과 충돌 위치를 바꾸는 시간 구간을 함께 증강했다. 기존의 모델 추가/TTA와 다르게 작은 백본 자체를 재학습했다. 검증 160개는 gradient 학습에서 제외했다.

| 조건 | 기존 단일 모델 | 새 단일 모델 | 신규 적중/기존 적중 손실 |
|---|---:|---:|---|
| native | 0.70000 | 0.67500 | 7/11 |
| blur_quantized | 0.68125 | 0.70000 | 8/5 |
| early_crop | 0.74375 | 0.73125 | 6/8 |

흐림 조건만 개선됐고 원본·초기 충돌 구간에서는 하락했다. 3개 앙상블에서 첫 모델을 교체하면 전체 적중률은 동률(3개 개선/3개 손실)이므로 최종 제출에 넣지 않았다.

## 상관관계 분석을 반영한 판단

- E54·E56에서 내부 개선과 공식 결과가 역전됐다. 내부 개선 하나만으로 목표점수나 1등 도달을 예측하지 않는다.
- 새 후보의 내부 하락이 공식 하락을 증명하지는 않는다. 다만 제한된 최종 제출에 채택할 긍정적인 근거가 없으므로 보류했다.
- CCD 검증 160개 정답은 50프레임 중 30~49에 몰려 있다. 앞부분 충돌을 평가하지 못하는 기존 검증의 빈틈을 확인했고, N9에서는 시간 구간을 앞당긴 스트레스 비교를 추가했다. 이는 새로운 실제 충돌 데이터 검증을 대체하지 않는다.
- 초기/후반 CCD 하위집합 중 초기(<25)는 0개이므로 해당 hit는 null로 기록했다. 처음 실행의 빈 집합 평균 경고를 유효한 점수로 취급하지 않았다.
- E57/E58은 여전히 미채점 실험 후보다. 이번 조사로 그 후보의 공식 점수 상승이 입증된 것은 아니다.

## 보존한 산출물

- `candidate_manifest.json`, `candidates/`: E53과 독립 후보 6개의 추론 코드.
- `accel_tcn.pt`, `accel_gru.pt`, `collision_robust.pt`: 실제 학습한 새 가중치.
- `training_report.json`, `benchmark_report.json`, `robust_collision_report.json`: 원자료별 점수와 비교 결과.
- `selection.json`: 선택 이유, 목표와 실측 구분, 파일 해시.
- `strategy_inventory.md`: 이번 비교 범위와 과거 전략 중복 점검. 전체 가능한 전략을 소진했다는 의미는 아니다.

기존 제출본을 삭제하지 않았다. 최종 제출용 `final_submission/`에는 선택한 ZIP 하나만 두었다. 나머지 실험은 재현·공식 결과 대조를 위해 보관한다.
