# Goal 실행 체크리스트

## 준비

- [x] 벤치마크 실행기와 controller 전체 계산시간 계측 추가
- [x] scale별 alpha/beta 실행 설정 계산 (`RMPCCOptimizerParam` 미수정)
- [x] 계산시간·deadline·주행시간·clearance·입력 변화·실패 지표 집계
- [x] 짧은 smoke test로 raw 로그와 요약 CSV 생성 확인

## MF Scale Sweep

- [x] 시행 1 — PSDF-MPCC-CC-MF, `covariance_growth_scale=0.1` — success
- [x] 시행 2 — PSDF-MPCC-CC-MF, `covariance_growth_scale=0.2` — success
- [x] 시행 3 — PSDF-MPCC-CC-MF, `covariance_growth_scale=0.3` — success
- [x] 시행 4 — PSDF-MPCC-CC-MF, `covariance_growth_scale=0.4` — success
- [x] 시행 5 — PSDF-MPCC-CC-MF, `covariance_growth_scale=0.5` — success
- [x] 시행 6 — PSDF-MPCC-CC-MF, `covariance_growth_scale=0.6` — success
- [x] 시행 7 — PSDF-MPCC-CC-MF, `covariance_growth_scale=0.7` — success
- [x] 시행 8 — PSDF-MPCC-CC-MF, `covariance_growth_scale=0.8` — timeout
- [x] 시행 9 — PSDF-MPCC-CC-MF, `covariance_growth_scale=0.9` — timeout
- [x] 시행 10 — PSDF-MPCC-CC-MF, `covariance_growth_scale=1` — success
- [x] 안전 우선 기준으로 최적 scale 선정 — selected=0.7

## Optimizer 비교

- [x] 시행 11 — PSDF-MPC (`psdf`) — success
- [x] 시행 12 — PSDF-MPCC (`mpcc`) — success
- [x] 시행 13 — PSDF-MPCC-CC-SF (`rmpcc_pv`) — success
- [x] 선정된 MF sweep 결과 재사용
- [x] 4개 optimizer 비교표 생성
- [x] 최적 scale 결론과 결과 요약 보고서 작성

완료 표시는 raw trial/clearance 로그와 run summary가 모두 존재할 때만 갱신됩니다.
