# Goal 실행 체크리스트

## 준비

- [x] controller 전체 계산시간 계측 및 벤치마크 집계기 준비
- [x] scale별 alpha/beta 실행 설정 계산

## MF Scale Sweep

- [ ] 시행 1 — PSDF-MPCC-CC-MF, `covariance_growth_scale=0.1`
- [ ] 시행 2 — PSDF-MPCC-CC-MF, `covariance_growth_scale=0.2`
- [x] 시행 3 — PSDF-MPCC-CC-MF, `covariance_growth_scale=0.3` — timeout
- [ ] 시행 4 — PSDF-MPCC-CC-MF, `covariance_growth_scale=0.4`
- [ ] 시행 5 — PSDF-MPCC-CC-MF, `covariance_growth_scale=0.5`
- [ ] 시행 6 — PSDF-MPCC-CC-MF, `covariance_growth_scale=0.6`
- [ ] 시행 7 — PSDF-MPCC-CC-MF, `covariance_growth_scale=0.7`
- [ ] 시행 8 — PSDF-MPCC-CC-MF, `covariance_growth_scale=0.8`
- [ ] 시행 9 — PSDF-MPCC-CC-MF, `covariance_growth_scale=0.9`
- [ ] 시행 10 — PSDF-MPCC-CC-MF, `covariance_growth_scale=1`
- [ ] 안전 우선 기준으로 최적 scale 선정

## Optimizer 비교

- [ ] 시행 11 — PSDF-MPC (`psdf`)
- [ ] 시행 12 — PSDF-MPCC (`mpcc`)
- [ ] 시행 13 — PSDF-MPCC-CC-SF (`rmpcc_pv`)
- [ ] 선정된 MF sweep 결과 재사용
- [ ] 4개 optimizer 비교표 생성

완료 표시는 raw trial/clearance 로그와 run summary가 모두 존재할 때만 갱신됩니다.
