# 25회 실행시간 추정

2026-09-17 02:35 KST에 보관한 소스 복사본으로 6개 조합을 각각 1회(seed 0) 순차 실행했다.
측정 PC: 13th Gen Intel(R) Core(TM) i9-13900KF, 프로젝트 .venv Python 3.10.12. 기존 acados 빌드 캐시 재사용.

6개 조합의 총 실제 경과 시간: 229.25초 = 3.82분.
조합별 25회, 총 150회 단순 환산: 95.52분 (약 1시간 36분).
프로세스 시작 비용을 한 번만 포함하면 94.43분 (약 1시간 34분).

| Map | Method | 1회 실제시간(s) | 결과 | Solver mean/p95(ms) |
|---|---|---:|---|---:|
| maze | psdf | 18.94 | success | 0.309 / 0.386 |
| maze | dcbf | 50.64 | success | 190.718 / 378.977 |
| maze | obca | 55.79 | success | 255.407 / 510.764 |
| oblique_maze | psdf | 15.51 | success | 0.336 / 0.456 |
| oblique_maze | dcbf | 38.86 | failure | 127.887 / 236.377 |
| oblique_maze | obca | 46.80 | failure | 184.537 / 414.352 |

목표 도달 4개, 정체 실패 2개(oblique_maze의 DCBF/OBCA). Solver 오류 trial은 없었다.
시간은 초기화·경로 계획·solver·저장·정리를 포함한다. 조합당 seed 하나로 계산한 추정이므로 다른 seed나 PC에서는 달라질 수 있다.
앞선 변경 전 코드 실행의 약 3시간 44분 추정은 현재 소스 추정에 사용하지 않았다.
CSV 6개 조합, seed/시작 pose 일치, 성공 solver 스텝 수와 mean/p95, 소스 복사본 hash를 검증했다.
측정 후 원본 작업 폴더에서 추가 변경된 파일: 없음.
