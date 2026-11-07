# Provenance 및 고정 조건 audit

A/B는 각 59개 snapshot 파일의 실제 SHA-256을 재계산했으며, 저장된 per-file/aggregate/manifest hash가 일치한다. 공통 aggregate는 `18377821b243cfbac53a14ad151b8195dc659e29d2cfe6655fe8592ed30091d6`이다. 환경·resolved optimizer parameters·timing scope가 A/B에서 정확히 일치한다. 20개 생성 YAML/run manifest와 4,400개 trial metadata의 고정 조건을 확인했다. Audit 370건 중 불일치 0건, trial 조건 불일치 0건이다.

B DoN .75의 200건은 A seed 0–49에서 재사용됐다. 원본 suite/trial CSV hash와 모든 NPZ/metadata 쌍의 바이트 SHA-256, 결과·pose·seed·geometry·parameters·timing·환경이 동일하고 B 내부 canonical 경로에 존재한다. 필터링된 성공 step 75,912행도 원본과 동일하다. 이들은 200개의 새로운 독립 관측이 아니다.

최초 디렉토리 `rectangle_don_100_20261105`는 preflight에서 CC-SF `d_col`이 과거 baseline에 없고 현재 0.00005여서 중단됐다. 실제 trial CSV는 0개다. 이후 snapshot 변경 파일은 baseline JSON 한 개뿐이며, resolved parameters는 최초와 현재가 동일하다. `approved_amendments`의 사용자 승인은 현재 CC-SF 유지와 이 기준 변경을 기록한다. 그 외 비진단 파라미터 검사는 유지됐고 amended baseline과 일치한다. 과거 네 parameter 원본 파일 SHA-256도 기록과 일치한다.

과거 CC-SF source와 현재 source는 다르다. 2026-11-03은 stage 0..N-1에 projected-variance 행 한 개와 terminal 제약 없음이었고, 현재는 stage 1..N에 hard PSDF guard 및 projected-variance 행 두 개, terminal 제약, measured-pose의 별도 stage-0 guard를 사용한다. 따라서 baseline 일치는 과거 제어 수식·source 일치가 아니다. 실제 unified diff를 보존했다.

MPCC `solve_nlp()`는 주 solver 실패 시 warm-start 상태를 갱신하고 같은 solver의 `AcadosSolution`을 반환하며 CC의 zero-input safe-stop/backup 구현이 없다. Runtime 카운터는 null이므로 safe-stop/recovery를 구조적 0으로만 설명할 수 있다. CC safe-stop은 fallback 제어 step 수이고 독립 위험 episode 수가 아니다. Aggregate 카운터 및 마지막 모드만으로 episode·최장 연속 길이·정확한 시점은 복원할 수 없다.

Run manifest `cli`의 60 s, s_path, psdf는 batch 실행에 사용되지 않은 parser 기본값이다. 저장 config/configurations와 `test_nmpc.py` override 경로에서 실제 80 s, maze 및 각 optimizer가 확인된다. Debug risk 보조 flag가 일부 true여도 master `use_debug=false`이며 metadata 전수 일치한다.

실행일은 A 2026-11-05T16:05:53.873879+09:00, B 2026-11-05T16:56:52.798695+09:00이다. 기록 환경은 Python 3.10.12, NumPy 1.24.4, Torch 2.4.1+cu121, CUDA 12.1, RTX 4090, CasADi 3.7.1, compute thread 6, torch interop thread 24다. 동일 recorded 환경은 확인되지만 CPU 모델·GPU driver·acados revision과 온도·전력·다른 프로세스 부하는 기록되지 않아 wall-time 변동을 배제할 수 없다. 주행/solver를 재실행하지 않았고 원본 파일을 변경하지 않았다.
