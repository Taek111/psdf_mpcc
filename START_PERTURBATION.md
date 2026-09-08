# Navigation trial 시작 위치 randomization

`test_nmpc.py`에서 `--perturb-start`로 켜고 `--no-perturb-start`로 끈다.
기본값은 **꺼짐**이다. `--single`, YAML batch, `--safe-stop-batch` 모두 적용된다.

```bash
python test_nmpc.py --single --maze-type maze --robot-shape rectangle --optimizer-type psdf --perturb-start --start-seed 42
python test_nmpc.py --single --maze-type maze --robot-shape rectangle --optimizer-type dcbf --perturb-start --start-seed 42
python test_nmpc.py --single --maze-type maze --robot-shape rectangle --optimizer-type obca --perturb-start --start-seed 42
```

`--optimizer-type obca`는 `control/obca_optimizer.py`의 `OBCAOptimizer`를 사용한다.
기존 `acados` 옵션은 별도의 SQP NMPC 구현이다.
같은 환경·footprint·설정에서는 세 방법에 **동일한 seed**를 사용하면 실제 시작 pose도 동일하다.
다음 trial에서는 seed를 43, 44, …로 바꾼다. `--robot-shape triangle` 또는 `pentagon`도 같은 방식이다.
seed를 생략하면 자동 생성되며, 한 YAML batch의 공통 설정을 사용하는 방법들은 생성된 seed를 공유한다.
별도 실행끼리 비교할 때는 명시적인 seed를 사용한다.

| 설정 | 기본값 | 의미 |
| --- | --- | --- |
| `--start-position-std` | `0.005` | world x/y 각각 독립 Gaussian 표준편차 5 mm |
| `--start-heading-std-deg` | `0.0` | heading 유지; 필요하면 예: `1.0`으로 yaw도 변동 |
| `--start-min-clearance` | `0.01` | 전체 footprint와 장애물·지도 경계 사이 최소 여유 10 mm |
| YAML `max_sigma` | `3.0` | 각 성분의 ±3σ 밖 표본을 거부; 기본 x/y 각각 최대 ±15 mm |
| YAML `max_attempts` | `1000` | 안전 표본을 못 찾으면 오류로 종료 |

실제 plant의 시작 pose에 한 번 적용하며, global planner도 그 상태에서 시작한다.
localization noise와는 별도의 RNG를 사용한다. 기존 `localization_error.seed`는 바뀌지 않는다.
Gaussian tail과 안전거리 미달 표본은 다시 뽑는다. 따라서 최종 분포는 범위와 안전거리로
조건화된 truncated Gaussian이다. 기본 위치 변동의 최대 유클리드 크기는 약 21.2 mm이다.

5 mm는 현재 footprint 크기(rectangle 150 × 90 mm, triangle 길이 150 mm,
pentagon 반지름 50 mm)보다 충분히 작다. `maze`/`oblique_maze` 시작점의
nominal 여유가 수 cm이므로 cm 단위의 표준편차보다 안전거리 제한에 덜 영향을 받는다.
표준편차 2.5/5/10 mm를 각각 5개 환경 × 3개 footprint × seed 0–999로
검사했다. 5 mm의 15,000개 표본은 모두 10 mm 이상 여유를 유지했고,
최소 여유는 10.119 mm였다. 각 조합에서 재추출은 3σ tail에 해당하는
5/1,000건(0.5%)뿐이었다. 10 mm에서는 `maze`의 triangle 재추출이
122/1,000건(12.2%)으로 늘어 안전거리 제한이 분포에 더 크게 영향을 주었다.
10 mm 검사는 현재 OBCA `margin_dist=0.01`을 기준으로 한다.
이 검사는 **초기 기하학적 안전거리**를 확인한다. 전체 MPC horizon의 feasibility나
모든 trial의 solver 수렴·목적지 도달을 보장하지는 않는다.

`config/config.yaml`의 `start_perturbation` 블록으로도 설정할 수 있다.
개별 `test_configs` 항목에 같은 이름의 블록을 넣으면 공통 설정을 덮어쓴다.
명시적인 CLI 옵션이 가장 우선한다. 예를 들어 trial별로 `seed: 42`와 `seed: 43`을
지정할 때는 공통 CLI `--start-seed`를 생략한다.

```yaml
start_perturbation:
  enabled: true
  seed: 42
  position_std: 0.005
  heading_std_deg: 0.0
  max_sigma: 3.0
  min_clearance: 0.01
  max_attempts: 1000
```

활성화한 실행의 출력 이름에는 `_startseed42`처럼 seed가 붙는다.
`data/start_pose_<출력이름>.json`에 seed, nominal/실제 pose, 변동량,
적용 설정, nominal/실제 clearance, 추출 시도 횟수를 solver 실행 전에 저장한다.
`output_root_dir`를 지정했다면 그 아래 `data/`에 저장한다.
같은 seed·방법·환경·shape의 재실행은 같은 출력 파일을 사용한다.
footprint 또는 설정을 바꾸면 안전거리 재추출 여부가 달라질 수 있으므로
비교 시 JSON의 `initial_pose`도 확인한다.

회귀 테스트는 `python -m unittest discover -s tests -p test_start_perturbation.py`로 실행한다.
초기 기능 추가 당시 solver 확인에서는 `maze`, seed 8, 0.3초 조건에서 PSDF/acados 모두
세 footprint의 3개 제어 step을 정상 완료했다. DCBF는 rectangle/triangle에서
첫 solve 후 기존 거리계산의 CasADi `Psd constraints not implemented yet` 오류가 발생했고,
pentagon 실행과 perturbation을 끈 rectangle 대조 실행은 120초 제한에 도달했다.
DCBF의 `b_robot_curr` 계산은 `(m,) + (m,1)`을 `(m,m)`으로 확장하며,
이 거리계산 오류는 nominal pose로도 별도 재현됐다. 따라서 DCBF 전체 실행 검증은 제한적이다.
