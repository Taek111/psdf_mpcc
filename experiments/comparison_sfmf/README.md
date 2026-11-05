# CC-SF / CC-MF 비교 실험 패키지

2026-11-05 저장. 기존 120회 주행과 후속 오프라인 분석을 복사한 패키지이며, 저장을 위해 새 주행이나 재분석을 실행하지 않았다.

먼저 [결과 정리](results.md)를 읽으면 수치, 그래프, 해석 범위와 raw data 링크를 함께 볼 수 있다.

| 경로 | 내용 |
|---|---|
| [results.md](results.md) | 120회 주행 결과 및 margin·위험도·switching 분석 |
| [figures/](figures/) | 논문 그래프 10종(PDF·PNG·영문 caption), 주행 요약 그림, 대표 사례 그림 |
| [raw_data/](raw_data/) | 원본 batch 전체: 120개 trial, 설정, source snapshot, 원본 보고서, 모든 파생 분석과 검증 기록 |
| [scripts/](scripts/) | 분석·그래프·검증 코드와 필요한 local helper의 보관본 |
| [context/previous_guard_ablation.md](context/previous_guard_ablation.md) | 이전 125 mm ablation의 참고 문서; 그 이전 실험 raw data는 이 패키지 범위 밖 |
| [MANIFEST.json](MANIFEST.json) | 파일 크기·SHA-256·원본 대응 경로와 복사 검증 결과 |
| [SHA256SUMS](SHA256SUMS) | 패키지 내용 확인용 SHA-256 목록 |

## 주 그림

- [Switching-boundary 비교](figures/fig02_switch_boundary.pdf): 동일 상태·고정 body-local covariance에서 SF·MF 잔차 변화를 비교한다.
- [공통 margin·위험도](figures/fig01_common_margin_risk.pdf): geometric margin, 공통 chance-budget margin, 위험예산 초과율을 비교한다.
- [비선택 feature 및 모델 오차](figures/fig03_feature_coverage.pdf): 전체 유효 주행 구간의 omission 빈도와 모델 오차를 보존한다.

그림의 작은 점은 trial 통계, 큰 점과 오차막대는 폭별 trial 통계의 mean±sample SD(ddof=1)다. SD는 신뢰구간이 아니다. 같은 seed를 여러 폭에서 사용했고 실패 trial의 관측 구간은 더 짧다. 모델 위험도 합 R은 실측 충돌 확률이 아니며 1을 넘을 수 있다. MF에 불리한 결과와 covariance 이상값도 제외하지 않았다.

## Raw data 범위

- 폭: 0.125–0.130 m, 0.001 m 간격.
- 방법: CC-SF, CC-MF. 각 폭·방법당 seed 0–9, 총 120회.
- `raw_data/width_XXXmm/<method>/seed_<seed>/`에 원본 checkpoint JSONL, cycle CSV, pose·covariance 로그, 설정·파라미터·종료 결과가 있다.
- [trial_results.csv](raw_data/trial_results.csv)와 [benchmark_summary.csv](raw_data/benchmark_summary.csv)는 주행 집계다.
- [constraint_stability_20261105/](raw_data/constraint_stability_20261105/)에는 240개 stage/trial cache, trial별 통계, boundary 전체 사건, covariance 재전파 감도 결과가 있다.
- `transition_analysis/`와 `omitted_feature_events/`를 포함한 기존 파생 자료도 원본 폴더 구조 그대로 보존했다.

Raw data 내부 문서·JSON·NPZ의 absolute source path 및 script hash는 당시 출처를 그대로 보존한다. 현재 파일은 `raw_data/`의 같은 상대 구조로 찾을 수 있다. 링크가 패키지 경로에 맞게 정리된 문서는 최상위 `results.md`다. `raw_data/benchmark_results.md`는 원본 보고서의 변경 없는 보관본이다.

## 오프라인 재현

아래는 필요할 때 실행하는 안내이며 패키징 과정에서 실행하지 않았다. 기존 결과를 보존하기 위해 새 `recomputed/` 폴더를 사용한다. Python 3.10+, NumPy, pandas, matplotlib, PyTorch, PyYAML이 필요하다. 원 분석 환경은 프로젝트의 `pp` conda 환경이다. Geometry·covariance 코드는 `raw_data/source_snapshot/`에서 읽는다.

패키지 최상위(`experiments/comparison_sfmf`)에서:

```bash
python scripts/extract_constraint_stability.py --results-dir raw_data --output-dir recomputed
python scripts/analyze_constraint_switch_boundaries.py --results-dir raw_data --analysis-dir recomputed --device cuda
python scripts/analyze_rollout_covariance_risk.py --experiment raw_data --analysis-dir recomputed --device cuda
python scripts/aggregate_constraint_stability.py --experiment raw_data --analysis-dir recomputed
python scripts/plot_constraint_stability_paper.py --analysis-dir recomputed
```

GPU가 없으면 boundary와 rollout의 `--device cuda`를 `--device cpu`로 바꿀 수 있다. 부동소수점 차이 때문에 세부 경계값이 달라질 수 있다. 원래 결과와 출처 검증은 이 패키지의 보관본을 기준으로 한다. 원 출처 경로가 포함된 cache를 이어 쓰지 않도록 위 명령에는 resume/skip-existing 옵션을 넣지 않았다.

`write_constraint_stability_report.py`는 원 프로젝트 구조를 전제로 하는 당시 코드의 보관본이다. 이 패키지에서 직접 실행하는 명령에는 포함하지 않았다. 재현된 그래프는 `recomputed/figures/`에 생성되며 최상위 `figures/`는 보관된 최종 그림이다.

## 무결성 확인

```bash
sha256sum -c SHA256SUMS
```

`SHA256SUMS`는 복사된 원본, 그림, 스크립트, 결과 문서와 README/context를 포함한다. 자기 자신과 `MANIFEST.json`은 순환 참조를 피하기 위해 목록에서 제외한다. 복사 단계에서 모든 원본 batch 파일과 대응 복사본의 SHA-256을 비교했다.
