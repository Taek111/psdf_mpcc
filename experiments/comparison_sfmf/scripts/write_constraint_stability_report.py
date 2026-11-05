"""Update existing benchmark reports with the complete offline stability analysis."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
RUN='sf_mf_width125_130mm_step001_10trials_20261104'
EXPERIMENT=ROOT/'benchmark_runs'/RUN
BASE=EXPERIMENT/'constraint_stability_20261105'
PREFIX=f'benchmark_runs/{RUN}/constraint_stability_20261105/'

def mean_sd(mean,sd,digits=3,scale=1):
    if not np.isfinite(mean): return '—'
    return f'{mean*scale:.{digits}f} ± {sd*scale:.{digits}f}' if np.isfinite(sd) else f'{mean*scale:.{digits}f} (SD 미정)'

def main():
    grouped=pd.read_csv(BASE/'width_mean_sd.csv')
    trials=pd.read_csv(BASE/'trial_metrics.csv')
    boundaries=pd.read_csv(BASE/'boundaries/per_trial_metrics.csv')
    boundary_report=json.loads((BASE/'boundaries/summary.json').read_text())
    assert boundary_report['completed_trial_stages']==240
    primary=boundaries[(boundaries.offset_um==10)&(boundaries.subset=='ordinary_inverse_cdf')]
    def lookup(width,method,metric,stage=1,scope='full_observed'):
        x=grouped[(grouped.width_mm==width)&(grouped.method==method)&(grouped.metric==metric)&(grouped.stage==stage)&(grouped.scope==scope)]
        assert len(x)==1,(width,method,metric)
        return x.iloc[0]
    def entry(w,m,k,scale=1,digits=3):
        r=lookup(w,m,k)
        return mean_sd(r['mean'],r['sd'],digits,scale)
    def widths_lower(metric):
        return ', '.join(str(w) for w in range(125,131) if lookup(w,'cc_mf',metric)['mean']<lookup(w,'cc_sf',metric)['mean']) or '없음'
    def boundary_wins(stage,key1,key2):
        a=primary[primary.stage==stage]
        ok=np.isfinite(a[key1])&np.isfinite(a[key2])
        return int((a.loc[ok,key2]<a.loc[ok,key1]).sum()),int(ok.sum())
    qwin1,n1=boundary_wins(1,'delta_sf_quantile_mean_abs','delta_mf_quantile_mean_abs')
    pwin1,_=boundary_wins(1,'delta_p_selected_mean_abs','delta_probability_sum_mean_abs')
    qwin10,n10=boundary_wins(10,'delta_sf_quantile_mean_abs','delta_mf_quantile_mean_abs')
    pwin10,_=boundary_wins(10,'delta_p_selected_mean_abs','delta_probability_sum_mean_abs')
    text=f'''<!-- constraint-stability:120trials_20261105:start -->

# 120회 주행의 constraint margin·위험도·switching 통계

분석일 2026-11-05(KST). 2026-11-04의 기존 120개 trial을 재분석했으며 새 주행은 실행하지 않았다. 폭 125–130 mm, 두 방법, 각 seed 0–9를 모두 포함한다. Stage 1(0.1 s)을 주 분석, stage 10(1 s 예측)을 보조 분석으로 고정했다. 성공 trial이나 특정 주행 위치만 선택하지 않았다.

**핵심 결과:** 동일 상태·covariance의 switching-boundary 비교에서 trial 평균 quantile 변화가 MF 쪽이 작은 경우는 stage 1 **{qwin1}/{n1}**, stage 10 **{qwin10}/{n10}**였다. 확률예산 잔차 변화로 비교하면 각각 **{pwin1}/{n1}**, **{pwin10}/{n10}**다. 이 분모는 유효 경계가 있는 source trajectory trial 수이며, 경계를 독립 반복으로 센 수가 아니다. 실제 주행 경로의 공통 위험예산 초과율 평균이 MF에서 낮은 폭은 **{widths_lower('R_exceedance_pct')} mm**였다. 경계 국소 민감도와 폐루프 주행의 margin·위험도는 구분해 해석한다.

## 분석 단위와 공통 기준

각 trial에서 평균·SD·P05·P95·위반 비율을 먼저 계산하고, 폭별 10개 trial 통계를 동일 가중으로 평균냈다. 그림의 작은 점은 trial 하나, 큰 점은 trial 통계의 평균, 오차막대는 **trial 간 sample SD(ddof=1)**다. 이는 standard error나 confidence interval이 아니다. 동일 seed를 폭별로 재사용했으므로 60개 또는 120개를 독립 반복으로 가정한 검정은 하지 않았다. 유효 표본이 없는 trial 통계는 NaN이며 0으로 대체하지 않는다.

- **공통 geometric margin:** h_G = φ−d_col [mm], d_col=0.05 mm. 해에서 geometry를 새로 평가한 값이다. h_G<−1 µm를 유의미한 음수 진단 기준으로 쓴다.
- **공통 chance-budget margin:** b=ε−R, R=Σ_i p_i, p_i=Φ((d_min−d_i)/σ_i), ε=0.1, d_min=0.1 mm. 두 방법의 궤적에 같은 multi-feature 평가를 적용한다. R은 구현의 Gaussian/Boole surrogate sum으로 중복 feature가 포함되고 1을 넘을 수 있다. 관측 충돌 확률이나 정확한 union probability가 아니다.
- **위험도 domain:** fresh φ>d_min인 separated geometry에서만 p_i와 R 통계를 사용한다. Domain 밖을 위험도 0으로 넣지 않고, domain 이탈 비율 및 `domain 이탈 OR R>ε` 비율을 별도 보고한다.
- **최적화 해의 분모:** 성공한 normal solve의 후보 상태를 사용한다. 실패한 solver iterate는 저장되지 않아 복원하지 않는다. 실패 trial도 종료 전의 유효 후보는 포함한다. 반복된 20-step terminal Guard stop은 위험도 평균에 반복 삽입하지 않고 실제 geometry/실패 통계로 분리한다.
- **계획 covariance:** 각 후보해에 저장된 body-local planning covariance로 모델 위험도를 계산한다. 실제 측정 covariance나 Monte Carlo 충돌 확률이 아니다. Stage 10은 실행된 다음 pose가 아닌 horizon 예측이다.
- **단위와 cap:** native SF 거리 잔차와 MF quantile 잔차의 크기·SD를 직접 비교하지 않는다. 공통 SF 거리 진단은 두 궤적 모두 archived SF의 floor/cap/gamma를 사용한다. Unclipped SF quantile은 cap이 활성인 native SF row와 동치가 아니다.

## 주행 궤적에서의 공통 margin·위험도

![공통 margin과 위험도]({PREFIX}figures/fig01_common_margin_risk.png)

각 항목은 **trial별 통계의 mean ± SD**다. Chance-budget margin은 percentage points(pp)로 표시한다. R 초과율은 valid separated 후보에 대한 조건부 비율이다.

| 폭(mm) | SF P05 h_G(mm) | MF P05 h_G(mm) | SF R 초과율(%) | MF R 초과율(%) | SF P05 b(pp) | MF P05 b(pp) |
|---:|---:|---:|---:|---:|---:|---:|
'''
    for w in range(125,131):
        text+='| '+str(w)+' | '+' | '.join([
            entry(w,'cc_sf','guard_margin_mm_p05'),entry(w,'cc_mf','guard_margin_mm_p05'),
            entry(w,'cc_sf','R_exceedance_pct'),entry(w,'cc_mf','R_exceedance_pct'),
            entry(w,'cc_sf','R_budget_residual_p05',100),entry(w,'cc_mf','R_budget_residual_p05',100)])+' |\n'
    text+=f'''
126·128·129 mm에서는 공통 R 초과율 평균이 MF에서 낮았지만, 125 mm의 차이는 매우 작고 공통 시간 비교에서 부호가 바뀐다. P05 geometric margin은 125·127·130 mm에서 오히려 SF가 크다. 130 mm에서는 두 방법 모두 실패했고 MF의 probability-domain 이탈 비율도 커서, 조건부 R 초과율 하나로 안전성 우위를 주장하지 않는다.

P05 budget margin과 P95 R은 b_P05=ε−R_P95 관계다. 평균 R 및 trial 내 SD도 버리지 않고 [risk moment 그림]({PREFIX}figures/figS06_risk_mean_and_dispersion.pdf)에 표시했다. 드문 큰 covariance 상태는 평균·SD를 크게 바꿀 수 있으므로 P05/P95와 함께 해석한다.

[Domain·실제 geometry·QP fallback]({PREFIX}figures/figS02_domain_and_execution.pdf)에서는 위험도 domain 밖 후보를 별도로 집계하고 실제 최소 signed PSDF를 보존한다. [동일 seed 공통 시간 비교]({PREFIX}figures/figS04_paired_exposure_sensitivity.pdf)는 두 trial의 마지막 normal solve 중 더 이른 시각까지 양쪽을 잘라 결과를 비교한다. 이는 관측 길이의 민감도 분석이며 같은 pose·covariance로 매칭한 인과 실험은 아니다. 모든 scope의 평균·SD와 paired 차이를 CSV에 저장했다.

## Feature switching 전후 잔차 변화

![Switching boundary 잔차 변화]({PREFIX}figures/fig02_switch_boundary.png)

각 궤적의 인접한 유효 nominal state에서 canonical selected feature가 바뀐 구간을 찾았다. 특정 progress 범위나 성공 여부로 고르지 않았다. 양끝 selector가 같고 중간에서 왕복 전환하는 구간은 탐지 대상이 아니므로 모든 기하학적 경계의 전수조사라고 부르지 않는다.

탐지된 경계 양쪽에 각각 10 µm의 유한 offset을 두고, **같은 두 상태와 고정된 왼쪽 body-frame covariance**에 SF와 MF를 함께 평가했다. 이동 크기는 ||Δxy||+r|Δθ|로 정의했다. 관측 구간 밖으로 외삽하지 않았고, 양끝 feature가 다르며 MF active/valid인 separated domain만 썼다. 주 그림은 양끝 모두 ordinary inverse-CDF branch인 동일 사건 집합을 사용하므로 clipping/linear extension의 영향을 제외한다.

상단은 SF standardized quantile과 MF quantile의 절대 변화, 하단은 ε−p_selected와 ε−R의 절대 변화다. 둘 다 trial 내 사건들의 평균을 먼저 계산했다. 그림 아래 n은 사건 수가 아닌 유효 trial 수다. 5/50 µm offset, all-active branch, initial-grid-bracket 내부 subset, P95/RMS 결과도 모두 보존했다. 수학적인 jump limit나 전체 함수의 연속성 증명으로 해석하지 않는다. 50 µm offset의 125 mm SF-source 확률 단위 비교에서는 MF 변화가 약 7.1% 더 큰 반례도 있어, 지정된 offset 조건을 명시해야 한다.

| 폭(mm) | Source path | 유효 trial | SF 평균 | MF 평균 | SF 확률예산 변화(pp) | MF 확률예산 변화(pp) |
|---:|---|---:|---:|---:|---:|---:|
'''
    a=primary[primary.stage==1]
    for w in range(125,131):
        for m in ['cc_sf','cc_mf']:
            r=a[(a.width_mm==w)&(a.method==m)]
            cols=['delta_sf_quantile_mean_abs','delta_mf_quantile_mean_abs','delta_p_selected_mean_abs','delta_probability_sum_mean_abs']
            v=[mean_sd(r[k].mean(),r[k].std(ddof=1),4,100 if i>=2 else 1) for i,k in enumerate(cols)]
            text+=f"| {w} | {m.upper().replace('_','-')} | {int(r[cols[0]].count())} | "+' | '.join(v)+' |\n'
    text+=f'''
[일반 주행의 temporal switching 변화]({PREFIX}figures/figS03_temporal_switch_changes.pdf)도 따로 계산했다. 이는 실제 연속 cycle 사이의 pose와 covariance 변화가 함께 들어가므로 boundary의 국소 민감도와 다른 지표다. Switch/non-switch/all-adjacent 분모와 변화량을 모두 CSV에 남겼으며 결측·fallback을 건너뛰어 연결하지 않았다.

## 비선택 feature 위험과 모델 오차

![Feature coverage와 모델 오차]({PREFIX}figures/fig03_feature_coverage.png)

앞선 사건별 35개 분석을 전체 유효 normal solve로 확대했다. Nominal feature를 고정한 채 후보해에서 `p_selected≤ε−1e−6`이고 `max p_other>ε+1e−6`인 비율을 trial별로 계산했다. 선택 feature와 같은 거리·전체 gradient를 가진 중복 slot은 대안에서 제외했다. 위험해진 대안의 nominal 거리차가 0.5 mm 이하인 비율도 따로 보고한다.

동일 SF 진단 row에 대해 `h_fresh−h_frozen`과 `h_frozen−h_affine`을 분리했다. 앞의 차이는 geometry 재평가 및 gradient/orientation 변화, 뒤의 차이는 covariance-state에 대한 risk-margin 선형화 오차다. 뒤의 값을 covariance dynamics defect라고 부르지 않는다. MF 궤적에서도 같은 SF row로 평가한 counterfactual 진단이며 native MF residual이 아니다.

비선택 feature 위험 빈도는 전체 주행 평균에서 MF가 일관되게 낮지 않았다. 특히 126·129·130 mm에서는 그 반대 방향도 관찰된다. 따라서 대표 충돌 사례에서 확인한 omission 메커니즘을 모든 주행의 주원인이나 일관된 감소율로 확대하지 않는다. 130 mm MF seed 5는 normal solve 4개, common risk-domain 후보 2개뿐이므로 trial 평균율의 높은 값과 큰 SD는 짧은 관측 구간과 함께 읽어야 한다.

## Covariance 일관성에 대한 보조 검산

[같은 후보 pose의 covariance 감도 그림]({PREFIX}figures/figS07_covariance_consistency.pdf)은 저장된 최적화 covariance와, reset 초기 covariance에서 실제 적용 입력으로 한 step 전파한 covariance를 비교한다. 두 값 모두 모델 기반 planning covariance이며 실제 측정 불확실성이 아니다. 원래 데이터나 결과를 교체하지 않았다.

129 mm MF seed 1, cycle 120의 저장된 heading Log-Cholesky 상태가 +9.6245인 반면 같은 입력의 동역학 전파는 −4.6342였다. 원본 checkpoint와 cache가 일치하여 오프라인 복원 오류가 아니다. 같은 후보 pose의 R은 저장 covariance에서 **67.94494**, 동역학 전파 covariance에서 **0.00040533**이었다. 이 값은 solver success 여부와 별개인 비선형 covariance dynamics 일관성 문제를 보여주며 실제 충돌 확률로 해석할 수 없다.

129 mm MF의 trial별 평균 R의 mean±SD는 저장 covariance에서 **0.06817±0.15094**, 전파 covariance에서 **0.02102±0.00473**이었다. 반면 R 초과율은 **5.460±2.124% → 5.308±2.218%**로 방향이 유지됐다. 따라서 큰 covariance 이상값은 mean/SD를 크게 바꾸지만, 이 폭에서 관측된 위험예산 초과율 감소를 만들어낸 원인은 아니다. [전체 120-trial 보조 통계]({PREFIX}rollout_covariance/mean_sd.csv)와 [원본 이상값 검증]({PREFIX}covariance_outlier_audit.json)을 함께 보존했다.

## 논문에서의 해석 범위

- 같은 상태·covariance의 switching-boundary 결과는 **국소 constraint 민감도 차이**를 보여준다. 다른 궤적의 raw margin SD보다 방법의 수학적 구조와 직접 연결된다.
- 실제 궤적의 공통 margin·risk 초과율과 비선택 feature 위험 빈도는 **폐루프에서 관측된 조건부 차이**다. SF hard/ungated와 MF soft/gated 정책, 경로와 관측 길이의 차이가 함께 포함된다.
- 모든 지표가 MF에 유리하다고 가정하지 않는다. 폭별 반대 방향 결과, 극단값, domain 이탈, 없는 trial 통계도 보존했다. Source trajectory와 evaluator를 구분하고, 모델 위험도를 실제 충돌 확률로 부르지 않는다.
- 현재 실험은 localization error를 끈 설정이다. 실제 확률적 충돌 보장의 검증에는 별도의 오차 샘플링 실험이 필요하다.

## 파일과 재현

주 그림: [공통 margin·risk PDF]({PREFIX}figures/fig01_common_margin_risk.pdf) · [Boundary PDF]({PREFIX}figures/fig02_switch_boundary.pdf) · [Feature coverage PDF]({PREFIX}figures/fig03_feature_coverage.pdf).

보조 그림: [Stage 10 margin·risk]({PREFIX}figures/figS01_stage10_margin_risk.pdf) · [Domain·실제 주행]({PREFIX}figures/figS02_domain_and_execution.pdf) · [Temporal 변화]({PREFIX}figures/figS03_temporal_switch_changes.pdf) · [공통 시간]({PREFIX}figures/figS04_paired_exposure_sensitivity.pdf) · [Stage 10 boundary]({PREFIX}figures/figS05_stage10_switch_boundary.pdf) · [Risk mean·SD]({PREFIX}figures/figS06_risk_mean_and_dispersion.pdf).

숫자: [Trial별 지표]({PREFIX}trial_metrics.csv) · [폭별 mean·SD]({PREFIX}width_mean_sd.csv) · [Paired 차이]({PREFIX}paired_trial_differences.csv) · [Boundary trial 통계]({PREFIX}boundaries/per_trial_metrics.csv) · [Boundary 전체 사건]({PREFIX}boundaries/all_events.csv) · [분모/coverage]({PREFIX}coverage.csv).

재현 순서: `extract_constraint_stability.py` → `analyze_constraint_switch_boundaries.py` → `aggregate_constraint_stability.py` → `analyze_rollout_covariance_risk.py` → `plot_constraint_stability_paper.py` → `write_constraint_stability_report.py`. Python은 기존 `pp` 환경을 사용한다. 원본 120회 로그와 source snapshot은 수정하지 않는다. Figure caption에는 각 그림의 분모·단위·오차막대 정의를 기록했다.

<!-- constraint-stability:120trials_20261105:end -->
'''
    start='<!-- constraint-stability:120trials_20261105:start -->'
    end='<!-- constraint-stability:120trials_20261105:end -->'
    for path,section in [(ROOT/'benchmark_results.md',text),
                         (EXPERIMENT/'benchmark_results.md',text.replace(']('+f'benchmark_runs/{RUN}/',']('))]:
        old=path.read_text()
        if start in old:
            lo=old.index(start); hi=old.index(end)+len(end)
            old=old[:lo]+section.strip()+old[hi:]
        else: old=section.strip()+'\n\n---\n\n'+old
        path.write_text(old)
    print('Updated existing root and batch benchmark reports with full 120-trial stability analysis.')

if __name__=='__main__': main()