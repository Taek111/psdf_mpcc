"""Validate paired runs and save trial CSV, summary JSON, and a Korean report."""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "safe"
METHODS = ("dcbf", "obca", "psdf")
SEEDS = tuple(range(42, 47))
FIELDS = (
    "trial", "seed", "method", "status", "navigation_success", "collision_free_success",
    "failure_reason", "final_time", "distance_to_goal", "completed_steps",
    "minimum_sampled_clearance", "contact_or_collision_samples", "solver_failure_steps",
    "initial_x", "initial_y", "initial_theta", "delta_x", "delta_y", "delta_theta",
    "sampling_attempts", "initial_clearance", "wall_seconds", "source_unchanged",
)


def main():
    records = {}
    for method in METHODS:
        for seed in SEEDS:
            path = ROOT / method / ("seed%d" % seed) / "result.json"
            if not path.exists():
                raise RuntimeError("Not complete: %s" % path)
            result = json.loads(path.read_text())
            assert result["source_unchanged"], path
            assert "capture_error" not in result, path
            assert result["seed"] == seed and result["method"] == method, path
            info = result["start_perturbation"]
            assert info["position_std"] == .02
            assert np.isclose(np.deg2rad(info["heading_std_deg"]), .05)
            assert info["initial_clearance"] >= .01
            records[method, seed] = result
    baseline_hashes = records[METHODS[0], SEEDS[0]]["source_sha256"]
    for result in records.values():
        assert result["source_sha256"] == baseline_hashes
    for seed in SEEDS:
        expected = records[METHODS[0], seed]["initial_pose"]
        for method in METHODS:
            result = records[method, seed]
            np.testing.assert_array_equal(result["initial_pose"], expected)
            trajectory_path = ROOT / method / ("seed%d" % seed) / "trajectory.npz"
            if trajectory_path.exists():
                with np.load(trajectory_path) as trajectory:
                    np.testing.assert_array_equal(trajectory["states"][0], expected)
                    assert len(trajectory["inputs"]) == result["completed_steps"]
                    assert np.isclose(trajectory["clearance"].min(), result["minimum_sampled_clearance"])
    rows = []
    for seed in SEEDS:
        for method in METHODS:
            result = records[method, seed]
            row = {field: result.get(field) for field in FIELDS}
            pose = result["initial_pose"]
            info = result["start_perturbation"]
            row.update(initial_x=pose[0], initial_y=pose[1], initial_theta=pose[2],
                       delta_x=info["delta_pose"][0], delta_y=info["delta_pose"][1],
                       delta_theta=info["delta_pose"][2], sampling_attempts=info["attempts"],
                       initial_clearance=info["initial_clearance"])
            rows.append(row)
    with (ROOT / "trials.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for method in METHODS:
        selected = [records[method, seed] for seed in SEEDS]
        count = sum(result["navigation_success"] for result in selected)
        safe_count = sum(result["collision_free_success"] for result in selected)
        summary[method] = dict(
            trials=len(selected), successes=count, success_rate=count/len(selected),
            collision_free_successes=safe_count, collision_free_success_rate=safe_count/len(selected),
            failures=[dict(seed=r["seed"], status=r["status"], reason=r.get("failure_reason"))
                      for r in selected if not r["navigation_success"]])
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Maze / Rectangle / DD 초기 자세 섭동 실험",
        "",
        "`test_nmpc.run_single_test()`를 호출해 각 방법을 5회 실행했다. 각 trial은 별도 프로세스에서 실행했다.",
        "",
        "- 환경: maze, rectangle (0.15 × 0.09 m), differential_drive, A* + Line of Sight.",
        "- 초기 자세: (0.075 m, 0.795 m, −π/2 rad).",
        "- 제안 가우시안 분포: 평균 0, 표준편차 (0.02 m, 0.02 m, 0.05 rad), 각 성분 독립.",
        "- 기존 sampling 규칙 유지: 성분별 ±3σ, 로봇 전체 footprint의 장애물/지도 경계 여유 거리 ≥ 0.01 m. 부적합 표본 재추출.",
        "- 따라서 최종 표본은 안전 조건으로 제한된 가우시안이며, 채택 표본의 실제 표준편차가 위 값과 같다는 의미는 아니다.",
        "- 공통 seed: 42, 43, 44, 45, 46. 같은 seed의 초기 자세는 세 방법에서 완전히 동일함을 검증했다.",
        "- 초기 실제 자세만 섭동했다. 주행 중 localization/measurement noise는 비활성화했다.",
        "- 제한 시간: simulation time 60 s (test_nmpc --single 기본값), Δt = 0.1 s. 조기 정체 종료 조건은 추가하지 않았다.",
        "- 성공: 기존 test_nmpc의 status == success. A* 경로가 2D이므로 경로 종점과의 위치 오차 ≤ 0.01 m로 판정하며, 최종 heading 오차는 적용되지 않는다.",
        "- 보조 검증: 초기 상태와 각 0.1 s 상태에서 footprint의 장애물/지도 경계 clearance를 검사했다. 연속 시간 충돌 검증은 아니다.",
        "- 각 optimizer의 현재 파라미터를 그대로 사용했다. 소스 SHA-256이 모든 실행 전후 동일함을 확인했다.",
        "- 애니메이션/플롯 생성을 비활성화했다. 로그와 trajectory.npz 및 CSV는 저장했다.",
        "",
        "| 방법 | 도달 성공 | 성공률 | 충돌 없이 도달 |",
        "|---|---:|---:|---:|",
    ]
    for method, data in summary.items():
        lines.append("| %s | %d/5 | %.0f%% | %d/5 |" % (
            method.upper(), data["successes"], 100*data["success_rate"], data["collision_free_successes"]))
    lines += ["", "## Trial별 결과", "",
              "| Trial | Seed | DCBF | OBCA | PSDF |",
              "|---|---:|---|---|---|"]
    for index, seed in enumerate(SEEDS, 1):
        cells = []
        for method in METHODS:
            r = records[method, seed]
            duration = r.get("final_time", r.get("captured_final_time", 0.))
            cells.append("%s (%.1f s)" % ("성공" if r["navigation_success"] else r["status"], duration))
        lines.append("| %d | %d | %s |" % (index, seed, " | ".join(cells)))
    lines += ["", "## 채택한 초기 자세", "",
              "| Seed | x [m] | y [m] | θ [rad] | 여유 거리 [m] | 추출 횟수 |",
              "|---:|---:|---:|---:|---:|---:|"]
    for seed in SEEDS:
        info = records[METHODS[0], seed]["start_perturbation"]
        lines.append("| %d | %.8f | %.8f | %.8f | %.8f | %d |" % (
            seed, *info["initial_pose"], info["initial_clearance"], info["attempts"]))
    lines += ["", "5회만으로 추정한 경험적 성공률이며, 한 trial이 20%p에 해당한다.", "",
              "재실행: WSL에서 `/usr/bin/python3 run_experiment.py --distribution safe --workers 6`.",
              "기존 result.json이 있는 trial은 건너뛴다. 실행 환경에는 Python 3.8 solver 패키지가 설치되어 있어 BooleanOptionalAction만 Python 3.10 argparse에서 가져왔다.", ""]
    (ROOT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
