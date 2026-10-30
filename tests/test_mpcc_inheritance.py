"""MPCC sharing contracts without acados code generation or navigation trials.

Prepared for a later requested test run; model algebra and device failures can
be checked without allocating CUDA tensors. Actual PSDF CUDA integration stays
in the explicitly gated geometry test.
"""

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import casadi as ca
import numpy as np

from control.cc_ecs_optimizer import CCECSOptimizer, CCECSOptimizerParam
from control.cc_mf_optimizer import CCMFOptimizer, CCMFOptimizerParam
from control.cc_sf_optimizer import CCSFOptimizer, CCSFOptimizerParam
from control.mpcc_optimizer import MPCCOptimizer, MPCCOptimizerParam
from models.geometry_utils import RectangleRegion
from sim.simulation_mpc import simulation_mpc
from utils.spline_path import build_cubic_spline_path_data


MPCC_TYPES = (
    (MPCCOptimizer, MPCCOptimizerParam),
    (CCMFOptimizer, CCMFOptimizerParam),
    (CCSFOptimizer, CCSFOptimizerParam),
)


class MPCCInheritanceTest(unittest.TestCase):
    def make_optimizer(self, optimizer_type, param_type):
        optimizer = optimizer_type()
        self.addCleanup(optimizer.cleanup)
        optimizer.param = param_type()
        return optimizer

    def test_tracking_is_shared_without_merging_method_identities(self):
        self.assertIn(MPCCOptimizer, CCSFOptimizer.__bases__)
        self.assertIn(MPCCOptimizer, CCMFOptimizer.__bases__)
        self.assertFalse(issubclass(CCMFOptimizer, CCSFOptimizer))
        self.assertTrue(issubclass(CCECSOptimizer, CCSFOptimizer))
        self.assertIsNot(CCMFOptimizerParam, CCSFOptimizerParam)
        for param_type in (CCSFOptimizerParam, CCECSOptimizerParam, CCMFOptimizerParam):
            self.assertEqual(param_type.__bases__, (MPCCOptimizerParam,))
        self.assertIs(CCSFOptimizer._build_mpcc_cost, MPCCOptimizer._build_mpcc_cost)
        self.assertIs(CCMFOptimizer._build_mpcc_cost, MPCCOptimizer._build_mpcc_cost)

    def test_common_noise_defaults_match_current_mf_tuning_but_ecs_stays_noise_free(self):
        expected = {
            "alpha_f": 0.0012, "alpha_v": 0.00024, "alpha_kappa": 0.0003,
            "beta_v": 0.012, "beta_kappa": 0.0048, "beta_omega": 0.0048,
        }
        for param_type in (CCMFOptimizerParam, CCSFOptimizerParam):
            with self.subTest(method=param_type.__name__):
                param = param_type()
                for name, value in expected.items():
                    self.assertAlmostEqual(getattr(param, name), value, msg=name)
                self.assertEqual((param.q_f0, param.q_l0, param.q_psi0), (0, 0, 0))
                self.assertEqual(param.integrator_type, "DISCRETE")
        ecs = CCECSOptimizerParam()
        for name in expected:
            self.assertEqual(getattr(ecs, name), 0.0, msg=name)
        self.assertEqual(MPCCOptimizerParam().integrator_type, "ERK")

    def test_param_initializers_inherit_shared_defaults_and_keep_arrays_independent(self):
        params = [param_type() for _, param_type in MPCC_TYPES]
        params.append(CCSFOptimizerParam())
        for param in params:
            self.assertEqual((param.horizon, param.tf), (20, 2.0))
            np.testing.assert_array_equal(param.mat_Qe, np.diag([1.0, 10.0]))
            np.testing.assert_array_equal(param.mat_Re, np.diag([1.0, 0.01]))
        params[0].mat_Qe[0, 0] = 123.0
        params[1].mat_Re[0, 0] = 456.0
        for param in params[1:]:
            self.assertEqual(param.mat_Qe[0, 0], 1.0)
        for param in (params[0], params[2], params[3]):
            self.assertEqual(param.mat_Re[0, 0], 1.0)
        for param in params[1:]:
            for field in ("use_soft_constraint", "slack_weight", "detection_frequency"):
                self.assertEqual(getattr(param, field), getattr(params[0], field), msg=field)
        self.assertFalse(hasattr(params[1], "use_risk_margin"))
        self.assertFalse(hasattr(params[2], "enable_backup_solver"))
        self.assertFalse(hasattr(params[2], "mf_active_terminal"))

    def test_cc_initializers_inherit_changed_shared_defaults(self):
        original_init = MPCCOptimizerParam.__init__

        def tuned_defaults(param):
            original_init(param)
            param.q_s = 3.25
            param.qp_solver_iter_max = 71

        with patch.object(MPCCOptimizerParam, "__init__", tuned_defaults):
            for param_type in (CCSFOptimizerParam, CCECSOptimizerParam, CCMFOptimizerParam):
                with self.subTest(parameter=param_type.__name__):
                    param = param_type()
                    self.assertEqual(param.q_s, 3.25)
                    self.assertEqual(param.qp_solver_iter_max, 71)
                    self.assertEqual(param.integrator_type, "DISCRETE")

    def test_removed_sf_window_keys_are_ignored_by_config_loader(self):
        param = CCSFOptimizerParam()
        output = io.StringIO()
        with redirect_stdout(output):
            simulation_mpc._apply_param_overrides(
                param,
                {"risk_margin_active_start_step": 2, "risk_margin_active_end_step": 3},
                "CC-SF",
            )
        for key in ("risk_margin_active_start_step", "risk_margin_active_end_step"):
            self.assertFalse(hasattr(param, key))
            self.assertIn(key, output.getvalue())

    def test_shared_cost_keeps_geometric_values_and_augmented_state_derivatives(self):
        state = np.array([1.2, -0.4, 0.3, 0.7])
        controls = np.array([0.35, -0.25, 0.3])
        path = np.array([0.8, -0.1, 0.0, 2.0, 0.5])
        reference_derivatives = None
        for optimizer_type, param_type in MPCC_TYPES:
            with self.subTest(method=optimizer_type.__name__):
                optimizer = self.make_optimizer(optimizer_type, param_type)
                param = optimizer.param
                model = optimizer.create_model(param)
                values = np.zeros(int(model.p.numel()))
                values[:5] = path
                full_state = state
                if int(model.x.numel()) == 8:
                    full_state = np.r_[state, np.log([0.01, 0.02, 0.03]), -0.004]
                    values[5] = -0.4  # Curvature must not alter the MPCC cost.
                costs = ca.vertcat(model.cost_expr_ext_cost, model.cost_expr_ext_cost_e)
                evaluate = ca.Function(
                    f"shared_cost_{optimizer_type.__name__}",
                    [model.x, model.u, model.p],
                    [costs, ca.jacobian(costs, model.x), ca.jacobian(costs, model.u)],
                )
                actual, dx, du = evaluate(full_state, controls, values)
                # Vertical reference: x offset is contour error, y offset is lag.
                tangent_length = 2.0 / (2.0 + param.tangent_reg_delta)
                contour = (state[0] - path[0]) * tangent_length
                lag = (path[1] + 0.2 * tangent_length - state[1]) * tangent_length
                tracking = contour**2 + 10.0 * lag**2 + 2.0 * 0.2**2
                stage = tracking + 0.35**2 + 0.01 * 0.25**2 + 0.625 * 0.3**2 - 0.3
                np.testing.assert_allclose(np.asarray(actual).ravel(), [stage, 5.0 * tracking])
                dx, du = np.asarray(dx), np.asarray(du)
                if reference_derivatives is None:
                    reference_derivatives = (dx, du)
                else:
                    np.testing.assert_allclose(dx[:, :4], reference_derivatives[0], atol=1e-14)
                    np.testing.assert_allclose(du, reference_derivatives[1], atol=1e-14)
                    np.testing.assert_array_equal(dx[:, 4:], np.zeros((2, 4)))

    def test_shared_path_helper_keeps_cc_layout_without_evaluating_curvature(self):
        path = build_cubic_spline_path_data(
            np.array([[0.0, 0.0], [0.5, 0.2], [1.0, 0.0]])
        )
        progress = float(path["s_breaks"][1])
        reference = None
        for optimizer_type, param_type in MPCC_TYPES:
            optimizer = self.make_optimizer(optimizer_type, param_type)
            optimizer.set_reference_trajectory(path)
            with patch(
                "utils.spline_path.curvature",
                side_effect=AssertionError("unused curvature must not be evaluated"),
            ):
                frame = optimizer._build_stage_path_parameter(progress)
            if optimizer_type is MPCCOptimizer:
                self.assertEqual(frame.shape, (5,))
                reference = frame
            else:
                self.assertEqual(frame.shape, (6,))
                np.testing.assert_allclose(frame[:5], reference)
                self.assertEqual(frame[5], 0.0)

    def test_progress_prediction_reads_explicit_backup_solver(self):
        class ProgressSolver:
            def __init__(self, rates):
                self.rates = rates

            def get(self, stage, field):
                if field == "x":
                    return np.array([0.0, 0.0, 0.0, np.nan])
                if field == "u":
                    return np.array([0.0, 0.0, self.rates[stage]])
                raise KeyError(field)

        for optimizer_type, param_type in MPCC_TYPES:
            optimizer = self.make_optimizer(optimizer_type, param_type)
            optimizer.N = 3
            optimizer.param.tf = 0.3
            optimizer.solver = ProgressSolver([0.0, 0.0, 0.0])
            backup = ProgressSolver([0.2, 0.4, 1.0])
            np.testing.assert_allclose(
                optimizer._predict_stage_s_values(0.1, 0.0, 0.21),
                [0.1, 0.1, 0.1, 0.1],
            )
            np.testing.assert_allclose(
                optimizer._predict_stage_s_values(0.1, 0.0, 0.21, solver=backup),
                [0.1, 0.12, 0.16, 0.21],
            )


class MPCCResourceOwnershipTest(unittest.TestCase):
    def test_uninitialized_cleanup_preserves_existing_generic_and_mpcc_artifacts(self):
        with (
            patch("utils.acados_codegen.TemporaryDirectory") as allocate,
            patch("shutil.rmtree") as rmtree,
        ):
            optimizer = MPCCOptimizer()
            optimizer.cleanup()
            allocate.assert_not_called()
            rmtree.assert_not_called()

    def test_cleanup_removes_owned_workspace_even_if_ocp_path_changes(self):
        fake_ocp = SimpleNamespace(solver_options=SimpleNamespace())
        fake_solver = object()
        optimizer = MPCCOptimizer()
        self.addCleanup(optimizer.cleanup)
        with (
            TemporaryDirectory() as unrelated,
            patch("control.mpcc_optimizer.AcadosOcpSolver", return_value=fake_solver) as create,
        ):
            sentinel = Path(unrelated) / "acados_ocp.json"
            sentinel.write_text("unrelated experiment", encoding="utf-8")
            optimizer.ocp = fake_ocp
            optimizer.create_solver()
            json_path = Path(optimizer.json_filename)
            workspace = json_path.parent
            create.assert_called_once_with(fake_ocp, json_file=str(json_path))
            self.assertIs(optimizer.solver, fake_solver)
            self.assertEqual(fake_ocp.code_export_directory, optimizer.code_export_directory)
            json_path.write_text("{}", encoding="utf-8")
            # Ownership is independent of the OCP's later configured path.
            fake_ocp.code_export_directory = unrelated
            optimizer.cleanup()
            optimizer.cleanup()
            self.assertFalse(workspace.exists())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unrelated experiment")


class MPCCSolverTimingTest(unittest.TestCase):
    def test_solver_time_excludes_reference_preparation_and_progress_reads(self):
        optimizer = MPCCOptimizer()
        self.addCleanup(optimizer.cleanup)
        optimizer.param = MPCCOptimizerParam()
        optimizer.param.horizon = optimizer.N = 2
        optimizer.param.tf = 0.2
        optimizer.state = SimpleNamespace(_x=np.zeros(3), _u=np.zeros(2))
        optimizer.create_model(optimizer.param)
        clock = SimpleNamespace(value=0.0)

        class Solver:
            status = 0

            def __init__(self):
                self.states = np.zeros((3, 4))
                self.inputs = np.zeros((2, 3))

            def get(self, stage, field):
                clock.value += 2.0
                return (self.states if field == "x" else self.inputs)[stage].copy()

            def set(self, stage, field, value):
                clock.value += 4.0
                if field == "x":
                    self.states[stage] = value
                elif field == "u":
                    self.inputs[stage] = value

            def solve(self):
                clock.value += 0.125
                return self.status

        optimizer.solver = Solver()
        with (
            patch("control.mpcc_optimizer.time.perf_counter", side_effect=lambda: clock.value),
            redirect_stdout(io.StringIO()),
        ):
            solution = optimizer.solve_nlp()
        self.assertEqual(solution.stats()["return_status"], "success")
        self.assertEqual(optimizer.solver_times, [0.125])
        self.assertGreater(clock.value, 10.0)


class MPCCCudaPolicyTest(unittest.TestCase):
    def test_device_validation_rejects_cpu_and_does_not_fall_back(self):
        with patch("control.mpcc_optimizer.torch.cuda.is_available", return_value=True):
            with self.assertRaisesRegex(ValueError, "CUDA"):
                MPCCOptimizer._require_cuda_device("cpu")
            self.assertEqual(MPCCOptimizer._require_cuda_device("cuda:0"), "cuda:0")
        with patch("control.mpcc_optimizer.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA.*unavailable"):
                MPCCOptimizer._require_cuda_device()
            self.assertEqual(CCMFOptimizerParam().augmented_psdf_device, "cuda")
            self.assertEqual(CCECSOptimizerParam().augmented_psdf_device, "cuda")

    def test_public_psdf_initializers_fail_before_tensor_allocation(self):
        system = SimpleNamespace(_geometry=SimpleNamespace(_geometries=[
            SimpleNamespace(_region=RectangleRegion(-0.05, 0.05, -0.03, 0.03)),
        ]))
        for optimizer_type, param_type in (*MPCC_TYPES, (CCECSOptimizer, CCECSOptimizerParam)):
            optimizer = optimizer_type()
            self.addCleanup(optimizer.cleanup)
            optimizer.param = param_type()
            if optimizer_type is CCMFOptimizer:
                initialize = optimizer.initialize_augmented_psdf
                arguments = (system,)
            else:
                initialize = optimizer.initialize_ped_model
                arguments = (system, [])
            for device, error in (("cpu", ValueError), ("cuda", RuntimeError)):
                with self.subTest(method=optimizer_type.__name__, device=device):
                    with (
                        patch("control.mpcc_optimizer.torch.cuda.is_available", return_value=False),
                        patch("torch.tensor") as tensor,
                        patch("torch.as_tensor") as as_tensor,
                    ):
                        with self.assertRaisesRegex(error, "CUDA"):
                            initialize(*arguments, device=device)
                        tensor.assert_not_called()
                        as_tensor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
