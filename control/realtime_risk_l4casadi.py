from typing import Callable, Text, Tuple, Union

import numpy as np
import torch

try:
    import torch.func as functorch
except ImportError:
    import functorch

from l4casadi.realtime import RealTimeL4CasADi


def _batched_jacobian_with_aux(
    func: Callable[[torch.Tensor], Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
    inputs: torch.Tensor,
    create_graph: bool = False,
):
    if inputs.shape[0] == 1:
        vmap_randomness = "same"
    else:
        # Matches l4casadi.realtime.sensitivities.
        vmap_randomness = "same"

    if not create_graph:
        with torch.no_grad():
            return functorch.vmap(
                functorch.jacrev(func, has_aux=True),
                randomness=vmap_randomness,
            )(inputs[:, None])

    return functorch.vmap(
        functorch.jacrev(func, has_aux=True),
        randomness=vmap_randomness,
    )(inputs[:, None])


class RiskAwareRealTimeL4CasADi(RealTimeL4CasADi):
    """Numeric helper that returns the usual Taylor params plus risk tensors."""

    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        approximation_order: int = 1,
        device: Union[torch.device, Text] = "cpu",
        name: Text = "rt_l4casadi_f",
    ):
        super().__init__(
            model=model,
            approximation_order=approximation_order,
            device=device,
            name=name,
        )

    def _get_params_and_risk(self, a_t: torch.Tensor):
        if len(a_t.shape) == 1:
            a_t = a_t.unsqueeze(0)

        if self.order != 1:
            raise NotImplementedError("get_params_with_risk currently supports approximation_order=1 only.")
        if not hasattr(self.model, "forward_with_risk"):
            raise AttributeError("Model must implement forward_with_risk() to expose risk tensors.")

        def _phi_with_risk(inp: torch.Tensor):
            phi, risk = self.model.forward_with_risk(inp)
            if phi.ndim == 1:
                phi = phi.unsqueeze(1)
            return phi, (phi, risk)

        df_a, (f_a, risk_a) = _batched_jacobian_with_aux(_phi_with_risk, a_t)
        params = [
            a_t.cpu().numpy(),
            f_a.cpu().numpy(),
            df_a.transpose(-2, -1).cpu().numpy(),
        ]
        return params, risk_a.squeeze(1).cpu().numpy()

    def get_params_with_risk(self, a: Union[np.ndarray, torch.Tensor]):
        a_t = torch.tensor(a).float().to(self.device)
        params, risk = self._get_params_and_risk(a_t)

        if len(params) == 0:
            return np.array([]), risk
        if len(a.shape) > 1:
            return np.hstack([p.reshape(p.shape[0], -1) for p in params]), risk
        return np.hstack([p.flatten() for p in params]), risk[0]
