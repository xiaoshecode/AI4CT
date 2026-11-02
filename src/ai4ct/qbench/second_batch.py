"""Staged second-batch dense Gibbs models and a matched classical control.

The conditional Gibbs definition follows Amin et al. arXiv:1601.02036,
equations 45--47. This is exact finite-dimensional matrix-exponential training;
it is neither a bound-based loss nor a hardware imaginary-time algorithm.
"""
from __future__ import annotations

import math

SECOND_BATCH_IDS = ("gibbs_qbm_3q", "gibbs_separable_3q", "dressed_classical_5d",
                    "dressed_separable_5q", "mera_pca8_8q")


def build_second_batch_model(model_id, config):
    import torch
    dtype = getattr(torch, config["dtype"])
    nn = torch.nn

    if model_id == "dressed_classical_5d":
        class ClassicalDressed(nn.Module):
            def __init__(self):
                super().__init__()
                self.model_id, self.config = model_id, config
                self.register_parameter("angles", None)
                self.front = nn.Linear(17, 5)
                self.middle = nn.Sequential(nn.Linear(5, 4), nn.Tanh(), nn.Linear(4, 5), nn.Tanh())
                self.readout = nn.Linear(5, 2)
                self.to(dtype=dtype)
                self.model_spec = {
                    "kind": "classical_dressed_control", "preprocessing": "train_only_standard17",
                    "front": "Linear(17,5), pi*tanh", "middle": "Linear(5,4),Tanh,Linear(4,5),Tanh",
                    "readout": "Linear(5,2)", "middle_trainable_parameters": 49,
                    "quantum_reference_middle_parameters": 45, "total_trainable_parameters": 151,
                    "matching": "same front/readout dimensions; 4 additional middle parameters, not exact parameter matching",
                    "loss": "two-logit cross entropy", "shots": None,
                }

            def forward(self, x):
                return self.readout(self.middle(math.pi * torch.tanh(self.front(x))))
        return ClassicalDressed()

    if model_id not in ("gibbs_qbm_3q", "gibbs_separable_3q"):
        raise ValueError(f"Unknown second-batch dense model: {model_id}")

    class ConditionalGibbs(nn.Module):
        def __init__(self):
            super().__init__()
            self.model_id, self.config = model_id, config
            self.n_qubits = 3
            self.beta = 1.0
            self.fields = nn.Linear(17, 3, dtype=dtype)
            nn.init.normal_(self.fields.weight, mean=0., std=.05)
            nn.init.zeros_(self.fields.bias)
            self.angles = nn.Parameter(torch.full((3,), math.log(math.expm1(.5)), dtype=dtype))
            initial_j = torch.empty(3, dtype=dtype).normal_(mean=0., std=.05)
            if model_id == "gibbs_qbm_3q":
                self.couplings = nn.Parameter(initial_j)
            else:
                self.register_buffer("couplings", torch.zeros_like(initial_j))
            pauli_x = torch.tensor([[0., 1.], [1., 0.]], dtype=dtype)
            pauli_z = torch.diag(torch.tensor([1., -1.], dtype=dtype))
            identity = torch.eye(2, dtype=dtype)
            def embed(operator, wire):
                result = torch.ones((1, 1), dtype=dtype)
                for q in range(3):
                    result = torch.kron(result, operator if q == wire else identity)
                return result
            self.register_buffer("z_ops", torch.stack([embed(pauli_z, q) for q in range(3)]))
            self.register_buffer("x_ops", torch.stack([embed(pauli_x, q) for q in range(3)]))
            self.register_buffer("zz_ops", torch.stack([self.z_ops[i] @ self.z_ops[j] for i, j in ((0, 1), (0, 2), (1, 2))]))
            self.register_buffer("identity", torch.eye(8, dtype=dtype))
            self.model_spec = {
                "kind": "exact_conditional_Gibbs", "n_qubits": 3, "beta": 1.0,
                "output_wire": 0, "hidden_wires": [1, 2], "wire0_bit": "most_significant",
                "pass_bit": 1, "probability": "sum(diag(rho)[4:8]); rho=exp(-beta H)/trace(exp(-beta H))",
                "hamiltonian": "-sum_i (W_i z+b_i) Z_i -sum_(i<j) J_ij Z_i Z_j -sum_i softplus(g_i) X_i",
                "coupling_pairs": [[0, 1], [0, 2], [1, 2]],
                "couplings_trainable": model_id == "gibbs_qbm_3q",
                "input": "17 train-standardized classical values; no data qubits",
                "initialization": "W~Normal(0,.05), b=0; Gamma=.5; J~Normal(0,.05) or fixed0",
                "loss": "exact conditional negative log likelihood via normalized log probabilities",
                "gradient": "Torch autodiff of batched torch.linalg.matrix_exp; no commuting-gradient shortcut",
                "numerics": "detach maximum eigenvalue shift before matrix_exp; floor class probability1e-12 then renormalize",
                "source": "https://arxiv.org/abs/1601.02036; Eq45-47",
                "scope": "classically simulated exact Gibbs model, not hardware state preparation or imaginary-time approximation",
                "total_trainable_parameters": 60 if model_id == "gibbs_qbm_3q" else 57,
                "separable_caveat": "J=0 factorizes Gibbs state; hidden fields/transverse terms cancel from output marginal",
            }

        def hamiltonian(self, x):
            longitudinal = torch.einsum("bi,ijk->bjk", self.fields(x), self.z_ops)
            transverse = torch.einsum("i,ijk->jk", torch.nn.functional.softplus(self.angles), self.x_ops)
            coupling = torch.einsum("i,ijk->jk", self.couplings, self.zz_ops)
            return -longitudinal - transverse - coupling

        def density_matrix(self, x):
            negative_h = -self.beta * self.hamiltonian(x)
            shift = torch.linalg.eigvalsh(negative_h).amax(dim=-1).detach()
            boltzmann = torch.linalg.matrix_exp(negative_h - shift[:, None, None] * self.identity)
            trace = boltzmann.diagonal(dim1=-2, dim2=-1).sum(-1)
            return boltzmann / trace[:, None, None]

        def class_probabilities(self, x):
            diagonal = self.density_matrix(x).diagonal(dim1=-2, dim2=-1)
            probabilities = torch.stack([diagonal[:, :4].sum(-1), diagonal[:, 4:].sum(-1)], dim=-1)
            probabilities = probabilities.clamp_min(1e-12)
            return probabilities / probabilities.sum(-1, keepdim=True)

        def forward(self, x):
            return torch.log(self.class_probabilities(x))
    return ConditionalGibbs()
