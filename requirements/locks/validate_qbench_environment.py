"""Synthetic SDK/environment validation; no patient data or project models."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata as md
import io
import json
import platform
import sys
from pathlib import Path
import numpy as np
import pennylane as qml
import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
torch.use_deterministic_algorithms(True)


def reference(x, weights):
    def ry(a):
        c, s = np.cos(a / 2), np.sin(a / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    def rx(a):
        c, s = np.cos(a / 2), np.sin(a / 2)
        return np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)
    eye = np.eye(2)
    c01 = np.eye(4)[[0, 1, 3, 2]]
    c10 = np.eye(4)[[0, 3, 2, 1]]
    state = np.array([1, 0, 0, 0], dtype=complex)
    for gate in (np.kron(ry(x[0]), eye), np.kron(eye, ry(x[1])),
                 c01, np.kron(ry(weights[0]), eye), np.kron(eye, rx(weights[1])), c10):
        state = gate @ state
    probabilities = abs(state)**2
    return np.array([probabilities @ np.array([1, 1, -1, -1]),
                     probabilities @ np.array([1, -1, 1, -1])])


def qnode_for(backend):
    device = qml.device(backend, wires=2, shots=None, seed=42)
    method = "backprop" if backend == "default.qubit" else "adjoint"
    @qml.qnode(device, interface="torch", diff_method=method)
    def circuit(inputs, weights):
        qml.RY(inputs[..., 0], 0)
        qml.RY(inputs[..., 1], 1)
        qml.CNOT([0, 1])
        qml.RY(weights[0], 0)
        qml.RX(weights[1], 1)
        qml.CNOT([1, 0])
        # TorchLayer concatenates list measurements along the feature axis.
        return [qml.expval(qml.Z(0)), qml.expval(qml.Z(1))]
    return circuit


def make_model(backend):
    torch.manual_seed(123)
    layer = qml.qnn.TorchLayer(qnode_for(backend), {"weights": (2,)},
                               init_method={"weights": torch.tensor([0.31, -0.47])})
    return torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Tanh(), layer,
                               torch.nn.Linear(2, 2)).double()


def check_backend(backend):
    circuit = qnode_for(backend)
    x = torch.tensor([0.23, -0.61], requires_grad=True)
    weights = torch.tensor([0.31, -0.47], requires_grad=True)
    output = torch.stack(circuit(x, weights))
    expected = reference(x.detach().numpy(), weights.detach().numpy())
    forward_error = float(np.max(abs(output.detach().numpy() - expected)))
    assert forward_error < 1e-10
    factors = torch.tensor([0.7, -0.2])
    gradients = torch.autograd.grad(output @ factors, (x, weights))
    actual_gradient = torch.cat(gradients).detach().numpy()
    vector = np.r_[x.detach().numpy(), weights.detach().numpy()]
    numeric = np.zeros(4)
    step = 1e-6
    for i in range(4):
        high, low = vector.copy(), vector.copy()
        high[i] += step
        low[i] -= step
        numeric[i] = ((reference(high[:2], high[2:]) - reference(low[:2], low[2:])) @ factors.numpy()) / (2*step)
    gradient_error = float(np.max(abs(actual_gradient - numeric)))
    assert gradient_error < 1e-7
    assert np.isfinite(actual_gradient).all() and np.all(abs(actual_gradient) > 1e-6)

    model = make_model(backend)
    data = torch.tensor([[.1, .2, -.3], [.6, -.2, .1], [-.7, .2, .8]])
    target = torch.tensor([0, 1, 0])
    batch_output = model(data)
    single_output = torch.cat([model(row.unsqueeze(0)) for row in data])
    batch_error = float((batch_output-single_output).abs().max().detach())
    assert batch_error < 1e-10
    loss = torch.nn.functional.cross_entropy(batch_output, target)
    loss.backward()
    norms = {name: float(param.grad.norm()) for name, param in model.named_parameters()}
    assert all(np.isfinite(norm) and norm > 1e-9 for norm in norms.values())
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = make_model(backend)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    reload_error = float((model(data)-restored(data)).abs().max().detach())
    assert reload_error == 0.0

    # Test optimizer state and epoch-boundary recovery with deterministic synthetic steps.
    def take_steps(net, optimizer, count):
        for _ in range(count):
            optimizer.zero_grad()
            torch.nn.functional.cross_entropy(net(data), target).backward()
            optimizer.step()
    original = make_model(backend)
    optimizer = torch.optim.Adam(original.parameters(), lr=.01)
    take_steps(original, optimizer, 2)
    checkpoint = copy.deepcopy({"model": original.state_dict(), "optimizer": optimizer.state_dict(),
                                 "torch_rng": torch.get_rng_state()})
    take_steps(original, optimizer, 3)
    resumed = make_model(backend)
    resumed.load_state_dict(checkpoint["model"])
    resume_optimizer = torch.optim.Adam(resumed.parameters(), lr=.01)
    resume_optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["torch_rng"])
    take_steps(resumed, resume_optimizer, 3)
    resume_error = max(float((a-b).abs().max().detach()) for a,b in zip(original.parameters(),resumed.parameters()))
    assert resume_error == 0.0
    return {"status": "passed", "forward_max_abs_error": forward_error,
            "gradient_max_abs_error": gradient_error, "parameter_gradient_norms": norms,
            "batch_max_abs_error": batch_error, "reload_max_abs_error": reload_error,
            "resume_parameter_max_abs_error": resume_error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"status": "passed", "python": sys.version, "executable": sys.executable,
              "prefix": sys.prefix, "base_prefix": sys.base_prefix,
              "system_site_packages": False,
              "platform": platform.platform(), "torch_cuda_runtime": torch.version.cuda,
              "versions": {k: md.version(k) for k in ["torch", "pennylane", "pennylane-lightning", "numpy", "scipy", "scikit-learn"]},
              "scope": "Synthetic CPU forward, independent matrix and finite-difference gradients, TorchLayer batch, checkpoint and Adam epoch-boundary recovery; no clinical training",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "backends": {b: check_backend(b) for b in ["default.qubit", "lightning.qubit"]},
              "unavailable": {"lightning.gpu": "Linux-only wheel; native Windows profile",
                              "cudaq": "Native Windows unsupported; WSL not installed"}}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
