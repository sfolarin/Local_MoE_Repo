#!/usr/bin/env python3
"""
local_moe_ase.py

ASE calculator for the holdout-40 Local-MoE potential (stage C1b), built on the
neighbour-list TorchSOAP descriptor.  It is self-contained: it reads the three
checkpoints directly and does not import the training script.

    E = N*e_ref + sum_i e0_i + sum_i sum_k w_ik * delta_ik

Energy, forces and stress all come from the SAME scalar energy by autograd, so the
forces are exactly -dE/dR (the routing weights are NOT detached) and the stress is
(1/V) dE/d(strain), evaluated with a symmetric strain applied to positions and cell.

    positive stress = tensile (ASE convention); pressure = -trace(stress)/3

dtype
-----
  float64 (default)  networks, statistics and features in float64: matches the earlier
                     "float64" MD protocol; the descriptor is float64 in both modes.
  float32            networks and features in float32 (the precision of the benchmark).

The descriptor module is imported from a file path (the neighbour-list version).
"""
import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ase.calculators.calculator import Calculator, all_changes

SI_Z, O_Z = 14, 8
N_EXPERT_CHANNELS = 7


class MLP(nn.Module):
    """Same layer layout (and therefore state-dict keys) as the trainer's MLP."""

    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def import_file(path, name):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class LocalMoECalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, common_checkpoint, routing_checkpoint, c1b_checkpoint, torchsoap_module,
                 device="cuda", dtype="float64", **kwargs):
        super().__init__(**kwargs)
        if dtype not in ("float64", "float32"):
            raise ValueError("dtype must be float64 or float32")
        self.device = torch.device(device)
        self.dtype = torch.float64 if dtype == "float64" else torch.float32
        self.dtype_name = dtype
        self._z = None
        self.symbols = None
        self.onehot = None
        self.n_atoms = None
        self._load(Path(common_checkpoint), Path(routing_checkpoint), Path(c1b_checkpoint), Path(torchsoap_module))

    # ------------------------------------------------------------------ loading
    def _freeze(self, model):
        model = model.to(device=self.device, dtype=self.dtype).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    def _vec(self, t):
        return t.to(device=self.device, dtype=self.dtype)

    def _load(self, common_p, routing_p, c1b_p, soap_p):
        ck = torch.load(common_p.expanduser().resolve(), map_location="cpu", weights_only=False)
        common = MLP(int(ck["input_dim"]), [int(h) for h in ck["hidden_dims"]], 1, 0.0)
        common.load_state_dict(ck["model_state_dict"])
        self.common = self._freeze(common)
        self.cm, self.cs = self._vec(ck["feature_mean"]), self._vec(ck["feature_std"])
        self.e_ref = float(ck["e_ref_eV_per_atom"])

        rk = torch.load(routing_p.expanduser().resolve(), map_location="cpu", weights_only=False)
        if len(rk["expert_names"]) != N_EXPERT_CHANNELS:
            raise RuntimeError(f"routing checkpoint has {len(rk['expert_names'])} outputs, expected 7")
        routing = MLP(int(rk["input_dim"]), [int(h) for h in rk["hidden_dims"]], N_EXPERT_CHANNELS,
                      float(rk["dropout"]))
        routing.load_state_dict(rk["model_state_dict"])
        self.routing = self._freeze(routing)
        self.rm, self.rs = self._vec(rk["feature_mean"]), self._vec(rk["feature_std"])
        self.expert_names = [str(x) for x in rk["expert_names"]]

        bk = torch.load(c1b_p.expanduser().resolve(), map_location="cpu", weights_only=False)
        if [str(x) for x in bk["expert_names"]] != self.expert_names:
            raise RuntimeError("C1b expert order does not match the routing checkpoint")
        residual = MLP(int(bk["input_dim"]), [int(h) for h in bk["residual_hidden_dims"]], N_EXPERT_CHANNELS, 0.0)
        residual.load_state_dict(bk["residual_model_state_dict"])
        self.residual = self._freeze(residual)

        # The residual head uses the COMMON normalisation (as in training and in the benchmark).
        mod = import_file(soap_p, "local_moe_torchsoap_md")
        self.descriptor = mod.TorchSOAPDescriptor(
            mod.TorchSOAPConfig(species=("Si", "O"), r_cut=5.0, n_max=6, l_max=4)).to(self.device)
        for p in self.descriptor.parameters():
            p.requires_grad_(False)
        self._soap_sig = inspect.signature(self.descriptor.compute_per_center)

    def _set_species(self, numbers):
        z = np.asarray(numbers, dtype=np.int64)
        bad = sorted(set(z.tolist()) - {SI_Z, O_Z})
        if bad:
            raise ValueError(f"unexpected atomic numbers {bad}; the model covers Si and O only")
        self._z = z.copy()
        self.n_atoms = len(z)
        self.symbols = ["Si" if v == SI_Z else "O" for v in z.tolist()]
        zt = torch.as_tensor(z, device=self.device)
        self.onehot = torch.stack([(zt == SI_Z), (zt == O_Z)], dim=1).to(self.dtype)

    # ------------------------------------------------------------------ model
    def _terms(self, positions, cell):
        kwargs = {"positions": positions, "cell": cell, "symbols": self.symbols}
        if "masses" in self._soap_sig.parameters:  # older module signature; masses are not used by the descriptor
            kwargs["masses"] = np.zeros(self.n_atoms)
        soap = self.descriptor.compute_per_center(**kwargs)
        x = torch.cat([soap.to(self.dtype), self.onehot], dim=1)
        e0 = self.common((x - self.cm) / self.cs).squeeze(-1)
        w = torch.softmax(self.routing((x - self.rm) / self.rs), dim=1)
        raw = self.residual((x - self.cm) / self.cs)
        delta = raw - raw.mean(dim=1, keepdim=True)
        energy = (positions.new_tensor(self.n_atoms * self.e_ref)
                  + e0.sum().to(torch.float64) + torch.sum(w * delta).to(torch.float64))
        return energy, e0, w, delta

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, list(properties), system_changes)
        atoms = self.atoms
        if not np.all(atoms.pbc):
            raise ValueError("the descriptor is fully periodic; set pbc=True in all directions")
        if self._z is None or len(atoms) != len(self._z) or not np.array_equal(atoms.numbers, self._z):
            self._set_species(atoms.numbers)
        dev = self.device
        with torch.enable_grad():
            pos = torch.tensor(np.asarray(atoms.positions, dtype=np.float64), dtype=torch.float64, device=dev,
                               requires_grad=True)
            cell = torch.tensor(np.asarray(atoms.cell.array, dtype=np.float64), dtype=torch.float64, device=dev)
            eps = torch.zeros(3, 3, dtype=torch.float64, device=dev, requires_grad=True)
            strain = 0.5 * (eps + eps.T)
            f_mat = torch.eye(3, dtype=torch.float64, device=dev) + strain
            pos_s = pos @ f_mat.T            # row vectors: r' = (I + strain) r
            cell_s = cell @ f_mat.T
            energy, _, _, _ = self._terms(pos_s, cell_s)
            g_pos, g_eps = torch.autograd.grad(energy, [pos, eps], allow_unused=False)
        volume = abs(float(torch.det(cell)))
        s = 0.5 * (g_eps + g_eps.T) / volume
        s = s.detach().cpu().numpy()
        e = float(energy.detach().cpu())
        self.results = {
            "energy": e,
            "free_energy": e,
            "forces": -g_pos.detach().cpu().numpy(),
            "stress": np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]]),
        }

    # ------------------------------------------------------------------ analysis helper
    @torch.no_grad()
    def per_atom_terms(self, atoms):
        """e0_i (eV), routing weights w_ik and centred residuals delta_ik (eV) for a structure."""
        if self._z is None or len(atoms) != len(self._z) or not np.array_equal(atoms.numbers, self._z):
            self._set_species(atoms.numbers)
        pos = torch.tensor(np.asarray(atoms.positions, dtype=np.float64), dtype=torch.float64, device=self.device)
        cell = torch.tensor(np.asarray(atoms.cell.array, dtype=np.float64), dtype=torch.float64, device=self.device)
        _, e0, w, delta = self._terms(pos, cell)
        return {"e0": e0.cpu().numpy(), "w": w.cpu().numpy(), "delta": delta.cpu().numpy()}
