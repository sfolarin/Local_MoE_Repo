#!/usr/bin/env python3
"""
moe_persistent_calculator.py

Single-run MoE calculator using persistent LAMMPS Python instances.
All experts run simultaneously in the same Python process — no subprocess
spawning, no temporary files. Forces are combined at every MD step.

This is the "Option 2" implementation that allows users to run MoE MD
as a single simulation, just like any other interatomic potential.

Key features:
  - N persistent lammps instances (one per expert)
  - Forces/energies gathered at every step via gather_atoms()
  - Dynamic gate weights updated every update_every steps
  - EMA smoothing to prevent force discontinuities
  - Full ASE Calculator interface — works with any ASE MD driver
  - Weight history recorded for publication figures

Usage:
  from moe_persistent_calculator import MoEPersistentCalculator
  calc = MoEPersistentCalculator(
      gate_ckpt  = "gate_18feat_7exp_with_melt_fast/gate_18feat_7exp_melt_fast.pt",
      pot_dir    = "/path/to/potentials",
      lmp_lib    = "/path/to/liblammps.so.0",
      lmp_python = "/path/to/lammps/python",
  )
  atoms.calc = calc

Run MD:
  python3 moe_persistent_calculator.py \
    --gate_ckpt gate_18feat_7exp_with_melt_fast/gate_18feat_7exp_melt_fast.pt \
    --structure  cristobalite_3x3x3.data \
    --T_start    300 --T_end 4000 --heat_rate 1.7 \
    --output_dir moe_single_run_output
"""

import os
import sys
import json
import time
import argparse
import ctypes
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ── LAMMPS library setup ──────────────────────────────────────────────────────
LMP_LIB    = "<path>/liblammps.so.0"
LMP_PYTHON = "<path>/python"
SI_MASS    = 28.0855
O_MASS     = 15.9994

# ── SOAP descriptor (auto-used when checkpoint has descriptor_type="SOAP") ────
def _get_soap_descriptor(ckpt):
    """Build SOAPDescriptor from checkpoint params."""
    try:
        from dscribe.descriptors import SOAP
        class SOAPDescriptor:
            def __init__(self, r_cut, n_max, l_max):
                self.soap = SOAP(
                    species  = ["Si","O"],
                    r_cut    = r_cut,
                    n_max    = n_max,
                    l_max    = l_max,
                    average  = "off",
                    sparse   = False,
                    periodic = True,
                )
                self.soap_dim = self.soap.get_number_of_features()

            def compute(self, atoms, is_zeolite=False):
                import numpy as np
                masses = atoms.get_masses()
                syms   = np.array(["Si" if abs(m-28.0855)<1.0 else "O"
                                   for m in masses])
                atoms.set_chemical_symbols(syms)
                si_idx = np.where(syms=="Si")[0]
                o_idx  = np.where(syms=="O")[0]
                desc   = self.soap.create(atoms)
                si_soap = desc[si_idx].mean(0) if len(si_idx)>0                           else np.zeros(self.soap_dim)
                o_soap  = desc[o_idx].mean(0)  if len(o_idx)>0                            else np.zeros(self.soap_dim)
                volume       = atoms.get_volume()
                mass_amu     = atoms.get_masses().sum()
                density      = mass_amu*1.66054e-24/(volume*1e-24)
                vol_per_atom = volume/len(atoms)
                scalar = np.array([vol_per_atom, density,
                                   float(is_zeolite)], dtype=np.float32)
                return np.concatenate([si_soap, o_soap, scalar]).astype(np.float32)

        return SOAPDescriptor(
            r_cut = ckpt.get("soap_rcut", 5.0),
            n_max = ckpt.get("soap_nmax", 6),
            l_max = ckpt.get("soap_lmax", 4),
        )
    except ImportError:
        raise ImportError("dscribe required for SOAP gate. "
                          "Install: pip install dscribe")



# ── Expert potential definitions ──────────────────────────────────────────────
# Each expert: LAMMPS commands to set up the potential
EXPERT_SETUP = {
    "ML-BKS": {
        "atom_style": "charge",
        "charges":    {"Si": 1.785, "O": -0.8925},
        "pair_cmds": [
            "pair_style buck/coul/long 11.0",
            "pair_coeff 1 1  3891.0 0.299 529.1",
            "pair_coeff 2 2   737.1 0.353   8.074",
            "pair_coeff 1 2 16555.0 0.158   4.95",
            "kspace_style pppm 1e-6",
        ],
    },
    "Original_BKS": {
        "atom_style": "charge",
        "charges":    {"Si": 2.4, "O": -1.2},
        "pair_cmds": [
            "pair_style buck/coul/long 11.0",
            "pair_coeff 1 1 0.0 1.0 0.0",
            "pair_coeff 2 2 1388.0 0.362 175.0",
            "pair_coeff 1 2 18000.0 0.205 133.0",
            "kspace_style pppm 1e-6",
        ],
    },
    "Chik_BKS": {
        "atom_style": "charge",
        "charges":    {"Si": 1.91, "O": -0.955},
        "pair_cmds": [
            "pair_style buck/coul/long 11.0",
            "pair_coeff 1 1 3150.0 0.351 627.0",
            "pair_coeff 2 2 660.0 0.386 26.84",
            "pair_coeff 1 2 27029.0 0.194 148.0",
            "kspace_style pppm 1e-6",
        ],
    },
    "PSO1_BKS": {
        "atom_style": "charge",
        "charges":    {"Si": 1.79, "O": -0.895},
        "pair_cmds": [
            "pair_style buck/coul/long 11.0",
            "pair_coeff 1 1     0.0  1.0    0.0",
            "pair_coeff 2 2 883.0 0.36 50.4",
            "pair_coeff 1 2 28954.0 0.185 111.9",
            "kspace_style pppm 1e-6",
        ],
    },
    "PSO2_BKS": {
        "atom_style": "charge",
        "charges":    {"Si": 2.35, "O": -1.175},
        "pair_cmds": [
            "pair_style buck/coul/long 11.0",
            "pair_coeff 1 1     0.0 1.0 1.0",
            "pair_coeff 2 2  1522.0  0.366  17.4",
            "pair_coeff 1 2 44409.0  0.187 201.0",
            "kspace_style pppm 1e-6",
        ],
    },
    "Tersoff": {
        "atom_style": "atomic",
        "charges":    None,
        "pair_cmds":  None,   # set dynamically with pot_file
        "pot_file":   "SiO.2013.vashishta",   # reuse if needed
        "tersoff_file": True,
    },
    "Vashishta": {
        "atom_style": "atomic",
        "charges":    None,
        "pair_cmds":  None,   # set dynamically with pot_file
        "pot_file":   "Vashishta.pot",
        "vashishta_file": True,
    },
}

# E_quartz reference per expert (eV/atom) — for deltaE calculation
E_QUARTZ_REF = {
    "ML-BKS":       -9.690244855967,
    "Original_BKS": -19.477263786008,
    "Chik_BKS":     -13.700101234568,
    "PSO1_BKS":     -12.098198353909,
    "PSO2_BKS":     -20.950408515503028,
    "Tersoff":       -6.658804049608007,
    "Vashishta":     -1.5275400696992714,
}


# ── Gate model ────────────────────────────────────────────────────────────────
class CompatibleGate(nn.Module):
    """Gate network supporting any number of hidden layers."""
    def __init__(self, n_feat=18, n_exp=7, hidden=(64,32)):
        super().__init__()
        layers = []
        in_d   = n_feat
        for h in hidden:
            layers += [nn.Linear(in_d, h), nn.ReLU()]
            in_d = h
        layers += [nn.Linear(in_d, n_exp)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

class SOAPGate(nn.Module):
    """Gate network with LayerNorm — matches train_soap_gate_7exp.py architecture."""
    def __init__(self, n_feat=783, n_exp=7, hidden=(256,128,64)):
        super().__init__()
        layers = []
        in_d   = n_feat
        for h in hidden:
            layers += [nn.Linear(in_d,h), nn.LayerNorm(h),
                       nn.ReLU(), nn.Dropout(0.1)]
            in_d = h
        layers += [nn.Linear(in_d, n_exp)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)



class TorchSOAPGate(nn.Module):
    """TorchSOAP gate — architecture: Linear→LayerNorm→SiLU→Dropout per block."""
    def __init__(self, n_feat=783, n_exp=7, hidden=(256,128,64), dropout=0.0):
        super().__init__()
        layers = []
        in_d   = n_feat
        for h in hidden:
            layers += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(dropout)]
            in_d    = h
        layers += [nn.Linear(in_d, n_exp)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

def load_gate(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd   = ckpt.get("gate_state_dict", ckpt)

    # Auto-detect n_features from first linear layer
    n_feat = ckpt.get("n_features", None)
    if n_feat is None:
        w_keys = [k for k in sd.keys() if k.endswith(".weight")]
        n_feat = sd[w_keys[0]].shape[1] if w_keys else 18

    # Auto-detect hidden sizes from checkpoint (supports 2 or 3 hidden layers)
    hidden = ckpt.get("hidden", None)
    if hidden is not None:
        hidden = tuple(hidden)
    else:
        w_keys = sorted([k for k in sd.keys()
                         if "net." in k and k.endswith(".weight")])
        # All linear weights except last = hidden layers
        # Each hidden layer has one weight; last weight outputs n_exp
        hidden_keys = w_keys[:-1]  # exclude output layer
        hidden = tuple(sd[k].shape[0] for k in hidden_keys)
        if not hidden:
            hidden = (64, 32)

    n_exp = ckpt.get("n_experts", None)
    if n_exp is None:
        n_exp = sd[w_keys[-1]].shape[0]

    # Use TorchSOAPGate for TorchSOAP checkpoints
    if ckpt.get("descriptor_type","") == "TorchSOAP":
        gate = TorchSOAPGate(n_feat, n_exp, hidden)
    elif ckpt.get("descriptor_type","") == "SOAP":
        gate = SOAPGate(n_feat, n_exp, hidden)
    else:
        gate = CompatibleGate(n_feat, n_exp, hidden)
    model_keys = set(gate.state_dict().keys())
    clean_sd   = {k: v for k, v in sd.items() if k in model_keys}
    gate.load_state_dict(clean_sd, strict=False)
    gate.eval()

    feat_mean     = np.array(ckpt["feat_mean"], dtype=np.float32)
    feat_std      = np.array(ckpt["feat_std"],  dtype=np.float32)
    feat_std[feat_std < 1e-8] = 1.0
    experts       = ckpt.get("expert_names", ckpt.get("experts"))
    feature_names = list(ckpt.get("feature_names", []))
    return gate, feat_mean, feat_std, experts, feature_names



def extract_descriptors(atoms, is_zeolite=False):
    from ase.neighborlist import neighbor_list as _nl
    masses = atoms.get_masses()
    syms   = np.array(["Si" if abs(m-SI_MASS)<1.0 else "O" for m in masses])
    atoms.set_chemical_symbols(syms)
    si_idx = np.where(syms=="Si")[0]
    o_idx  = np.where(syms=="O")[0]
    cell    = atoms.get_cell()
    lengths = cell.lengths()
    angles  = cell.angles()
    volume  = atoms.get_volume()
    density = np.sum(masses)*1.66054e-24/(volume*1e-24)

    si_to_o, o_to_si, sio_b = {}, {}, []
    try:
        ia,ja,da,Da = _nl("ijdD", atoms, 2.4)
        seen = set()
        for i,j,r,D in zip(ia,ja,da,Da):
            i,j=int(i),int(j)
            if syms[i]==syms[j]: continue
            si,oi=(i,j) if syms[i]=="Si" else (j,i)
            v=np.array(D) if syms[i]=="Si" else -np.array(D)
            key=(min(si,oi),max(si,oi))
            if key in seen: continue
            seen.add(key)
            si_to_o.setdefault(si,[]).append((oi,float(r),v))
            o_to_si.setdefault(oi,[]).append((si,float(r),-v))
            sio_b.append(float(r))
    except: pass

    def _ang(v1,v2):
        n1,n2=np.linalg.norm(v1),np.linalg.norm(v2)
        if n1<1e-10 or n2<1e-10: return 0.0
        return np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n1*n2),-1,1)))

    oa,sa,sc,oc=[],[],[],[]
    tet=0
    for si in si_idx:
        on=si_to_o.get(int(si),[])
        sc.append(len(on))
        if len(on)==4: tet+=1
        for k1 in range(len(on)):
            for k2 in range(k1+1,len(on)):
                a=_ang(on[k1][2],on[k2][2])
                if a>1: oa.append(a)
    for oi in o_idx:
        sn=o_to_si.get(int(oi),[])
        oc.append(len(sn))
        for k1 in range(len(sn)):
            for k2 in range(k1+1,len(sn)):
                a=_ang(sn[k1][2],sn[k2][2])
                if a>1: sa.append(a)

    sm=lambda l,d: float(np.mean(l)) if l else d
    ss=lambda l,d: float(np.std(l)) if len(l)>1 else d
    return np.array([
        lengths[0],lengths[1],lengths[2],
        angles[0],angles[1],angles[2],
        volume,density,float(is_zeolite),
        sm(sa,144.0),ss(sa,5.0),
        sm(oa,109.47),ss(oa,2.0),
        sm(sc,4.0),sm(oc,2.0),
        tet/max(len(si_idx),1),
        sm(sio_b,1.61),ss(sio_b,0.01),
    ], dtype=np.float32)



# ── Differentiable descriptor extraction for autograd gate-gradient forces ────
def _torch_weighted_mean_std(values, weights, default_mean, default_std, eps=1e-12):
    """Weighted mean/std helper that remains differentiable."""
    denom = torch.sum(weights) + eps
    mean = torch.sum(weights * values) / denom
    var = torch.sum(weights * (values - mean) ** 2) / denom
    std = torch.sqrt(torch.clamp(var, min=eps))

    # If there is effectively no data, fall back to training defaults.
    mean = torch.where(denom > 10.0 * eps, mean, values.new_tensor(float(default_mean)))
    std = torch.where(denom > 10.0 * eps, std, values.new_tensor(float(default_std)))
    return mean, std


def extract_descriptors_torch_autograd(
    positions,
    cell,
    symbols,
    masses,
    is_zeolite=False,
    smooth_cutoff=2.4,
    smooth_width=0.10,
    tetra_sigma=0.25,
):
    """
    Differentiable approximation of the existing 18 silica descriptors.

    Output order is intentionally kept identical to extract_descriptors():
      [a,b,c, alpha,beta,gamma, volume,density,is_zeolite,
       mean_SiOSi,std_SiOSi, mean_OSiO,std_OSiO,
       mean_Si_CN, mean_O_CN, smooth_tetra_score,
       mean_SiO_bond, std_SiO_bond]

    Notes:
      - Hard cutoffs/counts are replaced by smooth sigmoid bond weights.
      - PBC wrapping uses torch.round(); its derivative is zero almost everywhere,
        which is acceptable away from cell boundaries for MD perturbations.
      - This is meant to give autograd access to d w_i / dR for the gate force term.
    """
    dtype = positions.dtype
    device = positions.device
    eps = positions.new_tensor(1e-12)

    # Cell geometry. ASE stores cell vectors as rows; positions = frac @ cell.
    A, B, C = cell[0], cell[1], cell[2]
    a = torch.linalg.norm(A)
    b = torch.linalg.norm(B)
    c = torch.linalg.norm(C)

    def _angle(u, v):
        cosang = torch.sum(u * v) / (torch.linalg.norm(u) * torch.linalg.norm(v) + eps)
        cosang = torch.clamp(cosang, -1.0 + 1e-7, 1.0 - 1e-7)
        return torch.acos(cosang) * (180.0 / np.pi)

    alpha = _angle(B, C)
    beta  = _angle(A, C)
    gamma = _angle(A, B)
    volume = torch.abs(torch.det(cell))

    mass_total = torch.as_tensor(float(np.sum(masses)), dtype=dtype, device=device)
    density = mass_total * 1.66054 / (volume + eps)

    syms = list(symbols)
    si_mask_np = np.array([s == "Si" for s in syms], dtype=bool)
    o_mask_np  = np.array([s == "O"  for s in syms], dtype=bool)
    si_mask = torch.as_tensor(si_mask_np, dtype=torch.bool, device=device)
    o_mask  = torch.as_tensor(o_mask_np,  dtype=torch.bool, device=device)

    natoms = positions.shape[0]
    eye = torch.eye(natoms, dtype=torch.bool, device=device)

    # Minimum-image pair vectors dR[i,j] = R_j - R_i.
    inv_cell = torch.inverse(cell)
    frac = positions @ inv_cell
    dfrac = frac[None, :, :] - frac[:, None, :]
    dfrac = dfrac - torch.round(dfrac)
    dcart = dfrac @ cell
    dist = torch.linalg.norm(dcart + eps, dim=-1)

    # Smooth Si-O bond weights.
    rc = positions.new_tensor(float(smooth_cutoff))
    width = positions.new_tensor(float(smooth_width))
    w_bond = torch.sigmoid((rc - dist) / width)
    w_bond = torch.where(eye, torch.zeros_like(w_bond), w_bond)

    mask_SiO = si_mask[:, None] & o_mask[None, :]
    mask_OSi = o_mask[:, None] & si_mask[None, :]

    w_SiO = torch.where(mask_SiO, w_bond, torch.zeros_like(w_bond))
    w_OSi = torch.where(mask_OSi, w_bond, torch.zeros_like(w_bond))

    # Smooth coordination statistics.
    cn_si_all = torch.sum(w_SiO, dim=1)
    cn_o_all  = torch.sum(w_OSi, dim=1)
    cn_si = cn_si_all[si_mask]
    cn_o  = cn_o_all[o_mask]

    mean_cn_si = torch.mean(cn_si) if cn_si.numel() else positions.new_tensor(4.0)
    mean_cn_o  = torch.mean(cn_o)  if cn_o.numel()  else positions.new_tensor(2.0)

    # Smooth tetrahedral score; differentiable replacement for fraction CN==4.
    sigma = positions.new_tensor(float(tetra_sigma))
    if cn_si.numel():
        tet_score = torch.mean(torch.exp(-((cn_si - 4.0) ** 2) / (2.0 * sigma ** 2)))
    else:
        tet_score = positions.new_tensor(1.0)

    # Weighted Si-O bond length mean/std, using Si->O directed pairs only.
    sio_weights = w_SiO[mask_SiO]
    sio_dists = dist[mask_SiO]
    if sio_dists.numel():
        mean_sio, std_sio = _torch_weighted_mean_std(sio_dists, sio_weights, 1.61, 0.01)
    else:
        mean_sio = positions.new_tensor(1.61)
        std_sio = positions.new_tensor(0.01)

    # Smooth angle statistics.
    siosi_vals, siosi_wts = [], []  # Si-O-Si centered on O
    osio_vals, osio_wts = [], []    # O-Si-O centered on Si

    for center in range(natoms):
        if si_mask[center]:
            # O-Si-O: vectors from Si center to O neighbors.
            idx = torch.where(o_mask)[0]
            if idx.numel() >= 2:
                vecs = dcart[center, idx, :]
                ws = w_bond[center, idx]
                m = idx.numel()
                for aa in range(m):
                    for bb in range(aa + 1, m):
                        v1, v2 = vecs[aa], vecs[bb]
                        n1 = torch.linalg.norm(v1)
                        n2 = torch.linalg.norm(v2)
                        cosang = torch.sum(v1 * v2) / (n1 * n2 + eps)
                        cosang = torch.clamp(cosang, -1.0 + 1e-7, 1.0 - 1e-7)
                        theta = torch.acos(cosang) * (180.0 / np.pi)
                        wt = ws[aa] * ws[bb]
                        osio_vals.append(theta)
                        osio_wts.append(wt)

        elif o_mask[center]:
            # Si-O-Si: vectors from O center to Si neighbors.
            idx = torch.where(si_mask)[0]
            if idx.numel() >= 2:
                vecs = dcart[center, idx, :]
                ws = w_bond[center, idx]
                m = idx.numel()
                for aa in range(m):
                    for bb in range(aa + 1, m):
                        v1, v2 = vecs[aa], vecs[bb]
                        n1 = torch.linalg.norm(v1)
                        n2 = torch.linalg.norm(v2)
                        cosang = torch.sum(v1 * v2) / (n1 * n2 + eps)
                        cosang = torch.clamp(cosang, -1.0 + 1e-7, 1.0 - 1e-7)
                        theta = torch.acos(cosang) * (180.0 / np.pi)
                        wt = ws[aa] * ws[bb]
                        siosi_vals.append(theta)
                        siosi_wts.append(wt)

    if siosi_vals:
        siosi_vals = torch.stack(siosi_vals)
        siosi_wts = torch.stack(siosi_wts)
        mean_siosi, std_siosi = _torch_weighted_mean_std(siosi_vals, siosi_wts, 144.0, 5.0)
    else:
        mean_siosi = positions.new_tensor(144.0)
        std_siosi = positions.new_tensor(5.0)

    if osio_vals:
        osio_vals = torch.stack(osio_vals)
        osio_wts = torch.stack(osio_wts)
        mean_osio, std_osio = _torch_weighted_mean_std(osio_vals, osio_wts, 109.47, 2.0)
    else:
        mean_osio = positions.new_tensor(109.47)
        std_osio = positions.new_tensor(2.0)

    isz = positions.new_tensor(float(is_zeolite))

    feats = torch.stack([
        a, b, c,
        alpha, beta, gamma,
        volume, density, isz,
        mean_siosi, std_siosi,
        mean_osio, std_osio,
        mean_cn_si, mean_cn_o,
        tet_score,
        mean_sio, std_sio,
    ])
    return feats


def ase_cell_to_lammps_box(atoms):
    """Convert ASE cell to LAMMPS restricted triclinic parameters: lx, ly, lz, xy, xz, yz."""
    cell = atoms.get_cell()
    A = np.array(cell[0], dtype=float)
    B = np.array(cell[1], dtype=float)
    C = np.array(cell[2], dtype=float)
    lx = float(np.linalg.norm(A))
    if lx < 1e-14:
        raise ValueError("Cell vector A has near-zero length")
    ex = A / lx
    xy = float(np.dot(B, ex))
    B_perp = B - xy * ex
    ly = float(np.linalg.norm(B_perp))
    ey = B_perp / ly if ly > 1e-14 else np.array([0.0, 1.0, 0.0])
    xz = float(np.dot(C, ex))
    yz = float(np.dot(C, ey))
    C_perp = C - xz * ex - yz * ey
    lz = float(np.linalg.norm(C_perp))
    if lz < 1e-14:
        raise ValueError("Cell vector C has near-zero projected length")
    return lx, ly, lz, xy, xz, yz


def strain_atoms_voigt(atoms, component, delta):
    """Return atoms strained in ASE Voigt order [xx, yy, zz, yz, xz, xy]."""
    eps = np.zeros((3, 3), dtype=float)
    if component == 0:
        eps[0, 0] = delta
    elif component == 1:
        eps[1, 1] = delta
    elif component == 2:
        eps[2, 2] = delta
    elif component == 3:
        eps[1, 2] = eps[2, 1] = 0.5 * delta
    elif component == 4:
        eps[0, 2] = eps[2, 0] = 0.5 * delta
    elif component == 5:
        eps[0, 1] = eps[1, 0] = 0.5 * delta
    else:
        raise ValueError(f"Invalid Voigt component: {component}")
    a = atoms.copy()
    F = np.eye(3) + eps
    a.set_cell(np.dot(a.get_cell().array, F.T), scale_atoms=True)
    a.set_pbc(atoms.get_pbc())
    return a


# ── Persistent LAMMPS instance ────────────────────────────────────────────────
class PersistentLammps:
    """
    One persistent LAMMPS instance for one expert potential.
    Initialized once, atoms scatter/gathered at every step.
    """

    def __init__(self, expert_name, atoms, pot_dir, lmp_lib, lmp_python):
        self.expert = expert_name
        self.natoms = len(atoms)
        self.setup  = EXPERT_SETUP[expert_name]

        # Load LAMMPS
        os.environ["MPICH_GPU_SUPPORT_ENABLED"] = "0"
        ctypes.CDLL(lmp_lib, ctypes.RTLD_GLOBAL)
        sys.path.insert(0, lmp_python)
        from lammps import lammps
        self.lmp = lammps(cmdargs=["-screen","none","-log","none"])

        # Setup simulation box
        self._setup_box(atoms, pot_dir)
        print(f"  [ok] {expert_name}: LAMMPS instance ready ({self.natoms} atoms)")

    def _setup_box(self, atoms, pot_dir):
        """Use write+read_data for reliable atom placement."""
        import tempfile, os
        lmp  = self.lmp
        syms = atoms.get_chemical_symbols()
        style = self.setup["atom_style"]

        # Write temporary LAMMPS data file
        tmpf = tempfile.NamedTemporaryFile(suffix=".data", delete=False,
                                           mode="w")
        self._tmp_data = tmpf.name

        cell = atoms.get_cell()
        pos  = atoms.get_positions()
        natoms = len(atoms)

        # Convert cell to LAMMPS triclinic parameters
        ax, by, cz, bx, cx, cy = ase_cell_to_lammps_box(atoms)

        charges = self.setup.get("charges")

        tmpf.write(f"LAMMPS data file for {self.expert}\n\n")
        tmpf.write(f"{natoms} atoms\n")
        tmpf.write(f"2 atom types\n\n")
        tmpf.write(f"0.0 {ax:.10f} xlo xhi\n")
        tmpf.write(f"0.0 {by:.10f} ylo yhi\n")
        tmpf.write(f"0.0 {cz:.10f} zlo zhi\n")
        if abs(bx)+abs(cx)+abs(cy) > 1e-10:
            tmpf.write(f"{bx:.10f} {cx:.10f} {cy:.10f} xy xz yz\n")
        tmpf.write("\nMasses\n\n")
        tmpf.write("1 28.0855\n")
        tmpf.write("2 15.9994\n")
        if charges:
            tmpf.write("\nAtoms # charge\n\n")
            for i,(sym,p) in enumerate(zip(syms,pos)):
                atype = 1 if sym=="Si" else 2
                q = charges["Si"] if sym=="Si" else charges["O"]
                tmpf.write(f"{i+1} {atype} {q:.6f} "
                           f"{p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")
        else:
            tmpf.write("\nAtoms # atomic\n\n")
            for i,(sym,p) in enumerate(zip(syms,pos)):
                atype = 1 if sym=="Si" else 2
                tmpf.write(f"{i+1} {atype} "
                           f"{p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")
        tmpf.close()

        # Setup LAMMPS
        lmp.command("units metal")
        lmp.command(f"atom_style {style}")
        lmp.command("atom_modify map array")
        lmp.command("boundary p p p")
        lmp.command("box tilt large")
        lmp.command(f"read_data {self._tmp_data}")

        # Set potential
        if self.setup.get("tersoff_file"):
            pot = str(Path(pot_dir) / "Tersoff.pot")
            lmp.command("pair_style tersoff")
            lmp.command(f"pair_coeff * * {pot} Si O")
        elif self.setup.get("vashishta_file"):
            pot = str(Path(pot_dir) / "Vashishta.pot")
            lmp.command("pair_style vashishta")
            lmp.command(f"pair_coeff * * {pot} Si O")
        else:
            for cmd in self.setup["pair_cmds"]:
                lmp.command(cmd)

        lmp.command("neighbor 2.0 bin")
        lmp.command("neigh_modify delay 0 every 1 check yes")
        lmp.command("fix 1 all nve")
        lmp.command("run 0")
        os.unlink(self._tmp_data)

        self._last_box = np.array([ax, by, cz, bx, cx, cy], dtype=float)

        # Verify atom count
        lmp_natoms = lmp.get_natoms()
        assert lmp_natoms == natoms, \
            f"{self.expert}: LAMMPS has {lmp_natoms} atoms, ASE has {natoms}"

    def _update_box_if_needed(self, atoms, tol=1e-10):
        """
        Update the LAMMPS box to match the current ASE cell.

        Important:
        - Orthogonal LAMMPS boxes cannot accept xy/xz/yz tilt changes.
        - Triclinic boxes can accept tilt changes.
        - Therefore, only include tilt terms when the cell is actually tilted.
        """
        import numpy as np

        cell = atoms.get_cell()
        A = np.array(cell[0], dtype=float)
        B = np.array(cell[1], dtype=float)
        C = np.array(cell[2], dtype=float)

        ax = float(np.linalg.norm(A))
        if ax <= 1e-14:
            raise RuntimeError("Invalid cell: |A| is zero")

        ahat = A / ax

        bx = float(np.dot(B, ahat))
        b2 = float(np.dot(B, B))
        by = float(np.sqrt(max(b2 - bx * bx, 0.0)))
        if by <= 1e-14:
            raise RuntimeError("Invalid cell: computed by is zero")

        cx = float(np.dot(C, ahat))
        cy = float((np.dot(B, C) - bx * cx) / by)
        c2 = float(np.dot(C, C))
        cz = float(np.sqrt(max(c2 - cx * cx - cy * cy, 0.0)))

        # LAMMPS restricted triclinic parameters:
        # xhi=ax, yhi=by, zhi=cz, xy=bx, xz=cx, yz=cy
        box_vec = np.array([ax, by, cz, bx, cx, cy], dtype=float)

        if hasattr(self, "_last_box_vec"):
            if np.allclose(box_vec, self._last_box_vec, rtol=tol, atol=tol):
                return

        self._last_box_vec = box_vec.copy()

        tilt_tol = 1e-10
        is_tilted = (abs(bx) > tilt_tol) or (abs(cx) > tilt_tol) or (abs(cy) > tilt_tol)

        if is_tilted:
            self.lmp.command(
                "change_box all "
                f"x final 0.0 {ax:.16g} "
                f"y final 0.0 {by:.16g} "
                f"z final 0.0 {cz:.16g} "
                f"xy final {bx:.16g} "
                f"xz final {cx:.16g} "
                f"yz final {cy:.16g} "
                "remap units box"
            )
        else:
            self.lmp.command(
                "change_box all "
                f"x final 0.0 {ax:.16g} "
                f"y final 0.0 {by:.16g} "
                f"z final 0.0 {cz:.16g} "
                "remap units box"
            )

    def compute(self, atoms):
        """Update positions, compute forces and energy. Returns (E_per_atom, forces)."""
        import numpy as np
        lmp = self.lmp

        # NPT safety: keep persistent LAMMPS box consistent with ASE cell.
        self._update_box_if_needed(atoms)

        # Scatter new positions. Wrap a copy so positions remain inside the updated box.
        atoms_for_lmp = atoms.copy()
        atoms_for_lmp.wrap(eps=1e-12)
        pos_flat = atoms_for_lmp.get_positions().flatten().tolist()
        import ctypes
        pos_c = (ctypes.c_double * len(pos_flat))(*pos_flat)
        lmp.scatter_atoms("x", 1, 3, pos_c)

        # Compute
        lmp.command("run 0")

        # Gather forces — use LAMMPS natoms (may differ from ASE due to ghost atoms)
        f_raw  = lmp.gather_atoms("f", 1, 3)
        f_flat = np.array(f_raw, dtype=np.float64)
        lmp_natoms = lmp.get_natoms()
        forces = f_flat.reshape(lmp_natoms, 3)
        # Trim to ASE natoms if needed
        forces = forces[:self.natoms]

        # Energy
        E_total = lmp.get_thermo("pe")
        E_per_atom = E_total / self.natoms

        # Stress in ASE Voigt convention: [xx, yy, zz, yz, xz, xy]
        # LAMMPS pxx/pyy/... are pressures in bar for units metal.
        # ASE stress is negative pressure in eV/Angstrom^3.
        pxx = lmp.get_thermo("pxx")
        pyy = lmp.get_thermo("pyy")
        pzz = lmp.get_thermo("pzz")
        pxy = lmp.get_thermo("pxy")
        pxz = lmp.get_thermo("pxz")
        pyz = lmp.get_thermo("pyz")

        bar_to_evA3 = 1.0e5 / 1.602176634e11
        stress = -bar_to_evA3 * np.array([pxx, pyy, pzz, pyz, pxz, pxy],
                                          dtype=np.float64)

        return E_per_atom, forces, stress

    def close(self):
        try:
            self.lmp.close()
        except: pass


# ── Main MoE Calculator ───────────────────────────────────────────────────────
class MoEPersistentCalculator:
    """
    ASE-compatible MoE calculator using persistent LAMMPS instances.

    Unlike the subprocess-based calculator, this keeps N LAMMPS instances
    alive for the entire simulation. Forces are combined at every MD step
    — no file I/O, no process spawning.

    Parameters
    ----------
    gate_ckpt    : path to gate checkpoint (.pt)
    pot_dir      : directory containing Tersoff/Vashishta potential files
    lmp_lib      : path to liblammps.so.0
    lmp_python   : path to LAMMPS Python bindings directory
    active_thresh: experts with w < thresh are skipped
    update_every : recompute gate weights every N steps
    smooth_alpha : EMA smoothing coefficient for weights
    is_zeolite   : whether structure is a zeolite
    verbose      : print weight updates
    """

    def __init__(self,
                 gate_ckpt:     str,
                 pot_dir:       str,
                 lmp_lib:       str  = LMP_LIB,
                 lmp_python:    str  = LMP_PYTHON,
                 active_thresh: float = 0.03,
                 update_every:  int   = 100,
                 smooth_alpha:  float = 0.9,
                 is_zeolite:    bool  = False,
                 verbose:       bool  = True,
                 consistent_forces: bool = False,
                 consistent_stress: bool = False,
                 gate_fd_delta: float = 1e-4,
                 strain_fd_delta: float = 1e-5,
                 gate_fd_max_atoms: int = 64,
                 stress_fd_sign: float = 1.0,
                 stress_fd_scale: float = 1.0,
                 force_correction_method: str = "fd",
                 smooth_cutoff: float = 2.4,
                 smooth_width: float = 0.10,
                 tetra_sigma: float = 0.25):

        self.pot_dir       = pot_dir
        self.lmp_lib       = lmp_lib
        self.lmp_python    = lmp_python
        self.active_thresh = active_thresh
        self.update_every  = update_every
        self.smooth_alpha  = smooth_alpha
        self.is_zeolite    = is_zeolite
        self.verbose       = verbose

        # If True, add finite-difference gate-gradient force correction:
        # F = sum_i w_i F_i - sum_i E_i_aligned * d w_i / dR.
        # This is intended first for small-cell diagnostics, not large production runs.
        self.consistent_forces = bool(consistent_forces)
        self.consistent_stress = bool(consistent_stress)
        self.gate_fd_delta = float(gate_fd_delta)
        self.strain_fd_delta = float(strain_fd_delta)
        self.gate_fd_max_atoms = int(gate_fd_max_atoms)
        self.stress_fd_sign = float(stress_fd_sign)
        self.stress_fd_scale = float(stress_fd_scale)
        self.force_correction_method = str(force_correction_method).lower()
        self.smooth_cutoff = float(smooth_cutoff)
        self.smooth_width = float(smooth_width)
        self.tetra_sigma = float(tetra_sigma)
        if self.force_correction_method not in ("fd", "finite_difference", "autograd"):
            raise ValueError("force_correction_method must be 'fd' or 'autograd'")
        self._printed_fd_warning = False
        self._printed_stress_warning = False

        # Load gate
        print(f"[MoEPersistent] Loading gate: {gate_ckpt}")
        self.gate, self.feat_mean, self.feat_std, self.experts, self.feature_names = \
            load_gate(gate_ckpt)
        self.n_experts = len(self.experts)
        print(f"[MoEPersistent] Experts: {self.experts}")

        # Initialize SOAP descriptor if checkpoint uses SOAP
        ckpt_full = torch.load(gate_ckpt, map_location="cpu", weights_only=False)
        self._descriptor_type = ckpt_full.get("descriptor_type", "")
        self._torchsoap_desc  = None
        if ckpt_full.get("descriptor_type","") == "SOAP":
            print(f"[MoEPersistent] Using SOAP descriptor "
                  f"(rcut={ckpt_full.get('soap_rcut',5.0)}, "
                  f"nmax={ckpt_full.get('soap_nmax',6)}, "
                  f"lmax={ckpt_full.get('soap_lmax',4)})")
            self._soap_desc = _get_soap_descriptor(ckpt_full)
        # TorchSOAP descriptor (differentiable, for autograd force correction)
        # Also enabled for SOAP-type checkpoints (same architecture, different label)
        if self._descriptor_type in ("TorchSOAP", "SOAP"):
            try:
                import sys, os
                # Try to find torchsoap_autograd_module
                _gate_dir = str(Path(gate_ckpt).parent)
                if _gate_dir not in sys.path:
                    sys.path.insert(0, _gate_dir)
                from torchsoap_autograd_module import TorchSOAPDescriptor, TorchSOAPConfig
                _cfg = TorchSOAPConfig(
                    r_cut  = ckpt_full.get("soap_rcut", 5.0),
                    n_max  = ckpt_full.get("soap_nmax", 6),
                    l_max  = ckpt_full.get("soap_lmax", 4),
                )
                self._torchsoap_desc = TorchSOAPDescriptor(_cfg)
                print(f"  [TorchSOAP] Differentiable descriptor loaded "
                      f"(r_cut={_cfg.r_cut}, n_max={_cfg.n_max}, l_max={_cfg.l_max})")
            except Exception as _e:
                print(f"  [warn] TorchSOAP descriptor not available: {_e}")
                print(f"  [warn] Autograd force correction will fall back to FD")
                self._torchsoap_desc = None
        else:
            if not hasattr(self, "_soap_desc"):
                self._soap_desc = None

        # State
        self._initialized  = False
        self._lmp_instances: Dict[str, PersistentLammps] = {}
        self._weights      = np.ones(self.n_experts) / self.n_experts
        self._step         = 0
        self._weight_history = []
        self._energy_history = []  # per-expert raw energies (eV/atom)
        self._feat_history   = []
        self._atoms_ref      = None

    def initialize(self, atoms):
        """Initialize persistent LAMMPS instances for all active experts."""
        print(f"\n[MoEPersistent] Initializing {self.n_experts} LAMMPS instances...")

        # Compute initial weights to determine active experts
        w0 = self._compute_weights(atoms)
        self._weights = w0

        for i, exp in enumerate(self.experts):
            # Initialize all experts once. active_thresh is applied only during
            # force summation, so an expert can become active later if weights change.
            try:
                inst = PersistentLammps(
                    exp, atoms, self.pot_dir,
                    self.lmp_lib, self.lmp_python)
                self._lmp_instances[exp] = inst
                if w0[i] < self.active_thresh:
                    print(f"  [ready/inactive] {exp}: w={w0[i]:.4f} < {self.active_thresh}")
                else:
                    print(f"  [ready/active]   {exp}: w={w0[i]:.4f}")
            except Exception as e:
                print(f"  [warn] {exp}: init failed: {e}")

        self._initialized = True
        self._atoms_ref   = atoms.copy()
        print(f"[MoEPersistent] Initialized {len(self._lmp_instances)} experts")
        self._print_weights()

    def _compute_weights(self, atoms):
        """Compute gate weights — uses TorchSOAP, dscribe SOAP, or 18-feat based on checkpoint."""
        # TorchSOAP descriptor (differentiable, highest priority)
        if hasattr(self, "_torchsoap_desc") and self._torchsoap_desc is not None:
            import torch as _torch
            pos  = _torch.tensor(atoms.get_positions(), dtype=_torch.float64)
            cell = _torch.tensor(atoms.get_cell().array, dtype=_torch.float64)
            with _torch.no_grad():
                feats = self._torchsoap_desc(
                    positions=pos,
                    cell=cell,
                    symbols=atoms.get_chemical_symbols(),
                    masses=atoms.get_masses(),
                    is_zeolite=self.is_zeolite,
                ).float()
                feats_norm = (feats - _torch.tensor(self.feat_mean)) / _torch.tensor(self.feat_std)
                w = self.gate(feats_norm.unsqueeze(0)).squeeze(0).numpy()
            return w
        # Use dscribe SOAP descriptor if gate was trained with SOAP
        if hasattr(self, "_soap_desc") and self._soap_desc is not None:
            feats = self._soap_desc.compute(atoms, self.is_zeolite)
            feats_norm = (feats - self.feat_mean) / self.feat_std
            x = torch.tensor(feats_norm, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                w = self.gate(x).squeeze(0).numpy()
            return w
        feats      = extract_descriptors(atoms, self.is_zeolite)
        # Select only features this gate was trained on
        FEAT_NAMES_ALL = [
            "a","b","c","alpha","beta","gamma",
            "volume","density","is_zeolite",
            "mean_SiOSi","std_SiOSi","mean_OSiO","std_OSiO",
            "Si_coord_mean","O_coord_mean","tetrahedral_fraction",
            "mean_SiO_bond","std_SiO_bond",
        ]
        if hasattr(self,"feature_names") and self.feature_names and            len(self.feature_names) != len(feats):
            feat_dict = dict(zip(FEAT_NAMES_ALL, feats))
            feat_dict["ref_siosi"] = feat_dict.get("mean_SiOSi", 144.0)
            feats = np.array([feat_dict.get(n, 0.0) for n in self.feature_names],
                             dtype=np.float32)
        feats_norm = (feats - self.feat_mean) / self.feat_std
        x = torch.tensor(feats_norm, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            w = self.gate(x).squeeze(0).numpy()
        return w

    def _print_weights(self):
        print(f"  Weights (step {self._step}):")
        for exp, w in zip(self.experts, self._weights):
            bar = "█" * int(w * 20)
            print(f"    {exp:<15} {w:.4f} {bar}")

    def _compute_gate_force_fd(self, atoms, aligned_E_total, active_indices):
        """
        Finite-difference gate-gradient force correction.

        Dynamic MoE energy:
            E_MoE(R) = sum_i w_i(R) * E_i_aligned(R)

        Force:
            F = sum_i w_i F_i - sum_i E_i_aligned * d w_i / dR

        The first term is already computed from expert forces. This routine
        estimates the second term by central finite differences of gate weights
        with respect to Cartesian atom positions.

        Cost: 2 * 3N descriptor/gate evaluations per force call. Use first on
        9-atom Quartz or other small-cell diagnostics.
        """
        natoms = len(atoms)
        if natoms > self.gate_fd_max_atoms:
            raise RuntimeError(
                f"consistent_forces=True with finite-difference gate correction is "
                f"limited to <= {self.gate_fd_max_atoms} atoms for safety. "
                f"Current natoms={natoms}. Increase gate_fd_max_atoms only for tests."
            )

        delta = float(self.gate_fd_delta)
        if delta <= 0:
            raise ValueError("gate_fd_delta must be positive")

        base_pos = atoms.get_positions().copy()
        F_gate = np.zeros((natoms, 3), dtype=np.float64)
        active_indices = list(active_indices)
        E_active = np.asarray(aligned_E_total, dtype=np.float64)

        for a in range(natoms):
            for mu in range(3):
                atoms_p = atoms.copy()
                atoms_m = atoms.copy()

                pos_p = base_pos.copy()
                pos_m = base_pos.copy()
                pos_p[a, mu] += delta
                pos_m[a, mu] -= delta

                atoms_p.set_positions(pos_p)
                atoms_m.set_positions(pos_m)

                w_p = self._compute_weights(atoms_p)
                w_m = self._compute_weights(atoms_m)
                dw = (w_p - w_m) / (2.0 * delta)

                # F_gate[a,mu] = - sum_i E_i_aligned_total * d w_i / d r_{a,mu}
                F_gate[a, mu] = -float(np.sum(E_active[active_indices] * dw[active_indices]))

        return F_gate


    def _compute_weights_autograd(self, atoms):
        """
        Compute differentiable gate weights with respect to Cartesian positions.
        For TorchSOAP gates: uses fully differentiable TorchSOAP descriptor.
        For legacy gates: uses 18-feature differentiable descriptor.
        Returns (w, pos) where w has autograd graph through pos.
        """
        symbols = atoms.get_chemical_symbols()
        masses  = atoms.get_masses()
        pos     = torch.tensor(atoms.get_positions(), dtype=torch.float64, requires_grad=True)
        cell    = torch.tensor(atoms.get_cell().array, dtype=torch.float64)

        if self._descriptor_type in ("TorchSOAP", "SOAP") and self._torchsoap_desc is not None:
            # Fully differentiable TorchSOAP descriptor (call forward directly)
            feats = self._torchsoap_desc(
                positions=pos,
                cell=cell,
                symbols=symbols,
                masses=masses,
                is_zeolite=self.is_zeolite,
            ).to(dtype=torch.float32)
        else:
            # Legacy 18-feature differentiable descriptor
            feats = extract_descriptors_torch_autograd(
                positions=pos,
                cell=cell,
                symbols=symbols,
                masses=masses,
                is_zeolite=self.is_zeolite,
                smooth_cutoff=self.smooth_cutoff,
                smooth_width=self.smooth_width,
                tetra_sigma=self.tetra_sigma,
            ).to(dtype=torch.float32)

        feat_mean  = torch.tensor(self.feat_mean, dtype=torch.float32)
        feat_std   = torch.tensor(self.feat_std,  dtype=torch.float32)
        feats_norm = (feats - feat_mean) / feat_std
        x = feats_norm.unsqueeze(0)
        w = self.gate(x).squeeze(0)
        return w, pos

    def _compute_gate_force_autograd(self, atoms, aligned_E_total, active_indices):
        """
        Autograd gate-gradient force correction.

        This computes the missing term
            F_gate = - d/dR [sum_i w_i(R) E_i_aligned]
        using a differentiable approximation to the existing 18 descriptors.

        The expert aligned energies are treated as constants in this correction;
        their position derivatives are already included through sum_i w_i F_i.
        """
        w, pos = self._compute_weights_autograd(atoms)
        E = torch.tensor(aligned_E_total, dtype=w.dtype)
        mask = torch.zeros_like(w)
        for i in active_indices:
            mask[int(i)] = 1.0
        gate_energy = torch.sum(w * E * mask)
        grad = torch.autograd.grad(gate_energy, pos, create_graph=False, retain_graph=False)[0]
        F_gate = -grad.detach().cpu().numpy().astype(np.float64)
        return F_gate

    def _moe_aligned_energy_for_atoms(self, atoms, active_indices=None):
        """Compute aligned dynamic MoE energy for an arbitrary configuration."""
        if active_indices is None:
            active_indices = list(range(self.n_experts))
        active_indices = list(active_indices)
        natoms = len(atoms)
        w = self._compute_weights(atoms)
        E_total = 0.0
        w_used = 0.0
        for i in active_indices:
            exp = self.experts[i]
            if exp not in self._lmp_instances:
                continue
            wi = float(w[i])
            if wi < self.active_thresh:
                continue
            E_i, _F_i, _S_i = self._lmp_instances[exp].compute(atoms)
            dE_i = E_i - E_QUARTZ_REF.get(exp, E_i)
            E_total += wi * dE_i * natoms
            w_used += wi
        if w_used < 1e-12:
            raise RuntimeError("No experts available for finite-difference MoE energy")
        return float(E_total)

    def _compute_consistent_stress_fd(self, atoms, active_indices):
        """Finite-difference strain derivative of the aligned dynamic MoE energy."""
        natoms = len(atoms)
        if natoms > self.gate_fd_max_atoms:
            raise RuntimeError(
                f"consistent_stress=True finite-difference stress is limited to <= "
                f"{self.gate_fd_max_atoms} atoms for safety. Current natoms={natoms}."
            )
        delta = float(self.strain_fd_delta)
        if delta <= 0:
            raise ValueError("strain_fd_delta must be positive")
        V = float(atoms.get_volume())
        if V <= 0:
            raise ValueError("Atoms volume must be positive for stress calculation")
        stress = np.zeros(6, dtype=np.float64)
        for comp in range(6):
            atoms_p = strain_atoms_voigt(atoms, comp, +delta)
            atoms_m = strain_atoms_voigt(atoms, comp, -delta)
            E_p = self._moe_aligned_energy_for_atoms(atoms_p, active_indices=active_indices)
            E_m = self._moe_aligned_energy_for_atoms(atoms_m, active_indices=active_indices)
            dE_deps = (E_p - E_m) / (2.0 * delta)
            stress[comp] = self.stress_fd_sign * self.stress_fd_scale * dE_deps / V
        return stress


    def get_forces(self, atoms):
        """Compute MoE forces. Main entry point for MD."""
        if not self._initialized:
            self.initialize(atoms)

        self._step += 1

        # Update weights
        if self._step % self.update_every == 0:
            w_new = self._compute_weights(atoms)
            a     = self.smooth_alpha
            self._weights = a*self._weights + (1-a)*w_new

            # Record history
            feats = extract_descriptors(atoms, self.is_zeolite)
            self._weight_history.append({
                "step":    self._step,
                "temp":    float(atoms.get_temperature()),
                "weights": self._weights.copy(),
            })
            self._energy_history.append({
                "step":    self._step,
                "temp":    float(atoms.get_temperature()),
                "energies": dict(getattr(self, "_current_expert_energies", {})),
            })
            self._feat_history.append({
                "step":          self._step,
                "temp":          float(atoms.get_temperature()),
                "mean_SiOSi":    float(feats[9]),
                "std_SiOSi":     float(feats[10]),
                "tetrahedral":   float(feats[15]),
                "density":       float(feats[7]),
            })

            if self.verbose:
                self._print_weights()

        # Compute forces from all active experts
        natoms    = len(atoms)
        F_moe      = np.zeros((natoms, 3))
        stress_moe = np.zeros(6, dtype=np.float64)
        E_moe      = 0.0
        w_used     = 0.0

        # Store aligned total energies for finite-difference gate-gradient correction.
        aligned_E_total = np.zeros(self.n_experts, dtype=np.float64)
        active_indices = []

        for i, exp in enumerate(self.experts):
            w = self._weights[i]
            if w < self.active_thresh or exp not in self._lmp_instances:
                continue
            try:
                E_i, F_i, S_i = self._lmp_instances[exp].compute(atoms)

                # Use deltaE vs quartz for reported/aligned energy.
                # E_i is per-atom expert energy; dE_total is total aligned energy.
                dE = E_i - E_QUARTZ_REF.get(exp, E_i)
                dE_total = dE * natoms

                aligned_E_total[i] = dE_total
                active_indices.append(i)

                # Store raw per-atom energy for diagnostics
                if not hasattr(self, '_current_expert_energies'):
                    self._current_expert_energies = {}
                self._current_expert_energies[exp] = float(E_i)

                E_moe      += w * dE_total
                F_moe      += w * F_i
                stress_moe += w * S_i
                w_used     += w
            except Exception as e:
                print(f"  [warn] {exp} compute failed: {e}")

        if w_used < 1e-6:
            raise RuntimeError("All experts failed")

        if self.consistent_forces:
            if self.smooth_alpha != 0.0 or self.update_every != 1:
                if not self._printed_fd_warning:
                    print("[MoEPersistent][warn] consistent_forces=True is most meaningful with "
                          "update_every=1 and smooth_alpha=0.0. Current settings use an "
                          "approximate gate-gradient correction.")
                    self._printed_fd_warning = True

            if self.force_correction_method == "autograd":
                F_gate = self._compute_gate_force_autograd(
                    atoms=atoms,
                    aligned_E_total=aligned_E_total,
                    active_indices=active_indices,
                )
            else:
                F_gate = self._compute_gate_force_fd(
                    atoms=atoms,
                    aligned_E_total=aligned_E_total,
                    active_indices=active_indices,
                )
            F_moe += F_gate

            if self.verbose and (self._step == 1 or self._step % max(self.update_every, 1) == 0):
                print(f"  [consistent_forces:{self.force_correction_method}] |F_gate|max = {np.abs(F_gate).max():.6e} eV/Å")

        if self.consistent_stress:
            if self.smooth_alpha != 0.0 or self.update_every != 1:
                if not self._printed_stress_warning:
                    print("[MoEPersistent][warn] consistent_stress=True is most meaningful with "
                          "update_every=1 and smooth_alpha=0.0. Current settings use an "
                          "approximate strain derivative.")
                    self._printed_stress_warning = True

            stress_moe = self._compute_consistent_stress_fd(
                atoms=atoms,
                active_indices=active_indices,
            )

            if self.verbose and (self._step == 1 or self._step % max(self.update_every, 1) == 0):
                print(f"  [consistent_stress] stress = {stress_moe}")

        return F_moe, E_moe, stress_moe

    def get_weight_history(self):
        """Return weight evolution as arrays."""
        if not self._weight_history:
            return np.array([]), np.zeros((0, self.n_experts))
        steps   = np.array([h["step"]    for h in self._weight_history])
        temps   = np.array([h["temp"]    for h in self._weight_history])
        weights = np.array([h["weights"] for h in self._weight_history])
        return steps, temps, weights

    def save_history(self, output_dir):
        """Save weight and descriptor history."""
        out = Path(output_dir)
        steps, temps, weights = self.get_weight_history()
        if len(steps) > 0:
            np.savez(out/"weight_history.npz",
                     steps=steps, temps=temps, weights=weights,
                     experts=np.array(self.experts))
        with open(out/"feat_history.json","w") as f:
            json.dump(self._feat_history, f, indent=2, default=float)
        print(f"  Saved weight history: {out}/weight_history.npz")

    def close(self):
        for inst in self._lmp_instances.values():
            inst.close()
        self._lmp_instances.clear()


# ── ASE Calculator wrapper ────────────────────────────────────────────────────
from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms as ASEAtoms

class MoECalculator(Calculator):
    """
    ASE Calculator wrapping MoEPersistentCalculator.
    Use this directly with any ASE MD/optimizer.

    Example:
        calc = MoECalculator(
            gate_ckpt = "gate.pt",
            pot_dir   = "/path/to/pots",
        )
        atoms.calc = calc
        dyn = NVTBerendsen(atoms, ...)
        dyn.run(1000)
    """
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, gate_ckpt, pot_dir,
                 lmp_lib=LMP_LIB, lmp_python=LMP_PYTHON,
                 active_thresh=0.03, update_every=100,
                 smooth_alpha=0.9, is_zeolite=False,
                 verbose=True, consistent_forces=False,
                 consistent_stress=False,
                 gate_fd_delta=1e-4, strain_fd_delta=1e-5,
                 gate_fd_max_atoms=64, stress_fd_sign=1.0,
                 stress_fd_scale=1.0,
                 force_correction_method="fd",
                 smooth_cutoff=2.4,
                 smooth_width=0.10,
                 tetra_sigma=0.25,
                 **kwargs):
        super().__init__(**kwargs)
        self._moe = MoEPersistentCalculator(
            gate_ckpt=gate_ckpt, pot_dir=pot_dir,
            lmp_lib=lmp_lib, lmp_python=lmp_python,
            active_thresh=active_thresh,
            update_every=update_every,
            smooth_alpha=smooth_alpha,
            is_zeolite=is_zeolite,
            verbose=verbose,
            consistent_forces=consistent_forces,
            consistent_stress=consistent_stress,
            gate_fd_delta=gate_fd_delta,
            strain_fd_delta=strain_fd_delta,
            gate_fd_max_atoms=gate_fd_max_atoms,
            stress_fd_sign=stress_fd_sign,
            stress_fd_scale=stress_fd_scale,
            force_correction_method=force_correction_method,
            smooth_cutoff=smooth_cutoff,
            smooth_width=smooth_width,
            tetra_sigma=tetra_sigma)

    def calculate(self, atoms=None, properties=["energy", "forces", "stress"],
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        F, E, S = self._moe.get_forces(self.atoms)

        self.results["energy"] = float(E)
        self.results["forces"] = F
        self.results["stress"] = S

        # ------------------------------------------------------------
        # Diagnostics: expose per-expert energies and gate weights.
        # This is needed for elastic/phonon/debug analysis.
        # ------------------------------------------------------------
        try:
            self.results["gate_weights"] = np.array(
                self._moe._weights, dtype=float
            ).copy()
        except Exception:
            pass

        try:
            # Raw per-atom expert energies, stored as dictionary:
            # {expert_name: E_i_per_atom}
            expert_energy_dict = getattr(self._moe, "_current_expert_energies", {})
            self.results["expert_energy_dict_eV_per_atom"] = dict(expert_energy_dict)

            # Ordered array following self._moe.experts
            self.results["expert_energies_eV_per_atom"] = np.array(
                [expert_energy_dict.get(exp, np.nan) for exp in self._moe.experts],
                dtype=float
            )

            # Convert to total expert energies.
            # This is what our energy-strain curvature diagnostic needs.
            natoms = len(self.atoms)
            self.results["expert_energies_eV"] = (
                self.results["expert_energies_eV_per_atom"] * natoms
            )

            self.results["expert_names"] = list(self._moe.experts)

        except Exception:
            pass

    def get_weight_history(self):
        return self._moe.get_weight_history()

    def get_energy_history(self):
        return self._moe.get_energy_history()

    def save_history(self, output_dir):
        self._moe.save_history(output_dir)

    def close(self):
        self._moe.close()


# ── Weight evolution plotting ─────────────────────────────────────────────────
def plot_weight_evolution(output_dir, structure_name=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update({"font.family":"serif","axes.linewidth":0.8,"pdf.fonttype":42})

    COLORS = ["#7F7F7F","#6F86A8","#63B3CF","#F2A34A",
              "#E85C5C","#E67E22","#9B59B6"]

    npz = np.load(Path(output_dir)/"weight_history.npz", allow_pickle=True)
    steps, temps, weights = npz["steps"], npz["temps"], npz["weights"]
    experts = list(npz["experts"])

    with open(Path(output_dir)/"feat_history.json") as f:
        fh = json.load(f)
    siosi = np.array([h["mean_SiOSi"]  for h in fh])
    siosi_s=np.array([h["std_SiOSi"]   for h in fh])
    tet   = np.array([h["tetrahedral"] for h in fh])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="white")

    # Stacked area
    bottom = np.zeros(len(temps))
    for j,(exp,col) in enumerate(zip(experts,COLORS)):
        axes[0,0].fill_between(temps, bottom, bottom+weights[:,j],
                               alpha=0.8, color=col, label=exp)
        bottom += weights[:,j]
    axes[0,0].set_xlabel("Temperature (K)")
    axes[0,0].set_ylabel("Gating weight")
    axes[0,0].set_title(f"(a) Expert weights — {structure_name}")
    axes[0,0].legend(fontsize=7, loc="upper right", ncol=2)
    axes[0,0].set_ylim(0,1); axes[0,0].grid(alpha=0.2)
    axes[0,0].spines["top"].set_visible(False)
    axes[0,0].spines["right"].set_visible(False)

    # Individual lines
    for j,(exp,col) in enumerate(zip(experts,COLORS)):
        axes[0,1].plot(temps, weights[:,j], color=col, lw=1.5, label=exp)
    axes[0,1].set_xlabel("Temperature (K)")
    axes[0,1].set_ylabel("Gating weight")
    axes[0,1].set_title("(b) Expert weights — individual")
    axes[0,1].legend(fontsize=7, loc="upper right", ncol=2)
    axes[0,1].set_ylim(0,1); axes[0,1].grid(alpha=0.2)
    axes[0,1].spines["top"].set_visible(False)
    axes[0,1].spines["right"].set_visible(False)

    # Si-O-Si
    axes[1,0].plot(temps, siosi, "b-", lw=1.5)
    axes[1,0].fill_between(temps, siosi-siosi_s, siosi+siosi_s,
                            alpha=0.2, color="blue")
    axes[1,0].set_xlabel("Temperature (K)")
    axes[1,0].set_ylabel("Si-O-Si (°)")
    axes[1,0].set_title("(c) Si-O-Si angle ± std")
    axes[1,0].grid(alpha=0.2)
    axes[1,0].spines["top"].set_visible(False)
    axes[1,0].spines["right"].set_visible(False)

    # Tetrahedral fraction
    axes[1,1].plot(temps, tet, "r-", lw=1.5)
    axes[1,1].set_xlabel("Temperature (K)")
    axes[1,1].set_ylabel("Fraction")
    axes[1,1].set_title("(d) Tetrahedral fraction")
    axes[1,1].set_ylim(0, 1.05); axes[1,1].grid(alpha=0.2)
    axes[1,1].spines["top"].set_visible(False)
    axes[1,1].spines["right"].set_visible(False)

    plt.suptitle(f"Dynamic MoE-7exp — Single-run MD ({structure_name})",
                 fontsize=12)
    plt.tight_layout()
    for fmt in ("png","pdf"):
        fpath = Path(output_dir)/f"weight_evolution_{structure_name}.{fmt}"
        fig.savefig(str(fpath), dpi=300, bbox_inches="tight",
                    facecolor="white")
        print(f"  Saved: {fpath}")
    plt.close(fig)


# ── CLI runner ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Single-run MoE MD with persistent LAMMPS instances")
    ap.add_argument("--gate_ckpt",   required=True)
    ap.add_argument("--structure",   required=True,
                    help="LAMMPS data file or xyz")
    ap.add_argument("--pot_dir",     required=True,
                    help="Directory with Tersoff/Vashishta potential files")
    ap.add_argument("--lmp_lib",     default=LMP_LIB)
    ap.add_argument("--lmp_python",  default=LMP_PYTHON)
    ap.add_argument("--T_start",     type=float, default=300.0)
    ap.add_argument("--T_end",       type=float, default=4000.0)
    ap.add_argument("--heat_rate",   type=float, default=1.7,
                    help="K/ps heating rate")
    ap.add_argument("--dt_fs",       type=float, default=1.0)
    ap.add_argument("--equil_ps",    type=float, default=5.0)
    ap.add_argument("--update_every",type=int,   default=100)
    ap.add_argument("--smooth_alpha",type=float, default=0.9)
    ap.add_argument("--active_thresh",type=float,default=0.03)
    ap.add_argument("--consistent_forces", action="store_true",
                    help="Add finite-difference gate-gradient force correction. Use only for small tests.")
    ap.add_argument("--consistent_stress", action="store_true",
                    help="Use finite-difference strain derivative of the MoE energy for stress. Use only for small tests.")
    ap.add_argument("--gate_fd_delta", type=float, default=1e-4)
    ap.add_argument("--strain_fd_delta", type=float, default=1e-5)
    ap.add_argument("--gate_fd_max_atoms", type=int, default=64)
    ap.add_argument("--stress_fd_sign", type=float, default=1.0,
                    help="Multiplier for finite-difference stress sign; use -1 only for sign diagnostics.")
    ap.add_argument("--is_zeolite",  action="store_true")
    ap.add_argument("--output_dir",  default="moe_single_run")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load structure
    from ase.io import read as ase_read
    from ase import units
    from ase.md.nvtberendsen import NVTBerendsen
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.io.trajectory import Trajectory

    struct_path = Path(args.structure)
    try:
        atoms = ase_read(str(struct_path), format="lammps-data",
                         atom_style="atomic")
    except:
        atoms = ase_read(str(struct_path), format="lammps-data",
                         atom_style="charge")
    atoms.set_pbc(True)
    masses = atoms.get_masses()
    atoms.set_chemical_symbols(
        ["Si" if abs(m-SI_MASS)<1.0 else "O" for m in masses])

    struct_name = struct_path.stem
    print(f"\nStructure: {struct_name} ({len(atoms)} atoms)")

    # Create calculator
    calc = MoECalculator(
        gate_ckpt    = args.gate_ckpt,
        pot_dir      = args.pot_dir,
        lmp_lib      = args.lmp_lib,
        lmp_python   = args.lmp_python,
        active_thresh= args.active_thresh,
        update_every = args.update_every,
        smooth_alpha = args.smooth_alpha,
        is_zeolite   = args.is_zeolite,
        verbose      = True,
        consistent_forces = args.consistent_forces,
        consistent_stress = args.consistent_stress,
        gate_fd_delta = args.gate_fd_delta,
        strain_fd_delta = args.strain_fd_delta,
        gate_fd_max_atoms = args.gate_fd_max_atoms,
        stress_fd_sign = args.stress_fd_sign,
        stress_fd_scale = args.stress_fd_scale,
        force_correction_method = args.force_correction_method,
        smooth_cutoff = args.smooth_cutoff,
        smooth_width = args.smooth_width,
        tetra_sigma = args.tetra_sigma,
    )
    atoms.calc = calc

    dt         = args.dt_fs * units.fs
    n_equil    = int(args.equil_ps * 1000 / args.dt_fs)
    total_ps   = (args.T_end - args.T_start) / args.heat_rate
    n_heat     = int(total_ps * 1000 / args.dt_fs)
    log_every  = args.update_every

    print(f"\nMD protocol:")
    print(f"  Equilibrate: {args.equil_ps} ps at {args.T_start} K")
    print(f"  Heat:        {total_ps:.0f} ps  {args.T_start}→{args.T_end} K")
    print(f"  Weights updated every {args.update_every} steps")

    traj = Trajectory(str(out/f"{struct_name}.traj"), "w", atoms)
    log  = open(out/f"{struct_name}.log", "w")
    log.write("step,temp_K,energy_eV\n")

    # Equilibrate
    print(f"\n[Stage 1] Equilibrating at {args.T_start} K...")
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.T_start)
    dyn = NVTBerendsen(atoms, dt, temperature_K=args.T_start,
                       taut=0.1*units.ps)
    dyn.run(n_equil)
    traj.write(atoms)

    # Heat
    print(f"\n[Stage 2] Heating {args.T_start}→{args.T_end} K...")
    step_counter = [0]

    def log_step():
        s = step_counter[0]; step_counter[0] += 1
        T_now = args.T_start + args.heat_rate * s * args.dt_fs / 1000.0
        T_now = min(T_now, args.T_end)
        dyn_heat.set_temperature(temperature_K=T_now)
        if s % log_every == 0:
            T_act = atoms.get_temperature()
            E     = atoms.get_potential_energy()
            log.write(f"{n_equil+s},{T_act:.2f},{E:.6f}\n")
            log.flush()
        if s % (log_every*10) == 0:
            traj.write(atoms)

    dyn_heat = NVTBerendsen(atoms, dt,
                             temperature_K=args.T_start,
                             taut=0.1*units.ps)
    dyn_heat.attach(log_step, interval=1)
    dyn_heat.run(n_heat)

    log.close(); traj.close()

    # Save and plot
    calc.save_history(str(out))
    plot_weight_evolution(str(out), struct_name)
    calc.close()

    print(f"\n✅ Single-run MoE MD complete! Results in {out}/")


if __name__ == "__main__":
    main()
