#!/usr/bin/env python3
"""
torchsoap_autograd_module.py

Standalone differentiable SOAP-like descriptor for SiO2 MoE gate development.

Purpose
-------
This file is intentionally separate from moe_persistent_calculator.py.
It provides a PyTorch-autograd descriptor path:

    positions R -> Torch SOAP-like descriptor -> SOAP gate weights -> d w / dR

Important
---------
This is NOT guaranteed to be numerically identical to DScribe SOAP. It is a
separate differentiable SOAP-like implementation with the same high-level output
layout used by the existing SOAP gate:

    [mean SOAP over Si centers, mean SOAP over O centers, vol_per_atom, density, is_zeolite]

For n_max=6, l_max=4, species=[Si,O], the descriptor length is:

    2 * 390 + 3 = 783

Before using this with an existing DScribe-trained checkpoint, compare the
predicted gate weights. If weights differ significantly, retrain the gate using
this Torch descriptor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


@dataclass
class TorchSOAPConfig:
    species: Tuple[str, ...] = ("Si", "O")
    r_cut: float = 5.0
    n_max: int = 6
    l_max: int = 4
    radial_sigma: float = 0.50
    eps: float = 1.0e-12
    normalize_power: bool = True
    power_norm_lambda: float = 1.0e-2
class TorchSOAPDescriptor(torch.nn.Module):
    """
    Differentiable SOAP-like descriptor using PyTorch operations.

    The per-center descriptor dimension is:

        n_pair_channels * (l_max + 1)

    where:

        n_channels = n_species * n_max
        n_pair_channels = n_channels * (n_channels + 1) / 2

    For species=(Si,O), n_max=6, l_max=4:

        n_channels = 12
        n_pair_channels = 78
        dim = 78 * 5 = 390

    The global gate descriptor is:

        mean over Si centers: 390
        mean over O centers:  390
        scalars: vol_per_atom, density, is_zeolite
        total: 783
    """

    def __init__(self, config: TorchSOAPConfig | None = None):
        super().__init__()
        self.config = config or TorchSOAPConfig()
        self.species_to_index: Dict[str, int] = {
            s: i for i, s in enumerate(self.config.species)
        }

        # Radial centers. We avoid exactly zero to reduce singular behavior near r=0.
        centers = torch.linspace(
            0.25 * self.config.r_cut / self.config.n_max,
            self.config.r_cut,
            self.config.n_max,
            dtype=torch.float64,
        )
        self.register_buffer("radial_centers", centers)

        # Upper triangular channel-pair indices for power spectrum.
        n_channels = len(self.config.species) * self.config.n_max
        pair_i, pair_j = torch.triu_indices(n_channels, n_channels)
        self.register_buffer("pair_i", pair_i)
        self.register_buffer("pair_j", pair_j)

    @property
    def soap_dim(self) -> int:
        n_channels = len(self.config.species) * self.config.n_max
        n_pairs = n_channels * (n_channels + 1) // 2
        return n_pairs * (self.config.l_max + 1)

    @property
    def output_dim(self) -> int:
        return 2 * self.soap_dim + 3

    def forward(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        symbols: Sequence[str],
        masses: Sequence[float] | np.ndarray,
        is_zeolite: bool = False,
    ) -> torch.Tensor:
        """
        Return global 783-feature descriptor for a structure.

        Parameters
        ----------
        positions
            Cartesian coordinates, shape (N,3), requires_grad=True if gradients
            with respect to atomic positions are needed.
        cell
            ASE-style cell matrix with cell vectors as rows, shape (3,3).
        symbols
            Chemical symbols, length N. Expected symbols: Si and O.
        masses
            Atomic masses in amu, length N.
        is_zeolite
            Appended as scalar 0/1 feature.
        """
        cfg = self.config
        dtype = positions.dtype
        device = positions.device

        center_desc = self.compute_per_center(positions, cell, symbols)
        syms = np.array(list(symbols))
        si_mask = torch.as_tensor(syms == "Si", dtype=torch.bool, device=device)
        o_mask = torch.as_tensor(syms == "O", dtype=torch.bool, device=device)

        if torch.any(si_mask):
            si_mean = center_desc[si_mask].mean(dim=0)
        else:
            si_mean = torch.zeros(self.soap_dim, dtype=dtype, device=device)

        if torch.any(o_mask):
            o_mean = center_desc[o_mask].mean(dim=0)
        else:
            o_mean = torch.zeros(self.soap_dim, dtype=dtype, device=device)

        volume = torch.abs(torch.det(cell))
        vol_per_atom = volume / positions.shape[0]

        masses_t = torch.as_tensor(masses, dtype=dtype, device=device)
        # amu/A^3 -> g/cm^3 since 1 amu/A^3 = 1.66054 g/cm^3
        density = masses_t.sum() * 1.66054 / (volume + cfg.eps)
        isz = torch.tensor(float(is_zeolite), dtype=dtype, device=device)

        scalars = torch.stack([vol_per_atom, density, isz])
        return torch.cat([si_mean, o_mean, scalars], dim=0)

    def compute_per_center(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        symbols: Sequence[str],
    ) -> torch.Tensor:
        """
        Return periodic SOAP-like power spectrum for every atomic center.

        Shape
        -----
        (N, soap_dim)

        Important PBC detail
        --------------------
        The original implementation selected exactly one minimum-image
        vector for each ordered atom pair (i,j).  That is only sufficient
        when r_cut is smaller than half the relevant periodic repeat
        distance.

        Here we instead include ALL periodic images that can lie within
        r_cut.  The fractional pair displacement is first mapped into a
        central image, and a finite set of neighboring lattice
        translations is then enumerated.

        Only the true self interaction
            i == j and lattice shift == (0,0,0)
        is excluded.  Periodic self-images at nonzero lattice shifts are
        legitimate neighbors if they lie inside r_cut.

        The radial basis, angular basis, cutoff, channel ordering, power
        spectrum, and normalization are unchanged from the original
        implementation.
        """
        cfg = self.config
        dtype = positions.dtype
        device = positions.device

        n_atoms = positions.shape[0]
        n_species = len(cfg.species)
        n_channels = n_species * cfg.n_max

        # --------------------------------------------------------------
        # Fractional pair vectors in a central periodic image.
        #
        # ASE convention here:
        #     Cartesian = fractional @ cell
        #
        # dfrac[i,j] corresponds to R_j - R_i.
        # --------------------------------------------------------------
        inv_cell = torch.inverse(cell)

        frac = positions @ inv_cell
        dfrac = frac[None, :, :] - frac[:, None, :]

        # Central componentwise image.  Although round() itself is
        # piecewise constant, the full periodic-image SUM below is
        # invariant to which equivalent central representation is used,
        # provided all images capable of entering r_cut are enumerated.
        dfrac = dfrac - torch.round(dfrac)

        dcart0 = dfrac @ cell

        # --------------------------------------------------------------
        # Determine how many lattice translations are needed along each
        # fractional direction.
        #
        # Fractional coordinate f_k has plane spacing
        #
        #     h_k = 1 / || column_k(cell^{-1}) ||
        #
        # Since central fractional differences lie in [-1/2,1/2],
        # an image with integer shift n_k can only be within r_cut if
        #
        #     (|n_k| - 1/2) h_k < r_cut.
        #
        # Thus floor(r_cut/h_k + 1/2) is a sufficient per-direction
        # translation extent.
        #
        # The integer range is a neighbor-list topology decision, so it
        # is intentionally detached from autograd.  Forces/stress still
        # differentiate every included image vector through dcart0 and
        # cell.
        # --------------------------------------------------------------
        with torch.no_grad():
            inv_detached = inv_cell.detach()

            plane_spacing = 1.0 / torch.linalg.norm(
                inv_detached,
                dim=0,
            )

            nmax_t = torch.floor(
                positions.new_tensor(float(cfg.r_cut))
                / plane_spacing
                + 0.5
                + 1.0e-12
            ).to(torch.long)

            nx = int(nmax_t[0].item())
            ny = int(nmax_t[1].item())
            nz = int(nmax_t[2].item())

        shifts = [
            (a, b, c)
            for a in range(-nx, nx + 1)
            for b in range(-ny, ny + 1)
            for c in range(-nz, nz + 1)
        ]

        # Print once per descriptor instance so the diagnostic explicitly
        # records which periodic image set was used.
        if not getattr(self, "_periodic_shift_info_printed", False):
            ps = plane_spacing.detach().cpu().numpy()

            print(
                "[TorchSOAP PERIODIC] "
                f"r_cut={cfg.r_cut:.6f} A | "
                f"plane spacings=({ps[0]:.6f}, "
                f"{ps[1]:.6f}, {ps[2]:.6f}) A | "
                f"shift extents=({nx},{ny},{nz}) | "
                f"n_shifts={len(shifts)}"
            )

            self._periodic_shift_info_printed = True

        # --------------------------------------------------------------
        # Neighbor species.
        # --------------------------------------------------------------
        syms = list(symbols)

        neigh_species = torch.empty(
            n_atoms,
            dtype=torch.long,
            device=device,
        )

        for idx, sym in enumerate(syms):
            if sym not in self.species_to_index:
                raise ValueError(
                    f"Unknown species {sym!r}; expected {cfg.species}"
                )

            neigh_species[idx] = self.species_to_index[sym]

        species_masks = [
            (neigh_species == sp_idx).to(dtype)
            for sp_idx in range(n_species)
        ]

        eye = torch.eye(
            n_atoms,
            dtype=torch.bool,
            device=device,
        )

        # --------------------------------------------------------------
        # Accumulate c_l coefficients over ALL periodic images:
        #
        # c_{i,q,lm} =
        #     sum_{j,T} R_q(r_{ijT}) f_c(r_{ijT}) Y_lm(rhat_{ijT})
        #
        # with every translation T whose image lies within r_cut.
        # --------------------------------------------------------------
        coeff_by_l: List[torch.Tensor] = []

        for l in range(cfg.l_max + 1):
            n_m = 2 * l + 1

            coeff_by_l.append(
                torch.zeros(
                    (n_atoms, n_channels, n_m),
                    dtype=dtype,
                    device=device,
                )
            )

        for sa, sb, sc in shifts:

            shift_frac = positions.new_tensor(
                [float(sa), float(sb), float(sc)]
            )

            shift_cart = shift_frac @ cell

            # Pair vectors for this lattice image.
            dcart = dcart0 + shift_cart

            # Preserve the same distance convention as the original code.
            r = torch.linalg.norm(
                dcart + cfg.eps,
                dim=-1,
            )

            in_cut = r < cfg.r_cut

            # Exclude ONLY the true home-cell self interaction.
            if sa == 0 and sb == 0 and sc == 0:
                in_cut = in_cut & (~eye)

            cutoff = (
                self._cosine_cutoff(r)
                * in_cut.to(dtype)
            )

            # Nothing from this translation contributes.
            # Using .any().item() is only a control-flow optimization;
            # the included-image energy/force graph is unchanged.
            if not bool(in_cut.any().detach().item()):
                continue

            # Same radial basis as before:
            # shape (N,N,n_max)
            radial = (
                self._radial_basis(r)
                * cutoff[..., None]
            )

            # Same angular basis as before.
            # ylm_l shape = (N,N,2l+1)
            ylm_by_l = self._real_sph_harm_basis(
                dcart,
                r,
            )

            new_coeff_by_l: List[torch.Tensor] = []

            for l, ylm in enumerate(
                ylm_by_l[: cfg.l_max + 1]
            ):
                species_contrib = []

                for sp_idx in range(n_species):

                    # Keep only neighbors of this chemical species.
                    #
                    # rb shape:
                    #     (center i, neighbor j, radial n)
                    rb = (
                        radial
                        * species_masks[sp_idx][None, :, None]
                    )

                    # Sum over periodic neighbor atoms j:
                    #
                    #   rb[i,j,n] * Y[i,j,m]
                    #
                    # -> (i,n,m)
                    contrib = torch.einsum(
                        "ijn,ijm->inm",
                        rb,
                        ylm,
                    )

                    species_contrib.append(contrib)

                # Channel ordering remains
                #
                #   species_0: n=0...n_max-1
                #   species_1: n=0...n_max-1
                #
                # exactly as in the original implementation.
                contrib_all = torch.cat(
                    species_contrib,
                    dim=1,
                )

                # Out-of-place accumulation keeps the autograd graph
                # clean across lattice translations.
                new_coeff_by_l.append(
                    coeff_by_l[l] + contrib_all
                )

            coeff_by_l = new_coeff_by_l

        # --------------------------------------------------------------
        # Same power-spectrum construction as original implementation.
        # --------------------------------------------------------------
        power_by_l: List[torch.Tensor] = []

        for coeff in coeff_by_l:

            ci = coeff[:, self.pair_i, :]
            cj = coeff[:, self.pair_j, :]

            p_l = torch.sum(
                ci * cj,
                dim=-1,
            )

            power_by_l.append(p_l)

        desc = torch.cat(
            power_by_l,
            dim=-1,
        )

        if cfg.normalize_power:
            norm = torch.linalg.norm(
                desc,
                dim=-1,
                keepdim=True,
            )

            desc = desc / torch.sqrt(
                norm * norm
                + desc.new_tensor(
                    float(cfg.power_norm_lambda)
                ) ** 2
            )

        return desc


    def _cosine_cutoff(self, r: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        x = torch.clamp(r / cfg.r_cut, min=0.0, max=1.0)
        fc = 0.5 * (torch.cos(np.pi * x) + 1.0)
        return torch.where(r < cfg.r_cut, fc, torch.zeros_like(fc))

    def _radial_basis(self, r: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        centers = self.radial_centers.to(dtype=r.dtype, device=r.device)
        sigma = torch.tensor(cfg.radial_sigma, dtype=r.dtype, device=r.device)
        rb = torch.exp(-0.5 * ((r[..., None] - centers) / sigma) ** 2)
        return rb

    def _real_sph_harm_basis(
        self,
        dcart: torch.Tensor,
        r: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Polynomial real spherical-harmonic-like basis up to l=4.

        These channels are differentiable and rotationally structured, but they
        are not normalized to exactly match DScribe. They are suitable for a
        retrained Torch-SOAP gate or for prototyping autograd force correction.
        """
        cfg = self.config
        eps = cfg.eps
        dtype = dcart.dtype
        device = dcart.device

        x = dcart[..., 0] / (r + eps)
        y = dcart[..., 1] / (r + eps)
        z = dcart[..., 2] / (r + eps)
        one = torch.ones_like(x)

        basis: List[torch.Tensor] = []

        # l=0: 1 channel
        basis.append(one[..., None])

        if cfg.l_max >= 1:
            basis.append(torch.stack([x, y, z], dim=-1))

        if cfg.l_max >= 2:
            l2 = torch.stack([
                x * y,
                y * z,
                0.5 * (3.0 * z * z - 1.0),
                x * z,
                0.5 * (x * x - y * y),
            ], dim=-1)
            basis.append(l2)

        if cfg.l_max >= 3:
            l3 = torch.stack([
                y * (3.0 * x * x - y * y),
                x * y * z,
                y * (5.0 * z * z - 1.0),
                z * (5.0 * z * z - 3.0),
                x * (5.0 * z * z - 1.0),
                z * (x * x - y * y),
                x * (x * x - 3.0 * y * y),
            ], dim=-1)
            basis.append(l3)

        if cfg.l_max >= 4:
            l4 = torch.stack([
                x * y * (x * x - y * y),
                y * z * (3.0 * x * x - y * y),
                x * y * (7.0 * z * z - 1.0),
                y * z * (7.0 * z * z - 3.0),
                35.0 * z ** 4 - 30.0 * z * z + 3.0,
                x * z * (7.0 * z * z - 3.0),
                (x * x - y * y) * (7.0 * z * z - 1.0),
                x * z * (x * x - 3.0 * y * y),
                x ** 4 - 6.0 * x * x * y * y + y ** 4,
            ], dim=-1)
            basis.append(l4)

        return basis


def compute_torchsoap_features_from_ase(
    atoms,
    config: TorchSOAPConfig | None = None,
    is_zeolite: bool = False,
    requires_grad: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience wrapper for ASE Atoms.

    Returns
    -------
    features
        Torch tensor of shape (783,) for default Si/O nmax=6 lmax=4.
    positions
        Position tensor. If requires_grad=True, this is the tensor to use for
        torch.autograd.grad(..., positions).
    """
    desc = TorchSOAPDescriptor(config)
    positions = torch.tensor(
        atoms.get_positions(), dtype=torch.float64, requires_grad=requires_grad
    )
    cell = torch.tensor(atoms.get_cell().array, dtype=torch.float64)
    symbols = atoms.get_chemical_symbols()
    masses = atoms.get_masses()
    features = desc(positions, cell, symbols, masses, is_zeolite=is_zeolite)
    return features, positions


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + eps))


if __name__ == "__main__":
    print("TorchSOAPDescriptor module loaded.")
    cfg = TorchSOAPConfig()
    model = TorchSOAPDescriptor(cfg)
    print(f"Per-center SOAP dim: {model.soap_dim}")
    print(f"Global gate descriptor dim: {model.output_dim}")

