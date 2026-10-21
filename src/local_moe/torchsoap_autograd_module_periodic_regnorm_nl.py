#!/usr/bin/env python3
"""
torchsoap_autograd_module_periodic_regnorm_nl.py

Neighbour-list version of torchsoap_autograd_module_periodic_regnorm.py.

What changed relative to the original (and ONLY this)
-----------------------------------------------------
* ``compute_per_center`` now builds an edge list (i, j, lattice shift S) with a
  cell-list neighbour search and evaluates the radial / angular basis on those
  edges only.  Cost and memory are O(N * n_neighbours) instead of
  O(N^2 * n_shifts).
* The original dense implementation is kept verbatim as
  ``compute_per_center_dense`` so the two can be compared directly.
* ``TorchSOAPConfig`` gained two fields with defaults (``nl_skin``,
  ``nl_backend``); existing code that builds the config with
  species / r_cut / n_max / l_max is unaffected.
* The per-atom species lookup no longer writes one GPU element at a time.

What did NOT change: radial basis, angular basis, cosine cutoff, the
``r = |d + eps| < r_cut`` selection rule, channel ordering, power spectrum and
its normalisation, the global 783-feature layout in ``forward``.  The descriptor
is therefore the same function; trained routing / common / residual weights
stay valid.

The neighbour TOPOLOGY (which (i, j, S) triples exist) is a detached integer
decision, exactly as the shift range was in the dense version.  Pair vectors
d = R_j - R_i + S @ cell are rebuilt from the differentiable positions and cell,
so forces (and the cell gradient) are exact derivatives of the same energy.

Original header
---------------
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

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
    # --- added for the neighbour-list path (defaults keep old call sites valid)
    nl_skin: float = 1.0e-6      # search radius = r_cut + nl_skin; exact test is r < r_cut
    # "auto" (= "torch") | "torch" (GPU cell list, no host copy) | "matscipy" | "ase".
    # Override without editing code:  export TORCHSOAP_NL_BACKEND=matscipy
    nl_backend: str = field(
        default_factory=lambda: os.environ.get("TORCHSOAP_NL_BACKEND", "auto")
    )


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

    # ------------------------------------------------------------------
    # NEW: neighbour-list implementation
    # ------------------------------------------------------------------
    def _species_index(self, symbols: Sequence[str], device) -> torch.Tensor:
        """Species index per atom, built on the host and moved once."""
        idx = np.empty(len(symbols), dtype=np.int64)
        for k, sym in enumerate(symbols):
            if sym not in self.species_to_index:
                raise ValueError(
                    f"Unknown species {sym!r}; expected {self.config.species}"
                )
            idx[k] = self.species_to_index[sym]
        return torch.from_numpy(idx).to(device)

    @torch.no_grad()
    def _edges_torch(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        r_search: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Cell-list neighbour search in pure torch (runs where ``positions`` live).

        Atoms are binned in FRACTIONAL coordinates with bins at least r_search
        wide measured along the plane normals (bins per direction =
        max(1, floor(h_k / r_search))), so every neighbour lies within +-m_k
        bins, m_k = ceil(r_search * nbins_k / h_k).  Periodic images are tracked
        through the bin wrap:  S = img + wrap_i - wrap_j  reproduces
        R_j - R_i + S @ cell exactly for the ORIGINAL (unwrapped) positions.

        Returns int64 i, j and float64 S (E, 3), already filtered with the same
        distance test the dense code uses, home-cell self pair removed.
        """
        dev = positions.device
        f64 = torch.float64
        eps = float(self.config.eps)

        pos = positions.detach().to(f64)
        cel = cell.detach().to(f64)
        n = pos.shape[0]

        inv = torch.linalg.inv(cel)
        frac = pos @ inv
        h = 1.0 / torch.linalg.norm(inv, dim=0)                       # plane spacings
        nb = torch.clamp(torch.floor(h / r_search).to(torch.long), min=1)
        m = torch.clamp(
            torch.ceil(r_search * nb.to(f64) / h + 1.0e-9).to(torch.long), min=1
        )
        nbx, nby, nbz = (int(v) for v in nb.tolist())
        mx, my, mz = (int(v) for v in m.tolist())

        wrap = torch.floor(frac)                                       # (N, 3) float
        fw = frac - wrap
        b3 = torch.minimum(
            torch.floor(fw * nb.to(f64)).to(torch.long), nb - 1
        ).clamp(min=0)                                                 # (N, 3)
        bflat = (b3[:, 0] * nby + b3[:, 1]) * nbz + b3[:, 2]

        order = torch.argsort(bflat)
        counts = torch.bincount(bflat, minlength=nbx * nby * nbz)
        starts = torch.cumsum(counts, 0) - counts

        def rng(k):
            return torch.arange(-k, k + 1, device=dev)

        offs = torch.stack(
            torch.meshgrid(rng(mx), rng(my), rng(mz), indexing="ij"), dim=-1
        ).reshape(-1, 3)
        n_off = offs.shape[0]

        tgt = b3[:, None, :] + offs[None, :, :]                        # (N, K, 3)
        img = torch.div(tgt, nb, rounding_mode="floor")                # (N, K, 3)
        tb = tgt - img * nb
        tflat = (tb[..., 0] * nby + tb[..., 1]) * nbz + tb[..., 2]     # (N, K)

        cnt = counts[tflat].reshape(-1)
        st = starts[tflat].reshape(-1)
        total = int(cnt.sum().item())
        if total == 0:
            z = torch.zeros(0, dtype=torch.long, device=dev)
            return z, z.clone(), torch.zeros((0, 3), dtype=f64, device=dev)

        src = torch.repeat_interleave(torch.arange(n * n_off, device=dev), cnt)
        first = torch.cumsum(cnt, 0) - cnt
        within = torch.arange(total, device=dev) - torch.repeat_interleave(first, cnt)
        j = order[st[src] + within]
        i = torch.div(src, n_off, rounding_mode="floor")
        S = img.reshape(-1, 3)[src].to(f64) + wrap[i] - wrap[j]

        # exact distance test, identical convention to the dense code
        d = pos[j] - pos[i] + S @ cel
        r = torch.linalg.norm(d + eps, dim=-1)
        keep = r < r_search
        home_self = (i == j) & torch.all(S == 0.0, dim=1)
        keep = keep & ~home_self
        return i[keep], j[keep], S[keep]

    def _neighbour_edges(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        backend: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        All (i, j, S) with |R_j - R_i + S @ cell| < r_cut + nl_skin, fully
        periodic in all three directions (the dense version also ignores ``pbc``
        and always uses every periodic image).  The home-cell self pair
        (i == j, S == 0) is removed; periodic self images (i == j, S != 0) stay.
        Returns torch tensors (int64 i, int64 j, float64 S of shape (E, 3)) on
        the device of ``positions`` and the name of the backend that was used.
        """
        cfg = self.config
        dev = positions.device
        backend = backend or cfg.nl_backend
        if backend == "auto":
            backend = "torch"
        r_search = float(cfg.r_cut) + float(cfg.nl_skin)

        if backend == "torch":
            i, j, S = self._edges_torch(positions, cell, r_search)
            return i, j, S, "torch"

        pos = positions.detach().to("cpu", torch.float64).numpy()
        cel = cell.detach().to("cpu", torch.float64).numpy()

        if backend == "matscipy":
            from matscipy.neighbours import neighbour_list as _ms_nl
            i, j, S = _ms_nl(
                "ijS",
                pbc=np.array([True, True, True]),
                cell=cel,
                positions=pos,
                cutoff=r_search,
            )
        elif backend == "ase":
            from ase import Atoms
            from ase.neighborlist import neighbor_list as _ase_nl

            atoms = Atoms(
                numbers=np.ones(len(pos), dtype=int),
                positions=pos,
                cell=cel,
                pbc=True,
            )
            i, j, S = _ase_nl("ijS", atoms, r_search)
        else:
            raise ValueError(
                f"Unknown nl_backend {backend!r}; use auto|torch|matscipy|ase"
            )

        i = np.asarray(i, dtype=np.int64)
        j = np.asarray(j, dtype=np.int64)
        S = np.asarray(S, dtype=np.float64).reshape(-1, 3)
        home_self = (i == j) & np.all(S == 0.0, axis=1)
        if home_self.any():
            keep = ~home_self
            i, j, S = i[keep], j[keep], S[keep]
        return (
            torch.from_numpy(i).to(dev),
            torch.from_numpy(j).to(dev),
            torch.from_numpy(S).to(dev),
            backend,
        )

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

        Neighbour-list implementation of

            c_{i,q,lm} = sum_{j,T} R_q(r_ijT) f_c(r_ijT) Y_lm(rhat_ijT)

        over every periodic image (j, T) with r_ijT < r_cut, excluding only the
        home-cell self pair.  Identical in value to ``compute_per_center_dense``.
        """
        cfg = self.config
        dtype = positions.dtype
        device = positions.device

        n_atoms = positions.shape[0]
        n_species = len(cfg.species)
        n_channels = n_species * cfg.n_max

        sp_idx = self._species_index(symbols, device)

        # ---- topology: detached integer decision (cell list, O(N)) ----
        ii, jj, S, used = self._neighbour_edges(positions, cell)
        if not getattr(self, "_nl_info_printed", False):
            print(
                "[TorchSOAP NEIGHBOUR-LIST] "
                f"backend={used} | r_cut={cfg.r_cut:.6f} A | "
                f"N={n_atoms} | edges(search)={len(ii)}"
            )
            self._nl_info_printed = True

        S = S.to(dtype=dtype)

        # ---- geometry: differentiable in positions and cell ----
        # d_ijT = R_j - R_i + T @ cell   (ASE row-vector convention)
        dcart = positions[jj] - positions[ii] + S @ cell

        # Same distance convention and same selection rule as the dense code.
        r = torch.linalg.norm(dcart + cfg.eps, dim=-1)
        keep = r < cfg.r_cut
        if not bool(keep.all().detach().item()):
            ii, jj, dcart, r = ii[keep], jj[keep], dcart[keep], r[keep]

        # ---- basis on edges only ----
        cutoff = self._cosine_cutoff(r)                       # (E,)
        radial = self._radial_basis(r) * cutoff[..., None]    # (E, n_max)
        ylm_by_l = self._real_sph_harm_basis(dcart, r)        # list of (E, 2l+1)
        ylm = torch.cat(ylm_by_l[: cfg.l_max + 1], dim=-1)    # (E, (l_max+1)^2)

        # contrib[e, n, m] = radial[e, n] * ylm[e, m]
        contrib = radial[:, :, None] * ylm[:, None, :]

        # Accumulate onto (center i, neighbour species s) -> channel s * n_max + n.
        flat = ii * n_species + sp_idx[jj]
        acc = torch.zeros(
            (n_atoms * n_species, cfg.n_max, ylm.shape[-1]),
            dtype=dtype,
            device=device,
        ).index_add(0, flat, contrib)
        acc = acc.reshape(n_atoms, n_channels, ylm.shape[-1])

        coeff_by_l: List[torch.Tensor] = []
        off = 0
        for l in range(cfg.l_max + 1):
            n_m = 2 * l + 1
            coeff_by_l.append(acc[:, :, off:off + n_m])
            off += n_m

        # --------------------------------------------------------------
        # Same power-spectrum construction as original implementation.
        # --------------------------------------------------------------
        return self._power_spectrum(coeff_by_l)

    def _power_spectrum(self, coeff_by_l: List[torch.Tensor]) -> torch.Tensor:
        cfg = self.config
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

    # ------------------------------------------------------------------
    # ORIGINAL dense implementation, kept verbatim for A/B comparison.
    # ------------------------------------------------------------------
    def compute_per_center_dense(
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
        return self._power_spectrum(coeff_by_l)

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
