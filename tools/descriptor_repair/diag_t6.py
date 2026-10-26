import importlib.util, sys, os, csv, numpy as np, torch
from pathlib import Path
from ase import Atoms
from ase.neighborlist import neighbor_list
spec = importlib.util.spec_from_file_location("rc", "rotinv_cluster_check.py")
rc = importlib.util.module_from_spec(spec); sys.modules["rc"] = rc; spec.loader.exec_module(rc)

STAGE, MOD, RID = sys.argv[1], sys.argv[2], sys.argv[3]
rows = list(csv.DictReader(open(os.path.join(STAGE, "manifest.csv"), newline="")))
row = next(r for r in rows if r["source_file"] == RID)
assert row["split"] in ("train", "val"), "refusing a non train/val record"
f = np.load(Path(STAGE) / row["output_file"], allow_pickle=False)
at = Atoms(numbers=f["atomic_numbers"].astype(int), positions=np.asarray(f["positions"], float),
           cell=np.asarray(f["cell"], float), pbc=True)
print("%s  split=%s  N=%d  %s  cellpar=%s"
      % (RID, row["split"], len(at), at.get_chemical_formula(),
         np.round(at.get_cell().cellpar(), 3)))

prod = rc.load_production(MOD); C = rc.make_corrected_class(prod)
d = C().to(torch.float64).eval(); m = rc.Diag(d.soap_dim, 20261025, torch.float64).eval()
dev = torch.device("cpu"); rcut = float(d.config.r_cut)

ii, jj, dd = neighbor_list("ijd", at, rcut + 0.05)
g = np.abs(dd - rcut)
per_atom = np.full(len(at), np.inf)
for a_, gv in zip(ii, g):
    per_atom[a_] = min(per_atom[a_], gv)
print("neighbour pairs within r_cut+0.05 : %d (all periodic images)" % len(dd))
for t in (1e-3, 2e-3, 5e-3):
    print("pairs within %.0e A of r_cut      : %d" % (t, int((g < t).sum())))
print("closest any pair gets to r_cut    : %.3e A" % g.min())

e0, f0, _ = rc.efs(d, m, at, torch.float64, dev, stress=False)
print("diagnostic energy = %.6f eV   rms|F| = %.4f eV/A" % (e0, np.sqrt((f0 ** 2).mean())))
for h in (1e-3, 5e-4):
    print("  roundoff floor eps*|E|/h at h=%.0e : %.3e eV/A" % (h, 2.22e-16 * abs(e0) / h))

HS = (2e-3, 1e-3, 5e-4)
print("\n%-12s %-10s %-10s %-10s %-11s %-6s %s"
      % ("component", "dev 2e-3", "dev 1e-3", "dev 5e-4", "Richardson", "order",
         "min|r-rcut| of this atom"))
rng = np.random.default_rng(7); worst = (0.0, None)
for i in rng.choice(len(at), 10, replace=False):
    i = int(i); gi = float(per_atom[i])
    for k in range(3):
        fd = {}
        for h in HS:
            v = []
            for sg in (1, -1):
                cc = at.copy(); pp = at.get_positions().copy()
                pp[i, k] += sg * h; cc.set_positions(pp)
                v.append(rc.efs(d, m, cc, torch.float64, dev, stress=False)[0])
            fd[h] = -(v[0] - v[1]) / (2 * h)
        devs = {h: abs(fd[h] - f0[i, k]) for h in HS}
        rich = abs((4 * fd[5e-4] - fd[1e-3]) / 3 - f0[i, k])
        order = (np.log(devs[1e-3] / devs[5e-4]) / np.log(2)) if devs[5e-4] > 0 else float("nan")
        print("atom %-5d %-2s %.3e %.3e %.3e %.3e %6.2f  %.3e%s"
              % (i, "xyz"[k], devs[2e-3], devs[1e-3], devs[5e-4], rich, order, gi,
                 "  <-- within a step of the cutoff" if gi < 3e-3 else ""))
        if rich > worst[0]:
            worst = (rich, (i, k, gi, order))
print("\nworst Richardson residual %.3e eV/A at atom %d %s; that atom's min|r-rcut| "
      "= %.3e A; observed order %.2f"
      % (worst[0], worst[1][0], "xyz"[worst[1][1]], worst[1][2], worst[1][3]))
