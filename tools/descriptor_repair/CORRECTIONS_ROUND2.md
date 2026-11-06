# Corrections to audit round 2

An independent review of the round-2 package found six classes of overstatement.
All are corrected in the reports as of this commit. They are listed here rather
than applied silently, so that anyone who read the earlier text can see exactly
what changed and why.

## 1. own_0146 was described as event-driven; it is not

The earlier text said the drift was "event-driven, not accumulating" and
attributed it to a structural rearrangement that the timestep could not integrate
through. The trajectory contradicts the first part and nothing establishes the
second.

Measured: the criterion is first exceeded at **127.75 ps**, not around 150 ps;
**1490 of 2001** samples lie above it; and **61%** of the final drift (+0.302297
of +0.492553) had accumulated before the largest sampled increment of +0.209973
between 273.50 and 273.75 ps. The next four increments are +0.110351, +0.055894,
+0.037599 and +0.037402, so the largest is not isolated.

The rearrangement and timestep mechanism were inferences from energy and
temperature traces alone. No structural analysis was performed and no
smaller-timestep run was made. Both claims are withdrawn; the temperature and
energy observations remain, labelled as observations.

The duration was described as "two orders of magnitude" beyond 10 ps. It is
**50 times**.

## 2. Drift significance claims are withdrawn

The earlier text said `own_0146`'s slope was "about 128 sigma" and that none of
the other nine had a statistically resolvable slope. Both are wrong. Six of ten
have abs(slope)/SE above 4 (19.8, 6.6, 5.7, 5.5, 4.8 besides own_0146's 127.8),
so the second claim is false on its own numbers. And the residuals are strongly
autocorrelated, so OLS standard errors understate uncertainty and none of these
ratios supports a significance statement. Slopes are now reported descriptively
with their OLS standard errors and an explicit caveat.

## 3. Cost statistics now name their population and baseline

`median_of_medians_ms`, `peak_mem_max_GiB` (a **maximum**, not a median) and the
paired per-configuration speedups were being quoted interchangeably. They are now
three separate tables. The headline paired medians are 8.676736x against MACE
float64 and 6.929799x against float32, with median paired memory reductions of
98.081212% and 96.175662%, all over n = 105.

The image-enumerating subset was reported as "ten of 63 markers". Per
configuration it is **14 of 105**, median 2.912119x against 9.173889x for the
other 91; the ten-of-63 figure belongs to the family-merged grouping used in the
figure. Both are now given, each labelled.

## 4. Verification coverage is stated precisely

Stress covariance is evaluated on the **calculator routes only**; the trainer
route returns no stress, and the earlier table printed `0.000e+00` for it, which
turned a missing measurement into a perfect score. It now reads "not evaluated".

The conservativity check samples **six random (atom, component) pairs** for forces
and **three reciprocal pairs between atom 0 and its nearest neighbour** for the
Hessian. It is not the full Hessian and the earlier "curl-free to 7e-8 relative"
overstated the scope. The h and h/2 agreement is now described as consistent with
second-order truncation over the two steps tested, rather than as establishing an
order.

## 5. PSO2-BKS is small, not unused

The earlier text rounded PSO2-BKS to 0.000 and concluded the channel was "not
exercised by this cohort". The weight is nonzero in **all 52 structures**, mean
2.058e-04, range 1.023e-04 to 4.908e-04, with no exact zeros. The conclusion came
from the rounding and is withdrawn.

Figure subsets and ordering are now mapped explicitly for both retained routing
figures, including which regenerated data each draws on.

## 6. Provenance distinguishes contemporaneous from post hoc

The sidecars state in their own `note` field that their hashes were written after
the fact, and on their own they verify only that the artifacts **as they exist
now** match the pinned values.

The 500 ps executions do carry contemporaneous evidence: all **19** Slurm logs
print the four artifact hashes, the runner hash and the trainer hash before
integration begins, and nine of them document restart points. Those logs are
added to the package in this commit. The two kinds of evidence agree, and the
reports now say which is which rather than merging them.

Ten records correspond to 19 executions; each summary's `slurm_job` field names
only the last one.
