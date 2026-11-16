"""A Manhattan plot on concatenated chromosomes, named at their midpoints."""

import matplotlib.pyplot as plt
import numpy as np

import pqplot as pqp

hg38 = pqp.Genome.bundled("hg38")
linear = hg38.concatenated(gap=5_000_000)
rng = np.random.default_rng(2)
chrom = rng.choice([c.name for c in hg38], 20_000)
pos = rng.uniform(0, [hg38[c].length for c in chrom])
neg_log10_p = rng.exponential(1, len(pos))
peaks = rng.choice(len(pos), 12, replace=False)
neg_log10_p[peaks] += rng.uniform(8, 20, len(peaks))

fig, ax = plt.subplots(figsize=(6.5, 2.5))
ax.scatter(linear.position(chrom, pos), neg_log10_p, s=2, linewidths=0)
ax.axhline(-np.log10(5e-8), color="C3", linewidth=0.5)
ax.set_ylabel("$-\\log_{10} p$")
pqp.chromosome_axis(linear, ax)
