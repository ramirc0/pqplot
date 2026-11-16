"""Feature density per megabase painted on unbanded bodies."""

import numpy as np

import pqplot as pqp

hg38 = pqp.Genome.bundled("hg38")
rng = np.random.default_rng(3)
chrom = np.concatenate([np.full(c.length // 1_000_000 + 1, c.name) for c in hg38])
start = np.concatenate([np.arange(0, c.length, 1_000_000) for c in hg38])
bins = {"chrom": chrom, "start": start, "end": start + 1_000_000}
bins["per_mb"] = rng.gamma(2, 100, len(start))

kary = pqp.Karyogram(hg38, bands=False)
heat = kary.paint(bins, start="start", end="end", value="per_mb", cmap="magma")
kary.colorbar(heat, label="cCREs per Mb")
fig = kary.figure
