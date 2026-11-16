"""Gene density and copy number in tracks above and below horizontal chromosomes."""

import numpy as np

import pqplot as pqp

hg38 = pqp.Genome.bundled("hg38")
chroms = ["chr1", "chr2", "chr3", "chr7", "chr17"]
rng = np.random.default_rng(1)
chrom = np.concatenate([np.full(hg38[c].length // 1_000_000 + 1, c) for c in chroms])
start = np.concatenate([np.arange(0, hg38[c].length, 1_000_000) for c in chroms])
end = start + 1_000_000
log2 = rng.normal(0, 0.3, len(start)) + np.where(chrom == "chr7", 0.6, 0)
bins = {"chrom": chrom, "start": start, "end": end, "mid": (start + end) / 2, "log2": log2}
bins["per_mb"] = rng.gamma(2, 6, len(start))

kary = pqp.Karyogram(
    hg38,
    orientation="horizontal",
    chroms=chroms,
    tracks=[pqp.Track("genes", side="before"), pqp.Track("cnv", side="after", size=3)],
)
kary.track("genes").area(bins, value="per_mb", linewidth=0)
kary.track("cnv").scatter(bins, pos="mid", value="log2", s=3, c=log2, cmap="coolwarm")
kary.panel("chr1", track="cnv").set_ylim(-2, 2)
fig = kary.figure
