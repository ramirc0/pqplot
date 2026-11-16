"""Local ancestry painted on both homologs of each chromosome."""

from itertools import pairwise

import numpy as np

import pqplot as pqp

hg38 = pqp.Genome.bundled("hg38")
chroms = hg38.autosomes[:6]
rng = np.random.default_rng(4)
tracts = {"chrom": [], "start": [], "end": [], "ancestry": [], "hap": []}
for c in chroms:
    for hap in (1, 2):
        cuts = np.concatenate([[0], np.sort(rng.uniform(0, hg38[c].length, 4)), [hg38[c].length]])
        for lo, hi in pairwise(cuts):
            tracts["chrom"].append(c)
            tracts["start"].append(lo)
            tracts["end"].append(hi)
            tracts["ancestry"].append(rng.choice(["AFR", "EUR", "NAT"]))
            tracts["hap"].append(hap)

kary = pqp.Karyogram(hg38, chroms=chroms, copies=2, bands=False)
kary.paint(tracts, start="start", end="end", color="ancestry", copy="hap")
kary.legend(title="Ancestry")
fig = kary.figure
