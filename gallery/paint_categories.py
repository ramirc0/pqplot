"""Copy-number states painted on the chromosome bodies."""

import pqplot as pqp

hg38 = pqp.Genome.bundled("hg38")
segments = {
    "chrom": ["chr1", "chr1", "chr7", "chr9", "chr9", "chr10", "chr17"],
    "start": [0, 150e6, 0, 21.9e6, 60e6, 0, 30e6],
    "end": [120e6, 248e6, 159e6, 22.0e6, 138e6, 133e6, 31e6],
    "state": ["gain", "gain", "gain", "homdel", "loss", "loss", "homdel"],
}

kary = pqp.Karyogram(hg38, groups=2)
kary.paint(
    segments,
    start="start",
    end="end",
    color="state",
    palette={"gain": "tab:red", "loss": "tab:blue", "homdel": "tab:purple"},
)
kary.legend(title="Copy number")
fig = kary.figure
