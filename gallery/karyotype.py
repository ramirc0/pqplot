"""GWAS hits as marks and gene labels beside a two-group karyotype."""

import pqplot as pqp

hg38 = pqp.Genome.bundled("hg38")
hits = {
    "chrom": ["chr1", "chr1", "chr2", "chr5", "chr9", "chr9", "chr12", "chr19", "chr19", "chrX"],
    "pos": [55.0e6, 109.3e6, 21.0e6, 75.0e6, 22.0e6, 22.1e6, 121.0e6, 11.1e6, 44.9e6, 10.0e6],
    "trait": ["LDL", "LDL", "LDL", "LDL", "CAD", "T2D", "T2D", "LDL", "LDL", "BMI"],
    "gene": ["PCSK9", "CELSR2", "APOB", "HMGCR", "CDKN2B", "CDKN2A", "HNF1A", "LDLR", "APOE", "?"],
}
genes = {
    k: [v for v, gene in zip(col, hits["gene"], strict=True) if gene != "?"]
    for k, col in hits.items()
}

kary = pqp.Karyogram(
    hg38, groups=2, align="centromere", tracks=[pqp.Track("hits", side="after", size=5)]
)
kary.track("hits").mark(hits, color="trait")
kary.track("hits").label(genes, text="gene")
kary.legend(title="Trait")
fig = kary.figure
