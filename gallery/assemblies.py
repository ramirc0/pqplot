"""Two assemblies in one figure: rhesus from sizes alone, and banded mouse."""

import matplotlib.pyplot as plt

import pqplot as pqp

sizes = {
    "chr1": 223616942, "chr2": 196197964, "chr3": 185288947, "chr4": 169963040,
    "chr5": 187317192, "chr6": 179085566, "chr7": 169868564, "chr8": 145679320,
    "chr9": 134124166, "chr10": 99517758, "chr11": 133066086, "chr12": 130043856,
    "chr13": 108737130, "chr14": 128056306, "chr15": 113283604, "chr16": 79627064,
    "chr17": 95433459, "chr18": 74474043, "chr19": 58315233, "chr20": 77137495,
    "chrX": 153388924, "chrY": 11753682,
}  # fmt: skip
rhemac10 = pqp.Genome(pqp.Chromosome(name, length) for name, length in sizes.items())
split = {
    "chrom": list(sizes),
    "split": ["test" if i % 7 == 2 else "valid" if i % 7 == 5 else "train" for i in range(22)],
}

fig, (left, right) = plt.subplots(1, 2, figsize=(6.5, 3.5))
kary = pqp.Karyogram(rhemac10, ax=left, groups=2)
kary.paint(split, color="split")
kary.legend(title="Split")
pqp.Karyogram(pqp.Genome.bundled("mm39"), ax=right, groups=2)
