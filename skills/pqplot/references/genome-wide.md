# Genome-wide plots: `concatenated`, `chromosome_axis`, `genome_strip`

Lay the chromosomes end to end on one x axis, plot as usual, then name the
axis by chromosome.

## Manhattan plot

```python
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import pqplot as pqp

hg38 = pqp.Genome.bundled("hg38")
linear = hg38.concatenated(gap=5_000_000)

rng = np.random.default_rng(2)
chrom = rng.choice([c.name for c in hg38], 5_000)
gwas = pl.DataFrame(
    {
        "chrom": chrom,
        "pos": rng.uniform(0, [hg38[c].length for c in chrom]),
        "p": 10 ** -rng.exponential(1, len(chrom)),
    }
)

fig, ax = plt.subplots(figsize=(6.5, 2.5))
ax.scatter(linear.position(gwas["chrom"], gwas["pos"]), -np.log10(gwas["p"]), s=2, linewidths=0)
ax.axhline(-np.log10(5e-8), color="C3", linewidth=0.5)
ax.set_ylabel("$-\\log_{10} p$")
pqp.chromosome_axis(linear, ax)
```

- `genome.concatenated(gap=0)` returns a `LinearGenome`. Its chromosomes keep
  the genome's order. `gap` is bp between neighbors. A gap of about 0.2% of
  the genome length separates them visibly.
- `linear.position(chrom, pos)` maps arrays (or Series) to axis positions. It
  returns NaN for chromosomes not in the genome, and matplotlib then skips
  those points silently. Count the NaNs if the data might use other names.
- `linear.offset(chrom)` is where a chromosome starts, and `linear.length` is
  the full axis. Use them to place lines or text at a chromosome.
- `chromosome_axis(linear, ax, *, shade=True)` sets ticks at the chromosome
  midpoints named without `chr`, hides names that would collide, sets the x
  limits to the genome and shades every other chromosome. `shade="0.9"` (any
  color) or `shade=False` changes the shading. Call it after plotting.
- Color points by chromosome parity with `pl.col("chrom").replace_strict(...)`
  or a dict lookup. pqplot does not color points.

## Only some chromosomes

Build the genome you want to show, then concatenate it:

```python
autosomes = pqp.Genome([hg38[c] for c in hg38.autosomes]).concatenated(gap=5_000_000)
fig, ax = plt.subplots(figsize=(6.5, 2))
pqp.chromosome_axis(autosomes, ax)
```

## A strip of ideograms above the plot

`genome_strip(linear, ax=None, *, bands=True, labels=True)` draws every
chromosome of `linear` in one row, at the same offsets. Share x with the plot
below and use the same `LinearGenome` for both.

```python
fig, (strip, ax) = plt.subplots(
    2, 1, sharex=True, height_ratios=[1, 6], figsize=(6.5, 3), layout="constrained"
)
pqp.genome_strip(linear, ax=strip, labels=False)  # the axis below names them
ax.scatter(linear.position(gwas["chrom"], gwas["pos"]), -np.log10(gwas["p"]), s=2, linewidths=0)
pqp.chromosome_axis(linear, ax)
```

- It returns one `Ideogram` per chromosome, in axis order. Each has
  `.offset`, so `ideograms[i].paint(...)` takes bp on that chromosome.
- Without `ax`, it makes a 10 x 0.6 inch figure.
- Short chromosomes are only a few points wide at genome scale. Their bands
  blur, so `bands=False` can read better.
