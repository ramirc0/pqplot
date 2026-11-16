# One chromosome: `ideogram` and `zoom`

## Ideogram over a locus plot

The standard figure: a banded chromosome on top, the data below, the locus
boxed on the chromosome and joined to the plot by two lines.

```python
import matplotlib.pyplot as plt
import numpy as np

import pqplot as pqp

hg38 = pqp.Genome.bundled("hg38")
locus = pqp.Region.parse("chr17:43,044,295-43,125,483")
pos = np.linspace(locus.start, locus.end, 400)
signal = np.exp(-(((pos - 43.07e6) / 6e3) ** 2))

fig, (head, body) = plt.subplots(2, 1, figsize=(6.5, 2.5), height_ratios=[1, 3])
ideo = pqp.ideogram(hg38[locus.chrom], ax=head)
body.fill_between(pos, signal, linewidth=0)
body.set(xlim=(locus.start, locus.end), ylabel="Signal")
body.xaxis.set_major_formatter(lambda x, _: f"{x / 1e6:.2f} Mb")
box, lines = ideo.zoom(locus, into=body)
```

- `ideogram(chromosome, ax=None, *, bands=True, labels=True)` returns an
  `Ideogram`. It sets the Axes to `xlim=(0, length)`, `ylim=(-1, 1)` and turns
  the axis off. The name goes left of the body. Band names go below, and those
  that would overlap are hidden at draw time.
- Without `ax`, it makes a 6.5 x 0.75 inch figure with the constrained layout.
- Give the ideogram row about a quarter of the height. Its body spans half
  the Axes height.
- `zoom(region, into, *, color="C3")` boxes the region and draws two lines to
  the near corners of `into`. Both are placed at draw time, so call order and
  later layout changes do not matter. A tiny locus keeps a box at least 3
  points wide. The region must be on this chromosome, or it raises
  `ValueError`.
- Set the x limits of `into` to the locus yourself. `zoom` does not.
- With `layout="constrained"`, the zoom lines can cross the gap between rows.
  That is intended.

## Highlighting and annotating one chromosome

The x axis is in bp, so plain matplotlib calls place things on it.
`Ideogram.paint(starts, ends, colors)` fills spans of the body over the bands,
clipped to the outline. It takes arrays, not a table. Spans narrower than 1.5
points widen so they stay visible.

```python
ideo = pqp.ideogram(hg38["chr9"])
ideo.paint([21_900_000, 60_000_000], [22_100_000, 90_000_000], ["tab:red", "tab:blue"])
ideo.axes.annotate(
    "CDKN2A",
    ideo.xy(22_000_000, 1),  # (bp along, body half-widths across): +1 is the top edge
    xytext=(0, 6),
    textcoords="offset points",
    ha="center",
    fontsize="x-small",
)
```

`Ideogram.xy(along, across)` maps bp and body half-widths (-1 to 1 are the
body edges) to data coordinates. Use it to place markers or text against the
body.

## Ideograms where you want them

`pqp.Ideogram` is the artist that the entry points use. Build it directly for
custom layouts, such as several chromosomes in one Axes or a vertical one:

```python
fig, ax = plt.subplots(figsize=(2, 4))
for i, name in enumerate(["chr1", "chr2", "chr3"]):
    pqp.Ideogram(hg38[name], ax, orientation="vertical", center=i * 1.5, half=0.4)
ax.set(xlim=(-1, 4), ylim=(hg38["chr1"].length, 0))
ax.set_axis_off()
```

- `orientation`, `offset` (axis position of bp 0), `center` (midline across)
  and `half` (body half-width in axis units) place it. `bands=False` draws a
  plain body. Extra keyword arguments style the outline (`PathPatch`).
- It does not set limits, so set them yourself. Caps and the centromere waist
  are sized at draw time and stay round after resizes.
- For a whole karyotype, use `Karyogram` instead (`karyogram.md`).
