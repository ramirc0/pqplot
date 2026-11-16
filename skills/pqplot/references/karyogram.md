# Karyograms: layout, painting, legend and colorbar

`Karyogram` draws chromosomes side by side, each in its own panel. All panels
share one bp scale, so lengths compare across the figure.

```python
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import pqplot as pqp

hg38 = pqp.Genome.bundled("hg38")
kary = pqp.Karyogram(hg38, groups=2, align="centromere")
kary.figure.savefig("karyotype.png", dpi=300)
```

## Layout options

`Karyogram(genome, ax=None, *, orientation="vertical", groups=1, align="start", chroms=None, copies=1, bands=True, tracks=())`

| Option | Effect |
|---|---|
| `orientation` | `"vertical"`: chromosomes stand side by side, start at the top. `"horizontal"`: they lie in rows, start at the left, names on the left. |
| `groups=2` | Wrap into 2 even groups, in order. Groups stack down a vertical karyogram and sit side by side in a horizontal one. |
| `groups=[["1", "2"], ["X", "Y"]]` | Explicit groups. Not allowed with `chroms`. |
| `align` | `"start"`, `"centromere"` or `"end"` lines up the chromosomes in each group. Telocentric and unknown centromeres count as position 0. |
| `chroms=["1", "2", "X"]` | Draw only these, in this order. |
| `copies=2` | Draw homologs side by side in each body panel. |
| `bands=False` | Plain bodies. Use it when painting values, so bands do not compete. |
| `tracks=[pqp.Track(...)]` | Data strips beside each chromosome. See `tracks.md`. Declare them here. They cannot be added later. |

```python
horizontal = pqp.Karyogram(hg38, orientation="horizontal", chroms=["chr1", "chr2", "chr7"])
pairs = pqp.Karyogram(hg38, chroms=hg38.autosomes[:6], copies=2, bands=False)
explicit = pqp.Karyogram(hg38, groups=[["1", "2", "3"], ["X", "Y"]])
```

Without `ax`, the figure is sized to the content with the constrained layout.
The longest group is 2.5 inches (vertical) or 6 inches (horizontal) of bp.
With `ax=`, the karyogram fills that Axes, and your figure size sets the scale:

```python
fig, (left, right) = plt.subplots(1, 2, figsize=(6.5, 3.5))
pqp.Karyogram(hg38, ax=left, groups=2)
pqp.Karyogram(pqp.Genome.bundled("mm39"), ax=right, groups=2)
```

## Painting bodies

`kary.paint(data=None, *, chrom="chrom", start=None, end=None, color=None, value=None, palette=None, cmap=None, norm=None, copy=None)`
fills spans of the bodies over the bands. Pass `color` (categories) or `value`
(numbers), not both.

Categories, such as copy-number states:

```python
segments = pl.DataFrame(
    {
        "chrom": ["chr1", "chr7", "chr9", "chr9"],
        "start": [150_000_000, 0, 21_900_000, 60_000_000],
        "end": [248_000_000, 159_000_000, 22_000_000, 138_000_000],
        "state": ["gain", "gain", "homdel", "loss"],
    }
)
kary = pqp.Karyogram(hg38, groups=2)
kary.paint(
    segments,
    start="start",
    end="end",
    color="state",
    palette={"gain": "tab:red", "loss": "tab:blue", "homdel": "tab:purple"},
)
kary.legend(title="Copy number")
```

- `palette` sets the colors and the legend order. It must cover every
  category, or `KeyError`. Without it, the theme's color cycle goes in order
  of first appearance.
- Touching spans of one category merge. Spans narrower than 1.5 points widen
  so they stay visible, and narrow spans draw over wide ones.
- Without `start` and `end`, each row fills its whole chromosome, which suits
  per-chromosome labels such as train/test splits:

```python
split = {"chrom": ["chr1", "chr2", "chr3"], "split": ["train", "test", "valid"]}
kary = pqp.Karyogram(hg38, chroms=["1", "2", "3"])
kary.paint(split, color="split")
kary.legend(title="Split")
```

Values, such as density per bin, return a `ScalarMappable` for the colorbar:

```python
starts = np.arange(0, hg38["chr1"].length, 1_000_000)
bins = pl.DataFrame(
    {
        "chrom": "chr1",
        "start": starts,
        "end": starts + 1_000_000,
        "per_mb": np.random.default_rng(0).gamma(2, 100, len(starts)),
    }
)
kary = pqp.Karyogram(hg38, chroms=["1"], bands=False)
heat = kary.paint(bins, start="start", end="end", value="per_mb", cmap="magma")
kary.colorbar(heat, label="cCREs per Mb")
```

- `norm=matplotlib.colors.LogNorm()` or `Normalize(vmin, vmax)` fixes the
  scale, for example to share it between figures.

Homologs: with `copies=2`, `copy=` names the column that says which copy a row
paints. Its sorted levels map to copies left to right. Without `copy=`, rows
paint every copy.

```python
tracts = {
    "chrom": ["chr1", "chr1", "chr1", "chr1"],
    "start": [0, 1e8, 0, 1.6e8],
    "end": [1e8, 2.49e8, 1.6e8, 2.49e8],
    "ancestry": ["EUR", "AFR", "AFR", "EUR"],
    "hap": [1, 1, 2, 2],
}
kary = pqp.Karyogram(hg38, chroms=["1"], copies=2, bands=False)
kary.paint(tracts, start="start", end="end", color="ancestry", copy="hap")
kary.legend(title="Ancestry")
```

## Legend and colorbar

- `kary.legend(title=None, **kwargs)` lists every category painted, marked or
  drawn as intervals so far, in one legend. A category keeps one color across
  calls. Call it after all layers. Calling it again replaces it. With no
  categories drawn yet it raises `ValueError`. Defaults: outside the top
  right, no frame.
- `kary.colorbar(mappable, label=None, **kwargs)` puts a 7 x 144 point bar
  right of the karyogram, top-aligned. It takes the return value of `paint`
  with `value=` or of a track's `heatmap`. Calling it again replaces the bar.
- Both sit at the top right. To show both, move the legend, for example
  `kary.legend(loc="lower left", bbox_to_anchor=(1, 0))`.
- In a karyogram's own figure, both grow the figure so chromosomes keep
  their size. With `ax=`, leave room yourself.

## Drawing into panels

Panels are plain Axes. They are children of a hidden host Axes (`kary.ax`),
so `fig.axes` and `plt.gca()` do not reach them.

```python
kary = pqp.Karyogram(hg38, groups=2)
body = kary.panel("chr9")  # body panel; kary.panel("9", track="cnv") for a track
body.axhline(22_000_000, color="C3", linewidth=0.75)  # vertical: bp runs down y
for panel in kary.panels().values():
    panel.axhline(10_000_000, color="C0", linewidth=0.75, linestyle=":")
ideo = kary.ideograms("chr9")[0]  # one Ideogram per copy
ideo.paint([21_900_000], [22_100_000], "C3")
```

To name positions, use a label track (`tracks.md`). Free text in a body panel
runs into the neighboring chromosome.

- In a vertical body panel, x runs across the copies (0 to 1 for one copy)
  and y is bp, inverted. A horizontal panel swaps them.
- `Ideogram.xy(bp, across)` gives data coordinates against the body, with
  -1 and 1 at its edges. `Ideogram.paint(starts, ends, colors)` paints one
  copy directly.
- Panels are placed at draw time. Their positions are only final after a
  draw (`fig.draw_without_rendering()`).
