# Tracks: data beside each chromosome

Declare tracks when you build the karyogram, then draw into them through
`kary.track(name)`.

```python
import numpy as np
import polars as pl

import pqplot as pqp

hg38 = pqp.Genome.bundled("hg38")
kary = pqp.Karyogram(
    hg38,
    chroms=["chr1", "chr2", "chr7"],
    orientation="horizontal",
    tracks=[pqp.Track("genes", side="before"), pqp.Track("cnv", side="after", size=3)],
)
```

`Track(name, side="before", size=2)`:

- `side="before"` is left of a vertical karyogram and above a horizontal one.
  `"after"` is right or below.
- `size` is the width across the chromosome, in body widths. Use 1 for a thin
  heatmap, 2 to 3 for scatter or areas, 5 or more for labels.
- Names must be unique. Across each chromosome, the order is the `before`
  tracks as listed, the body, then the `after` tracks as listed.

## Methods

Each method takes a table with one row per point or bin, splits it by
chromosome and draws into each chromosome's panel. Rows on chromosomes that
are not drawn are skipped. Most return a dict of artists keyed by chromosome
name.

| Method | Columns (defaults) | Draws | Returns |
|---|---|---|---|
| `mark` | `chrom`, `pos`, optional `color` category | triangles on the edge facing the body, pointing at it | dict of `PathCollection` |
| `label` | `chrom`, `pos`, `text` | text with elbow leaders, spread apart when crowded | dict of `LabelLane` |
| `scatter` | `chrom`, `pos`, `value` | one marker per row | dict of `PathCollection` |
| `line` | `chrom`, `pos`, `value` | a line per chromosome, sorted by position | dict of `Line2D` |
| `area` | `chrom`, `start`, `end`, `value` | filled steps from `baseline=0` | dict of `StepPatch` |
| `heatmap` | `chrom`, `start`, `end`, `value` | bins colored across the full track width | `ScalarMappable` |
| `intervals` | `chrom`, `start`, `end`, optional `color` category | boxes across the track width | dict of `Spans` |

Extra keyword arguments go to the matplotlib call (`scatter`, `plot`,
`stairs`, `PolyCollection` or `Text`). Arrays among them with one value per
row are split by chromosome. Column names are not, so pass `c=df["log2"]`,
not `c="log2"`.

## Value tracks

```python
rng = np.random.default_rng(1)
starts = np.concatenate([np.arange(0, hg38[c].length, 1_000_000) for c in ["chr1", "chr2", "chr7"]])
chrom = np.concatenate(
    [np.full(len(range(0, hg38[c].length, 1_000_000)), c) for c in ["chr1", "chr2", "chr7"]]
)
bins = pl.DataFrame({"chrom": chrom, "start": starts, "end": starts + 1_000_000}).with_columns(
    mid=(pl.col("start") + pl.col("end")) / 2,
    log2=pl.Series(rng.normal(0, 0.3, len(starts))),
    per_mb=pl.Series(rng.gamma(2, 6, len(starts))),
)

kary.track("genes").area(bins, value="per_mb", linewidth=0)
points = kary.track("cnv").scatter(
    bins, pos="mid", value="log2", s=3, c=bins["log2"], cmap="coolwarm"
)
kary.panel("chr1", track="cnv").set_ylim(-2, 2)  # horizontal: values on y; shared by the track
```

- Panels of one track share the value axis. Set limits on any one of them.
  A **vertical** karyogram puts values on **x** (`set_xlim`). A horizontal one
  puts them on y (`set_ylim`).
- The value axis shows on one panel only, ticked at its outermost round
  values. A vertical karyogram shows it at the top of its first chromosome. A
  horizontal one shows it right of the first chromosome of its last group.
- `scatter` with a numeric `c` shares one color scale across chromosomes.
  For a colorbar, pass any of the returned collections:
  `kary.colorbar(next(iter(points.values())), label="log2")`.
- Markers have no edge by default. Pass `linewidths=` for one.
- `area` bins of one chromosome must not overlap. Gaps stay empty.
  `baseline=` moves the fill origin, for example to 0 for signed values.
- `heatmap` returns the `ScalarMappable` for `kary.colorbar`. It takes `cmap`
  and `norm` like `paint`.

To title the value axis, label the panel that shows it. In a vertical
karyogram that is the first chromosome's track panel. Move the title to the
top, or it collides with the chromosome name below:

```python
vertical = pqp.Karyogram(hg38, groups=2, tracks=[pqp.Track("genes", side="before")])
vertical.track("genes").area(bins, value="per_mb", linewidth=0)
axis_panel = vertical.panel("chr1", track="genes")
axis_panel.set_xlabel("genes/Mb", fontsize="x-small")
axis_panel.xaxis.set_label_position("top")
```

## Marks and labels

```python
hits = pl.DataFrame(
    {
        "chrom": ["chr1", "chr1", "chr2", "chr9", "chr19"],
        "pos": [55_000_000, 109_300_000, 21_000_000, 22_000_000, 44_900_000],
        "trait": ["LDL", "LDL", "LDL", "CAD", "LDL"],
        "gene": ["PCSK9", "CELSR2", "APOB", "CDKN2B", "APOE"],
    }
)
kary = pqp.Karyogram(
    hg38, groups=2, align="centromere", tracks=[pqp.Track("hits", side="after", size=5)]
)
kary.track("hits").mark(hits, color="trait", palette={"LDL": "C0", "CAD": "C3"})
kary.track("hits").label(hits, text="gene")
kary.legend(title="Trait")
```

- Marks and labels can share one track. Both start from the edge that faces
  the body and hide the track's value axis.
- `mark(size=16)` is the marker area in points squared, as in `scatter`.
- Labels spread apart at draw time and keep their order along the
  chromosome. When more labels than fit, the last rows are hidden with a
  `UserWarning`. Sort the table by importance first, and lower `fontsize`
  (default `"x-small"`) or label fewer rows.
- Call `label` once per track with every row. Separate calls, such as one
  per category, place their labels independently, so they can overlap. Show
  the category through `mark(color=...)` instead.
- Labels do not count in the layout. If they run into the next chromosome or
  off the figure, raise the track `size`.

## Intervals

```python
peaks = pl.DataFrame(
    {
        "chrom": ["chr1", "chr1", "chr7"],
        "start": [1_000_000, 30_000_000, 55_000_000],
        "end": [1_000_400, 30_002_000, 55_300_000],
        "kind": ["peak", "peak", "amplicon"],
    }
)
kary = pqp.Karyogram(hg38, chroms=["1", "7"], tracks=[pqp.Track("peaks", size=1)])
kary.track("peaks").intervals(peaks, color="kind")
kary.legend()
```

- Intervals narrower than 1.5 points widen so small peaks stay visible.
- Categories from `intervals`, `mark` and `paint` share one legend.
