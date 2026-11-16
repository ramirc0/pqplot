---
name: pqplot
description: Draw chromosome ideograms, karyograms and genome-wide plots in matplotlib with pqplot (`import pqplot as pqp`). Use whenever the user wants a banded chromosome over a locus or gene plot, a whole-karyotype figure (vertical or horizontal, wrapped into rows, homolog pairs), copy-number segments, local ancestry, train/test splits or binned values painted on chromosomes, GWAS hits, gene labels, peaks, coverage or density in tracks beside each chromosome, or a Manhattan-style x axis named by chromosome. Covers hg19, hg38, hs1 (T2T-CHM13), mm10, mm39 and any genome from a .chrom.sizes, .fai or cytoBand file. Also use when code imports pqplot, or when the user would reach for karyoploteR, pyideogram, tagore, PhenoGram or hand-drawn chromosome rectangles, even if they never name pqplot. Router skill. Read the matching file under references/ before writing pqplot code.
---

# pqplot

pqplot draws banded chromosomes in matplotlib and puts data on them. Every
chromosome and track is a plain matplotlib `Axes` with positions in bp, so any
matplotlib call works on it. Every entry point takes `ax=None`. Without an Axes
it makes a figure sized to the content. With one, it fills that slot of your
figure.

```python
import pqplot as pqp

hg38 = pqp.Genome.bundled("hg38")
```

Install with `uv add pqplot`. Python 3.12 or later, matplotlib 3.10 or later.
These references match **pqplot 0.1.0**. Check `importlib.metadata.version("pqplot")`
and re-verify behavior if it differs.

## Task to reference

| If the task is... | Read |
|---|---|
| loading an assembly, a `.chrom.sizes`/`.fai`/cytoBand file, picking chromosomes, parsing a locus, shaping input tables | `references/genomes-and-tables.md` |
| one chromosome over a locus plot, zooming a region into the Axes below, highlighting bands on one chromosome | `references/ideogram.md` |
| Manhattan plots or any plot along the concatenated genome, a strip of all chromosomes above it | `references/genome-wide.md` |
| whole karyotypes: layout, groups, alignment, homologs, painting bodies by category or value, legend, colorbar, drawing into panels | `references/karyogram.md` |
| data beside each chromosome: marks, gene labels, scatter, lines, areas, heatmaps, intervals | `references/tracks.md` |

A karyogram with tracks needs both `karyogram.md` and `tracks.md`.

## Rules that apply everywhere

1. **Columns are names or arrays.** Data arguments take any table indexed by
   column name (polars, pandas, pyarrow, dict of lists) plus column names. Each
   column argument also accepts the values themselves. Defaults are
   `chrom="chrom"`, `pos="pos"`, `start="start"`, `end="end"`,
   `value="value"` and `text="text"`. Rename columns or pass the names.
2. **Extra keyword arguments are not columns.** `c=`, `s=` and other
   matplotlib arguments go straight to matplotlib. `c="log2"` raises. Pass the
   array (`c=df["log2"]`). Arrays with one value per row are split by
   chromosome for you.
3. **Unmatched rows vanish silently.** Names match across `chr1`/`1` and
   `chrM`/`MT` only. `Chr1`, `CHR1`, RefSeq accessions (`NC_000001.11`),
   unplaced contigs, null chromosomes and chromosomes not drawn are skipped
   without a warning. Normalize names first. Count the rows whose chromosome
   is not in the genome, and report them.
4. **Positions are bp, as given.** Nothing shifts between 0-based and 1-based.
   `Region.parse("chr1:100-200")` keeps 100 and 200.
5. **Panels are child Axes.** Karyogram panels are not in `fig.axes`, and
   `plt.gca()` returns the hidden host. Get them with
   `kary.panel(chrom, track=None)` or `kary.panels(track=None)`.
6. **The value axis turns with the karyogram.** In a vertical karyogram, bp
   runs down y (inverted) and track values run along x, so use `set_xlim`. In
   a horizontal one, use `set_ylim`. Panels of one track share their value
   axis, so one call sets them all.
7. **Set the theme first.** Band, outline and default category colors come
   from `rcParams` when the artists are made. Apply a style (for example
   `poikilos`) before building the figure.
8. **Own figure or yours.** With `ax=None`, the figure uses the constrained
   layout and widens for a legend or colorbar, so the chromosomes keep their
   size. With `ax=`, you own the figure size and nothing widens.
9. **Legend and colorbar share a corner.** Both sit at the top right of the
   karyogram. Draw one, or move the legend with `loc=` and `bbox_to_anchor=`.
10. **Labels do not count in the layout.** Size a label track to the longest
    label (`size=5` fits about six characters at the default font). Labels
    that still do not fit are hidden from the last row up, with a
    `UserWarning`. Put the important rows first. Under `pytest -W error` that
    warning fails the test.
11. **Bundled genomes are primary only.** They hold 1 to 22 (19 in mouse), X
    and Y. They have no chrM and no unplaced or alt contigs.

## Workflow

1. Pick the entry point from the table and read its reference.
2. Load the genome. Filter and rename the data so every `chrom` value is in
   it (rule 3).
3. Build the figure. Save with `fig.savefig(...)` or `kary.figure.savefig(...)`.
4. Render a PNG and look at it. Check for clipped labels, the legend
   overlapping a colorbar and empty chromosomes that should have data.
