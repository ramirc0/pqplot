# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The public API is
every name in `pqplot.__all__`, and every method and attribute those names
document.

## [Unreleased]

### Changed

- `Genome`, `Chromosome`, `Band` and `Region` raise `ValueError` for input
  they used to accept silently: chromosomes that share a name (`chr1` and
  `1`), lengths of 0 or less, centromeres outside the chromosome, bands that
  end before they start, and regions with a negative or reversed span.
- `Genome.from_sizes` and `Genome.from_cytobands` raise `ValueError` when no
  chromosome is chosen by default, with a hint to pass `chroms=`.
- `LinearGenome.position` returns NaN for positions outside their
  chromosome, so they no longer land on a neighbor. It raises `ValueError`
  when `chrom` and `pos` differ in length. `concatenated` rejects empty
  genomes and negative or infinite gaps.
- `Karyogram` raises `ValueError` when there is nothing to draw, a group is
  empty, a chromosome is drawn twice, `chroms` or a group is one string, or
  `copies` is not a whole number. `Track` rejects empty names and sizes that
  are not positive.
- `Karyogram.legend` raises `ValueError` when no layer drew categories.
  `Karyogram.panels` raises `KeyError` for an unknown track, as `panel`
  does.
- Data layers check that their columns have one length and raise
  `ValueError` when they differ. They used to truncate or fail with an
  `IndexError`. A missing column raises `KeyError` naming it.
- `TrackPanels.area` raises `ValueError` for bins that overlap or end before
  they start, which it used to draw as wrong steps. `Ideogram.paint` and
  `Karyogram.paint` raise for spans that end before they start.

### Fixed

- `TrackPanels.heatmap` and `TrackPanels.intervals` hide the track's value
  axis, as `mark` and `label` do. It showed a meaningless 0 to 1 scale.
- `chroms=` accepts a one-shot iterator. A generator used to give an empty
  genome.
- Sizes and centromere files accept spaces between columns, padded names, a
  byte order mark, and BED track or browser lines. A malformed line raises
  an error that names the file and line.
- `Region.parse` keeps colons in the name, as in `HLA-A*01:01:1-100`.
- A chromosome column of floats, such as a pandas integer column with
  nulls, matches names: `1.0` finds chromosome 1.
- A second `Karyogram.colorbar` call replaces the bar. It used to stack a
  new bar and widen the figure again.
- A karyogram's own figure grows in height to hold its legend or colorbar,
  so a horizontal karyogram of a few chromosomes no longer clips them.
- A karyogram's own figure leaves room for chromosome names at larger font
  sizes.
- `chromosome_axis` called again on one Axes replaces its ticks and shading.
  A second genome used to crash at draw time. `shade=None` draws no shading.
- Band labels thin out correctly on an inverted x axis.
- `Ideogram.zoom` lines pass behind the band labels, and labels under the
  locus box hide. It rejects regions past the chromosome end and target
  Axes in another figure.
- `Ideogram.paint` takes scalars and rejects reversed spans or starts and
  ends of different lengths. `Ideogram` takes `clip_on=`.
- Rows with a missing start, end, position, category, copy or text are
  skipped in every layer. A null start or end used to erase neighboring
  spans in `paint`, a NaN label position moved every other label, and a
  null category crashed.
- Legends and color scales count only the rows that are drawn. Rows on
  chromosomes left out of the karyogram no longer add legend entries or
  stretch a colorbar.
- A category keeps one color across `paint`, `mark` and `intervals` calls.
  The same label used to get a new color in each call.
- `scatter` keeps one color scale across chromosomes when only `vmin` or
  `vmax` is given, and no longer warns for RGB colors. String positions
  become numbers in `scatter` and `line`, as in the other layers.
- Lists with one value per row, such as `alpha=[...]`, are split by
  chromosome like arrays, in every layer.

## [0.1.0] - 2026-11-15

### Added

- `Genome` with bundled hg19, hg38, hs1, mm10 and mm39 assemblies, and loading
  from `.chrom.sizes` or `.fai` files with an optional centromere BED.
- `Region` parses loci such as `chr17:43,044,295-43,125,483`. Chromosome names
  match across `chr1` and `1`.
- `ideogram` draws one banded chromosome. `Ideogram.zoom` links a region to an
  Axes below it.
- `Karyogram` lays out whole karyotypes, vertical or horizontal, in groups,
  aligned by centromere and with homolog pairs.
- `Karyogram.paint` colors chromosome bodies by category or by value.
- `Track` holds data beside each chromosome as marks, labels, scatter, lines,
  areas, heatmaps and intervals. Crowded labels spread apart.
- `genome_strip` and `chromosome_axis` set up genome-wide plots such as
  Manhattan plots.

[Unreleased]: https://github.com/ramirc0/pqplot/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ramirc0/pqplot/releases/tag/v0.1.0
