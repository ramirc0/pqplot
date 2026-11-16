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
  when `chrom` and `pos` differ in length. `concatenated` rejects negative
  or infinite gaps.
- `Karyogram` raises `ValueError` when there is nothing to draw, a group is
  empty, a chromosome is drawn twice, `chroms` or a group is one string, or
  `copies` is not a whole number. `Track` rejects empty names and sizes that
  are not positive.
- `Karyogram.legend` raises `ValueError` when no layer drew categories.
  `Karyogram.panels` raises `KeyError` for an unknown track, as `panel`
  does.

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
