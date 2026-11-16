# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The public API is
every name in `pqplot.__all__`, and every method and attribute those names
document.

## [Unreleased]

### Fixed

- `TrackPanels.heatmap` and `TrackPanels.intervals` hide the track's value
  axis, as `mark` and `label` do. It showed a meaningless 0 to 1 scale.

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
