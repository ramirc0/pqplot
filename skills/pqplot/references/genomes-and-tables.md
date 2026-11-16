# Genomes, regions and input tables

## Bundled assemblies

```python
import pqplot as pqp

hg38 = pqp.Genome.bundled("hg38")  # also "hg19", "hs1" (T2T-CHM13), "mm10", "mm39"
len(hg38)  # 24: chr1..chr22, chrX, chrY
hg38["1"].length, hg38["chr1"].centromere  # names match across chr1 and 1
```

- Bands and centromeres come from UCSC cytoBand tables (T2T for hs1).
- Mouse chromosomes are telocentric. They get a centromere knob at position
  0, and `chromosome.telocentric` is True.
- Bundled genomes hold no chrM, unplaced or alt contigs.

## Other genomes

From a sizes or FASTA index file. The genome is unbanded. By default it keeps
the primary chromosomes (numbered, X, Y, Z, W) in natural order and drops
contigs. A file with none of them, such as `Chr1` or `2L` names, raises
`ValueError`. Pass `chroms=` then. Columns may be separated by tabs or
spaces. A malformed line raises an error naming the file and line.

```python
import tempfile
from pathlib import Path

folder = Path(tempfile.mkdtemp())
sizes = folder / "rheMac10.chrom.sizes"
sizes.write_text("chr2\t196197964\nchr1\t223616942\nchrUn_NW_1\t5000\n")
centromeres = folder / "centromeres.bed"
centromeres.write_text("chr1\t120000000\t122000000\n")

rhemac = pqp.Genome.from_sizes(sizes, centromeres=centromeres)
[c.name for c in rhemac]  # ['chr1', 'chr2'], contig dropped
rhemac["2"].centromere  # None: unknown, drawn without a waist
```

Pass `chroms=[...]` to keep a chosen set in a chosen order. It raises
`KeyError` if one is missing from the file.

From a UCSC-style cytoband table (`chrom start end name gieStain`, gzipped or
plain): `pqp.Genome.from_cytobands(path, chroms=None)`. The centromere spans
the `acen` bands.

From scratch, such as sizes held in a dict:

```python
sizes = {"chr1": 223_616_942, "chr2": 196_197_964, "chrX": 153_388_924}
custom = pqp.Genome(pqp.Chromosome(name, length) for name, length in sizes.items())
with_centromere = pqp.Chromosome("chr1", 223_616_942, centromere=(120_000_000, 122_000_000))
```

`centromere=(0, 0)` marks a telocentric chromosome. `None` means unknown.

## Picking chromosomes

A `Genome` iterates in display order. It cannot be sliced. Build a smaller one
or pass `chroms=` to `Karyogram`.

```python
autosomes = pqp.Genome([hg38[c] for c in hg38.autosomes])
"chrX" in hg38, "chrM" in hg38  # (True, False)
[(c.name, c.label, c.length) for c in hg38][:2]  # label drops the "chr" prefix
```

## Regions

```python
locus = pqp.Region.parse("chr17:43,044,295-43,125,483")  # commas optional
locus.chrom, locus.start, locus.end
pqp.Region("chr9", 21_900_000, 22_000_000)
```

`Region.parse` keeps the numbers as written. It raises `ValueError` for bad
text or start after end.

## Input tables

Layers take `data` plus column arguments. A column argument is a column name
in `data` or the values themselves, so these are the same:

```python
import numpy as np
import polars as pl

hits = pl.DataFrame({"chromosome": ["1", "chr9"], "bp": [55_000_000, 22_000_000]})
kary = pqp.Karyogram(hg38, chroms=["1", "9"], tracks=[pqp.Track("hits", side="after")])
kary.track("hits").mark(hits, chrom="chromosome", pos="bp")
kary.track("hits").mark(chrom=np.array(["1", "chr9"]), pos=np.array([55e6, 22e6]))
```

- polars, pandas, pyarrow tables and dicts of lists or arrays all work.
- Default names: `chrom`, `pos`, `start`, `end`, `value`, `text`.
- Rows of one chromosome can come in any order.
- Columns must have one length, or the call raises `ValueError`.
- Rows with a null chromosome, position, span, category or text are skipped.
  A null value leaves a gap. Only drawn rows enter legends and color scales.
- A float chromosome column (an int column with nulls) matches: `1.0` is
  chromosome 1.

Check what will be dropped before you plot:

```python
calls = pl.DataFrame({"chrom": ["chr1", "Chr2", "chrUn_KI270302v1", "MT"]})
dropped = [c for c in calls["chrom"].unique(maintain_order=True) if c not in hg38]
assert dropped == ["Chr2", "chrUn_KI270302v1", "MT"]
calls = calls.with_columns(pl.col("chrom").str.replace(r"^(?i)chr", "chr"))
```
