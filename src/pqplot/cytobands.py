"""Readers for cytoband tables, chromosome sizes and centromere BED files.

The readers return every contig in a file. `Genome` picks the chromosomes.
"""

import gzip
from collections import defaultdict
from collections.abc import Callable, Iterator
from os import PathLike

type BandRow = tuple[int, int, str, str]


def read_bands(path: str | PathLike) -> dict[str, list[BandRow]]:
    """Read a cytoband table such as UCSC ``cytoBand.txt.gz`` or the T2T BED.

    Parameters
    ----------
    path : str or PathLike
        Tab-separated file, gzipped or plain, with columns chrom, start, end,
        band name and stain.

    Returns
    -------
    dict of str to list of tuple
        ``(start, end, name, stain)`` rows for each chromosome, in file order.

    Examples
    --------
    >>> from importlib.resources import files
    >>> bands = read_bands(files("pqplot") / "data/hg38/cytoBand.txt.gz")
    >>> bands["chr1"][0]
    (0, 2300000, 'p36.33', 'gneg')
    """
    rows = defaultdict(list)
    # Tabs only: unplaced contigs in UCSC tables have an empty band name.
    for chrom, row in _records(path, 3, _band, separator="\t"):
        rows[chrom].append(row)
    return dict(rows)


def read_sizes(path: str | PathLike) -> dict[str, int]:
    """Read chromosome lengths from a ``.chrom.sizes`` or ``.fai`` file.

    Parameters
    ----------
    path : str or PathLike
        File, gzipped or plain, whose first two columns are the chromosome
        name and its length, separated by tabs or spaces.

    Returns
    -------
    dict of str to int
        Length of each chromosome, in file order.

    Raises
    ------
    ValueError
        If a line is malformed or a chromosome appears more than once.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> path = Path(tempfile.mkdtemp()) / "x.chrom.sizes"
    >>> _ = path.write_text("chr1\\t1000\\nchr2\\t500\\n")
    >>> read_sizes(path)
    {'chr1': 1000, 'chr2': 500}
    """
    sizes = {}
    for chrom, length in _records(path, 2, lambda f: (f[0], int(f[1]))):
        if chrom in sizes:
            raise ValueError(f"{path}: chromosome {chrom!r} appears more than once")
        sizes[chrom] = length
    return sizes


def read_centromeres(path: str | PathLike) -> dict[str, tuple[int, int]]:
    """Read centromere spans from a BED file.

    Parameters
    ----------
    path : str or PathLike
        BED file, gzipped or plain. Several rows for one chromosome merge into
        the span from their lowest start to their highest end. Track and
        browser lines are skipped.

    Returns
    -------
    dict of str to tuple of int
        ``(start, end)`` of each chromosome's centromere.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> path = Path(tempfile.mkdtemp()) / "centromeres.bed"
    >>> _ = path.write_text("chr1\\t120\\t130\\nchr1\\t100\\t110\\n")
    >>> read_centromeres(path)
    {'chr1': (100, 130)}
    """
    spans = {}
    for chrom, start, end in _records(path, 3, lambda f: (f[0], int(f[1]), int(f[2]))):
        lo, hi = spans.get(chrom, (start, end))
        spans[chrom] = (min(lo, start), max(hi, end))
    return spans


def _band(fields: list[str]) -> tuple[str, BandRow]:
    name, stain = (fields[3:] + ["", ""])[:2]
    return fields[0], (int(fields[1]), int(fields[2]), name, stain)


def _records[T](
    path: str | PathLike,
    columns: int,
    parse: Callable[[list[str]], T],
    separator: str | None = None,
) -> Iterator[T]:
    """Parse each data line, raising errors that name the file and line."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = [field.strip() for field in line.rstrip("\r\n").split(separator)]
            try:
                if len(fields) < columns:
                    raise ValueError(f"expected {columns} columns")
                yield parse(fields)
            except ValueError as error:
                raise ValueError(f"{path} line {number}: {error}: {line.strip()!r}") from None
