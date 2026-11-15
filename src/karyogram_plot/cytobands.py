"""Readers for cytoband tables, chromosome sizes and centromere BED files.

The readers return every contig in a file. `Genome` picks the chromosomes.
"""

import gzip
from collections import defaultdict
from collections.abc import Iterator
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
    >>> bands = read_bands(files("karyogram_plot") / "data/hg38/cytoBand.txt.gz")
    >>> bands["chr1"][0]
    (0, 2300000, 'p36.33', 'gneg')
    """
    rows = defaultdict(list)
    for chrom, start, end, *rest in _fields(path):
        name, stain = (rest + ["", ""])[:2]
        rows[chrom].append((int(start), int(end), name, stain))
    return dict(rows)


def read_sizes(path: str | PathLike) -> dict[str, int]:
    """Read chromosome lengths from a ``.chrom.sizes`` or ``.fai`` file.

    Parameters
    ----------
    path : str or PathLike
        Tab-separated file, gzipped or plain, whose first two columns are the
        chromosome name and its length.

    Returns
    -------
    dict of str to int
        Length of each chromosome, in file order.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> path = Path(tempfile.mkdtemp()) / "x.chrom.sizes"
    >>> _ = path.write_text("chr1\\t1000\\nchr2\\t500\\n")
    >>> read_sizes(path)
    {'chr1': 1000, 'chr2': 500}
    """
    return {chrom: int(length) for chrom, length, *_ in _fields(path)}


def read_centromeres(path: str | PathLike) -> dict[str, tuple[int, int]]:
    """Read centromere spans from a BED file.

    Parameters
    ----------
    path : str or PathLike
        BED file, gzipped or plain. Several rows for one chromosome merge into
        the span from their lowest start to their highest end.

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
    for chrom, start, end, *_ in _fields(path):
        lo, hi = spans.get(chrom, (int(start), int(end)))
        spans[chrom] = (min(lo, int(start)), max(hi, int(end)))
    return spans


def _fields(path: str | PathLike) -> Iterator[list[str]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        for line in handle:
            if line.strip() and not line.startswith("#"):
                yield line.rstrip("\n").split("\t")
