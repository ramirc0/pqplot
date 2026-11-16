"""Genome model: chromosomes, bands, stains, regions and concatenation."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from importlib.resources import as_file, files
from os import PathLike

import numpy as np

from karyogram_plot import cytobands

BUNDLED = {
    "hg19": "hg19/cytoBand.txt.gz",
    "hg38": "hg38/cytoBand.txt.gz",
    "hs1": "hs1/chm13v2.0_cytobands_allchrs.bed",
    "mm10": "mm10/cytoBand.txt.gz",
    "mm39": "mm39/cytoBand.txt.gz",
}
PRIMARY = re.compile(r"(?:chr)?(?:\d+|[XYZW])")
LETTER_ORDER = {"X": 0, "Y": 1, "Z": 2, "W": 3, "MT": 4}


def chrom_key(name: object) -> str:
    """Return the key that matches one chromosome across naming styles.

    ``chr1`` and ``1`` share a key, and so do ``chrM`` and ``MT``.

    Examples
    --------
    >>> chrom_key("chr1"), chrom_key("chrM")
    ('1', 'MT')
    """
    key = str(name).removeprefix("chr")
    return "MT" if key == "M" else key


def natural_order(name: object) -> tuple[int, int, str]:
    """Return a sort key that puts numbered chromosomes first, by number.

    X, Y, Z, W and the mitochondrion follow, then other names alphabetically.

    Examples
    --------
    >>> sorted(["chrX", "chr10", "chr2"], key=natural_order)
    ['chr2', 'chr10', 'chrX']
    """
    key = chrom_key(name)
    if key.isdigit():
        return (0, int(key), "")
    return (1, LETTER_ORDER.get(key, len(LETTER_ORDER)), key)


@dataclass(frozen=True)
class Stain:
    """Giemsa stain of a band, named as in UCSC tables.

    Parameters
    ----------
    name : str
        ``gneg``, ``gposNN``, ``acen``, ``gvar`` or ``stalk``.

    Examples
    --------
    >>> Stain("gpos33").density
    0.33
    """

    name: str

    @property
    def density(self) -> float | None:
        """Staining density from 0 (``gneg``) to 1 (``gpos100``).

        Bare ``gpos`` counts as 1. Stains that are not G-bands, such as
        ``acen``, ``gvar`` and ``stalk``, have no density.

        Examples
        --------
        >>> Stain("gneg").density, Stain("gpos75").density, Stain("acen").density
        (0.0, 0.75, None)
        """
        if self.name == "gneg":
            return 0.0
        if self.name.startswith("gpos"):
            return int(self.name.removeprefix("gpos") or 100) / 100
        return None


@dataclass(frozen=True)
class Band:
    """One cytoband, in 0-based half-open bp coordinates.

    Parameters
    ----------
    start, end : int
        Span of the band.
    name : str
        Band name, such as ``p36.33``.
    stain : Stain
        Giemsa stain of the band.

    Examples
    --------
    >>> Band(0, 2_300_000, "p36.33", Stain("gneg")).name
    'p36.33'
    """

    start: int
    end: int
    name: str
    stain: Stain


@dataclass(frozen=True)
class Chromosome:
    """One chromosome with its bands and centromere.

    Parameters
    ----------
    name : str
        Name as it appears in the source file, such as ``chr1``.
    length : int
        Length in bp.
    bands : tuple of Band
        Bands sorted by start. Empty when the assembly has none.
    centromere : tuple of int or None
        ``(start, end)`` of the centromere. ``(0, 0)`` marks a telocentric
        chromosome. None means the position is unknown.

    Examples
    --------
    >>> Chromosome("chr1", 1000, centromere=(400, 450)).telocentric
    False
    """

    name: str
    length: int
    bands: tuple[Band, ...] = ()
    centromere: tuple[int, int] | None = None

    @property
    def label(self) -> str:
        """Display name without the ``chr`` prefix.

        Examples
        --------
        >>> Chromosome("chrX", 1000).label
        'X'
        """
        return self.name.removeprefix("chr")

    @property
    def telocentric(self) -> bool:
        """Whether the centromere sits at position 0.

        Examples
        --------
        >>> Genome.bundled("mm39")["chr1"].telocentric
        True
        """
        return self.centromere == (0, 0)


@dataclass(frozen=True)
class Region:
    """A span on one chromosome.

    Parameters
    ----------
    chrom : str
        Chromosome name.
    start, end : int
        Span in bp.

    Examples
    --------
    >>> Region("chr17", 43_044_295, 43_125_483).end
    43125483
    """

    chrom: str
    start: int
    end: int

    @classmethod
    def parse(cls, text: str) -> Region:
        """Parse a locus such as ``chr17:43,044,295-43,125,483``.

        Commas are optional. The numbers are kept as written, with no shift
        between 1-based and 0-based coordinates.

        Parameters
        ----------
        text : str
            Locus in ``chrom:start-end`` form.

        Returns
        -------
        Region

        Raises
        ------
        ValueError
            If the text is not a locus or start exceeds end.

        Examples
        --------
        >>> Region.parse("chr17:43,044,295-43,125,483")
        Region(chrom='chr17', start=43044295, end=43125483)
        """
        match = re.fullmatch(r"\s*([^:\s]+):([\d,]+)-([\d,]+)\s*", text)
        if match is None:
            raise ValueError(f"not a region: {text!r}")
        chrom, start, end = match.groups()
        start, end = int(start.replace(",", "")), int(end.replace(",", ""))
        if start > end:
            raise ValueError(f"region start exceeds end: {text!r}")
        return cls(chrom, start, end)


class Genome:
    """An ordered set of chromosomes.

    Lookups match names across styles, so ``genome["1"]`` finds ``chr1``.

    Parameters
    ----------
    chromosomes : iterable of Chromosome
        Chromosomes in display order.

    Examples
    --------
    >>> genome = Genome([Chromosome("chr1", 1000), Chromosome("chr2", 500)])
    >>> genome["2"].length
    500
    """

    def __init__(self, chromosomes: Iterable[Chromosome]):
        self._by_key = {chrom_key(c.name): c for c in chromosomes}

    def __getitem__(self, name: object) -> Chromosome:
        try:
            return self._by_key[chrom_key(name)]
        except KeyError:
            raise KeyError(f"no chromosome {name!r} in genome") from None

    def __contains__(self, name: object) -> bool:
        return chrom_key(name) in self._by_key

    def __iter__(self) -> Iterator[Chromosome]:
        return iter(self._by_key.values())

    def __len__(self) -> int:
        return len(self._by_key)

    @property
    def autosomes(self) -> list[str]:
        """Names of the numbered chromosomes.

        Examples
        --------
        >>> Genome.bundled("mm39").autosomes[-1]
        'chr19'
        """
        return [c.name for c in self if chrom_key(c.name).isdigit()]

    @classmethod
    def bundled(cls, assembly: str) -> Genome:
        """Load a banded assembly that ships with the package.

        Parameters
        ----------
        assembly : str
            ``hg19``, ``hg38``, ``hs1`` (T2T-CHM13), ``mm10`` or ``mm39``.

        Returns
        -------
        Genome
            The primary chromosomes, 1 to 22 (19 in mouse), X and Y.

        Raises
        ------
        ValueError
            If the assembly is not bundled.

        Examples
        --------
        >>> hg38 = Genome.bundled("hg38")
        >>> len(hg38), hg38["chr1"].centromere
        (24, (121700000, 125100000))
        """
        if assembly not in BUNDLED:
            raise ValueError(f"no bundled assembly {assembly!r}; choose from {list(BUNDLED)}")
        with as_file(files("karyogram_plot") / "data" / BUNDLED[assembly]) as path:
            return cls.from_cytobands(path)

    @classmethod
    def from_cytobands(cls, path: str | PathLike, chroms: Iterable[str] | None = None) -> Genome:
        """Read a genome from a cytoband table.

        The centromere spans the ``acen`` bands. A chromosome with no ``acen``
        band and no numbered p band, as in mouse, is telocentric. A
        chromosome with p bands but no ``acen`` band has an unknown
        centromere.

        Parameters
        ----------
        path : str or PathLike
            Table in UCSC ``cytoBand`` form, gzipped or plain.
        chroms : iterable of str, optional
            Chromosomes to keep, in this order. Names match across styles.
            By default, the primary chromosomes in natural order. Primary
            means numbered, X, Y, Z or W.

        Returns
        -------
        Genome

        Raises
        ------
        KeyError
            If a chromosome in ``chroms`` is not in the file.

        Examples
        --------
        >>> from importlib.resources import files
        >>> path = files("karyogram_plot") / "data/hg38/cytoBand.txt.gz"
        >>> [c.name for c in Genome.from_cytobands(path, chroms=["X", "Y"])]
        ['chrX', 'chrY']
        """
        rows = cytobands.read_bands(path)
        chromosomes = []
        for name in _select(rows, chroms):
            bands = tuple(Band(s, e, n, Stain(stain)) for s, e, n, stain in sorted(rows[name]))
            chromosomes.append(Chromosome(name, bands[-1].end, bands, _centromere(bands)))
        return cls(chromosomes)

    @classmethod
    def from_sizes(
        cls,
        path: str | PathLike,
        centromeres: str | PathLike | None = None,
        chroms: Iterable[str] | None = None,
    ) -> Genome:
        """Read an unbanded genome from a ``.chrom.sizes`` or ``.fai`` file.

        Parameters
        ----------
        path : str or PathLike
            Sizes or FASTA index file.
        centromeres : str or PathLike, optional
            BED file of centromere spans. Names match across styles.
            Chromosomes it lacks get an unknown centromere.
        chroms : iterable of str, optional
            Chromosomes to keep, in this order. By default, the primary
            chromosomes in natural order, which drops unplaced and alt
            contigs.

        Returns
        -------
        Genome

        Raises
        ------
        KeyError
            If a chromosome in ``chroms`` is not in the file.

        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> path = Path(tempfile.mkdtemp()) / "x.chrom.sizes"
        >>> _ = path.write_text("chr2\\t500\\nchrUn_1\\t9\\nchr1\\t1000\\n")
        >>> [c.name for c in Genome.from_sizes(path)]
        ['chr1', 'chr2']
        """
        sizes = cytobands.read_sizes(path)
        spans = cytobands.read_centromeres(centromeres) if centromeres else {}
        spans = {chrom_key(name): span for name, span in spans.items()}
        return cls(
            Chromosome(name, sizes[name], centromere=spans.get(chrom_key(name)))
            for name in _select(sizes, chroms)
        )

    def concatenated(self, gap: float = 0) -> LinearGenome:
        """Lay the chromosomes end to end on one axis.

        Parameters
        ----------
        gap : float, default 0
            Space in bp between neighboring chromosomes.

        Returns
        -------
        LinearGenome

        Examples
        --------
        >>> linear = Genome.bundled("hg38").concatenated(gap=5_000_000)
        >>> linear.offset("chr2")
        253956422.0
        """
        return LinearGenome(self, gap)


class LinearGenome:
    """Chromosomes of a genome laid end to end, for genome-wide plots.

    Parameters
    ----------
    genome : Genome
        Chromosomes in axis order.
    gap : float, default 0
        Space in bp between neighboring chromosomes.

    Examples
    --------
    >>> linear = LinearGenome(Genome([Chromosome("1", 100), Chromosome("2", 50)]), gap=10)
    >>> linear.length
    160.0
    """

    def __init__(self, genome: Genome, gap: float = 0):
        self.genome = genome
        starts = np.cumsum([0.0] + [c.length + gap for c in genome])
        self._offsets = {chrom_key(c.name): start for c, start in zip(genome, starts, strict=False)}
        self.length = float(starts[-1] - gap)

    def offset(self, chrom: str) -> float:
        """Return the axis position where a chromosome starts.

        Examples
        --------
        >>> LinearGenome(Genome([Chromosome("1", 100), Chromosome("2", 50)])).offset("chr2")
        100.0
        """
        return float(self._offsets[chrom_key(chrom)])

    def position(self, chrom: object, pos: object) -> np.ndarray:
        """Map chromosome positions onto the concatenated axis.

        Parameters
        ----------
        chrom : array-like of str or str
            Chromosome of each position. Names match across styles.
        pos : array-like of int or int
            Position in bp on that chromosome.

        Returns
        -------
        numpy.ndarray of float
            Axis positions. NaN where the chromosome is not in the genome, so
            matplotlib skips those points.

        Examples
        --------
        >>> linear = LinearGenome(Genome([Chromosome("chr1", 100), Chromosome("chr2", 50)]))
        >>> linear.position(["chr2", "1", "chrM"], [5, 7, 3])
        array([105.,   7.,  nan])
        """
        chroms = np.asarray(chrom).astype(str)
        names, index = np.unique(chroms.ravel(), return_inverse=True)
        offsets = np.array([self._offsets.get(chrom_key(n), np.nan) for n in names])
        return offsets[index].reshape(chroms.shape) + np.asarray(pos, dtype=float)


def _select(names: Iterable[str], chroms: Iterable[str] | None) -> list[str]:
    if chroms is None:
        return sorted((n for n in names if PRIMARY.fullmatch(n)), key=natural_order)
    by_key = {chrom_key(n): n for n in names}
    missing = [c for c in chroms if chrom_key(c) not in by_key]
    if missing:
        raise KeyError(f"chromosomes not in file: {missing}")
    return [by_key[chrom_key(c)] for c in chroms]


def _centromere(bands: tuple[Band, ...]) -> tuple[int, int] | None:
    acen = [b for b in bands if b.stain.name == "acen"]
    if acen:
        return (min(b.start for b in acen), max(b.end for b in acen))
    if not any(re.match(r"p\d", b.name) for b in bands):
        return (0, 0)
    return None
