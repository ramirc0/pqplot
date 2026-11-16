"""Column arrays from any table, grouped by chromosome."""

from __future__ import annotations

import numpy as np

from karyogram_plot.genome import chrom_key


def column(table: object, spec: object) -> np.ndarray:
    """Return one column of a table as a NumPy array.

    Parameters
    ----------
    table : DataFrame, dict or None
        Any table indexed by column name, such as a polars or pandas
        DataFrame, a pyarrow Table or a dict of sequences.
    spec : str or array-like
        Column name, or the values themselves.

    Returns
    -------
    numpy.ndarray

    Raises
    ------
    ValueError
        If `spec` names a column but there is no table.

    Examples
    --------
    >>> column({"pos": [5, 7]}, "pos")
    array([5, 7])
    >>> column(None, [1.5])
    array([1.5])
    """
    if not isinstance(spec, str):
        return np.asarray(spec)
    if table is None:
        raise ValueError(f"column {spec!r} needs a table; pass data or the values themselves")
    return np.asarray(table[spec])


def rows_by_chrom(chroms: np.ndarray) -> dict[str, np.ndarray]:
    """Group row indices by chromosome.

    Parameters
    ----------
    chroms : numpy.ndarray
        Chromosome name of each row. Names match across styles.

    Returns
    -------
    dict of str to numpy.ndarray
        Row indices in input order, keyed by `karyogram_plot.genome.chrom_key`.

    Examples
    --------
    >>> {k: v.tolist() for k, v in rows_by_chrom(np.array(["chr2", "1", "chr1"])).items()}
    {'1': [1, 2], '2': [0]}
    """
    if len(chroms) == 0:
        return {}
    names, index = np.unique(chroms.astype(str), return_inverse=True)
    keys, key_index = np.unique([chrom_key(n) for n in names], return_inverse=True)
    rows = key_index[index]
    order = np.argsort(rows, kind="stable")
    groups = np.split(order, np.cumsum(np.bincount(rows, minlength=len(keys)))[:-1])
    return dict(zip(keys.tolist(), groups, strict=True))
